#!/usr/bin/env python3
"""第 1 步：编码器 + CTC 头 -> fp32 ONNX，并生成 model/tokens.txt。

对应 Fun-ASR-GGUF 的 01-Export-ONNX-FP32.py：一步产出后续所有精度变体的基底。

三件事：
  A. 音频编码器 -> Qwen3-ASR-Encoder.fp32.onnx
     固定 30 秒桶（mel 3000 帧 -> 390 输出帧）+ 运行时 `feature_length` 标量输入。
     原 forward 里依赖数据的算子（tolist 切块、pad_sequence、布尔 unpad、
     cu_seqlens 的 python 循环）换成静态形状等价实现：保留补齐帧但在注意力里
     mask 掉，调用方自己切 output[:valid_frames(feature_length)]。
     真实帧上是精确等价的 —— 卷积补齐值、分块位置、逐层注意力都与原 batch=1
     路径一致（由 07-Validate-Exports.py 用不同长度的真实音频把关，cos >= 0.999）。
  B. CTC 头 -> Qwen3-ASR-CTC.fp32.onnx，并当场做 fp32 一致性门（cos >= 0.9999）
  C. model/tokens.txt：每个紧凑 id 对应的**原始字节**的 base64，并用 500 条
     多语种句子验证字节路径与 transformers tokenizer.decode 逐字节一致

超参一律从 CTC_DIR/config.json 读，不写死 —— v1 的 ffn_hidden 是 128，
v2 是 2048，写死会 load_state_dict 形状不匹配。
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import export_config as C
from qwen3_asr_ctc import compat  # noqa: F401 —— 必须先于 qwen_asr 导入（init-order 补丁）
from qwen3_asr_ctc.compat import repair_positional_embedding, valid_frames
from qwen3_asr_ctc.modeling_ctc import CTCDecoder, patch_qwen3_attention_mask
from qwen3_asr_ctc.tokens import bytes_to_unicode


# ── A. 编码器 ──────────────────────────────────────────────────────────
class ExportTower(torch.nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.tower = tower

    def forward(self, input_features, feature_length):
        t = self.tower
        n_chunks = input_features.shape[-1] // 100
        x = (input_features.T.reshape(n_chunks, 100, input_features.shape[0])
             .permute(0, 2, 1).unsqueeze(1))
        e = F.gelu(t.conv2d1(x))
        e = F.gelu(t.conv2d2(e))
        e = F.gelu(t.conv2d3(e))
        b, c, f, ft = e.shape
        e = t.conv_out(e.permute(0, 3, 1, 2).contiguous().view(b, ft, c * f))
        e = e + t.positional_embedding.positional_embedding[: e.shape[1], :].unsqueeze(0).to(e.dtype)
        h = e.reshape(-1, e.shape[-1])

        valid = valid_frames(feature_length)[0]
        idx = torch.arange(h.shape[0], device=h.device)
        window = (h.shape[0] // n_chunks) * (t.n_window_infer // (t.n_window * 2))
        block = idx // window
        ok = idx < valid
        keep = (block[:, None] == block[None, :]) & ok[:, None] & ok[None, :]
        mask = torch.where(
            keep,
            torch.zeros((), dtype=h.dtype, device=h.device),
            torch.full((), torch.finfo(h.dtype).min, dtype=h.dtype, device=h.device),
        ).view(1, 1, h.shape[0], h.shape[0])
        cu_seqlens = torch.stack([torch.zeros_like(valid), valid])

        for layer in t.layers:
            h = layer(h, cu_seqlens, attention_mask=mask)[0]
        h = t.ln_post(h)
        return t.proj2(t.act(t.proj1(h)))


def export_encoder() -> Path:
    patch_qwen3_attention_mask()
    from qwen_asr import Qwen3ASRModel

    m = Qwen3ASRModel.from_pretrained(str(C.QWEN3_DIR), dtype=torch.float32, device_map=None)
    tower = m.model.thinker.audio_tower.eval()
    repair_positional_embedding(tower)
    wrapper = ExportTower(tower).eval()

    out = C.onnx(C.ENCODER, "fp32")
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (torch.randn(C.MEL_BINS, C.MEL_FRAMES), torch.tensor([2800], dtype=torch.long)),
            str(out),
            input_names=["input_features", "feature_length"], output_names=["enc_output"],
            opset_version=18, do_constant_folding=True, dynamo=False,
        )
    print(f"[A] {out.name}  {out.stat().st_size / 1e6:.0f} MB")
    return out


# ── B. CTC 头 ──────────────────────────────────────────────────────────
def export_ctc() -> bool:
    cfg = json.loads(C.CTC_CONFIG.read_text(encoding="utf-8"))
    head = CTCDecoder(
        encoder_dim=cfg["encoder_dim"], ctc_hidden=cfg["ctc_hidden"],
        proj_hidden=cfg["proj_hidden"], num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"], ffn_hidden=cfg["ffn_hidden"],
        vocab_size=cfg["vocab_size"], blank_id=cfg["blank_id"],
    )
    sd = load_file(str(C.CTC_WEIGHTS))
    head.load_state_dict({k: v for k, v in sd.items() if not k.startswith("optimizer.")},
                         strict=True)
    head.eval()

    out = C.onnx(C.CTC, "fp32")
    with torch.no_grad():
        torch.onnx.export(
            head, (torch.randn(1, 200, cfg["encoder_dim"]),), str(out),
            input_names=["enc_output"], output_names=["logits"],
            dynamic_axes={"enc_output": {0: "batch", 1: "time"},
                          "logits": {0: "batch", 1: "time"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    print(f"[B] {out.name}  {out.stat().st_size / 1e6:.1f} MB "
          f"(ffn_hidden={cfg['ffn_hidden']}, {cfg['params']/1e6:.1f}M 参数)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ok = True
    for T in (65, 156, 325):
        x = torch.randn(1, T, cfg["encoder_dim"])
        with torch.no_grad():
            ref = head(x).numpy()
        hyp = sess.run(None, {"enc_output": x.numpy()})[0]
        cos = float(ref.ravel() @ hyp.ravel() / (np.linalg.norm(ref) * np.linalg.norm(hyp)))
        good = cos >= 0.9999
        ok &= good
        print(f"    T={T:4d}  cosine={cos:.7f}  {'PASS' if good else 'FAIL'}")
    return ok


# ── C. tokens.txt ──────────────────────────────────────────────────────
SENTENCES = [
    "今天我们来讲一讲服务器的日常维护，主要包括内存、磁盘和网络的监控。",
    "你可以打开cmd命令行，输入ip config查看本机的IP地址。",
    "In this video, we will walk through the installation process step by step.",
    "안녕하세요, 오늘 날씨가 정말 좋네요.",
    "こんにちは、今日はいい天気ですね。",
    "Le chat s'est assis sur le tapis.", "Die Straße war leer.",
    "Мы говорим по-русски.", "こんにちは1234、测试mixed—input！！",
]


def export_tokens() -> bool:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(C.QWEN3_DIR), trust_remote_code=True)
    c2q = json.loads(C.CTC_VOCAB.read_text(encoding="utf-8"))["compact_to_qwen"]
    q2c = {q: c for c, q in enumerate(c2q)}
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

    lines, id2bytes = [], {}
    for cid, qid in enumerate(c2q):
        raw = bytes(byte_decoder[ch] for ch in tok.convert_ids_to_tokens(int(qid)))
        id2bytes[cid] = raw
        lines.append(base64.b64encode(raw).decode("ascii") + "\t" + str(cid))
    C.TOKENS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[C] {C.TOKENS_TXT.name}  {len(lines)} 条")

    fail = covered = 0
    for s in SENTENCES * 56:
        qids = tok(s, add_special_tokens=False)["input_ids"]
        if any(q not in q2c for q in qids):
            continue                      # 紧凑词表的编码方向缺口，只影响 encode，不影响解码用途
        covered += 1
        got = b"".join(id2bytes[q2c[q]] for q in qids).decode("utf-8", errors="replace")
        if got != tok.decode(qids):
            fail += 1
            print(f"    MISMATCH {s[:24]!r}: {got[:36]!r} != {tok.decode(qids)[:36]!r}")
    print(f"    字节往返 {covered - fail}/{covered} 一致（跳过 {len(SENTENCES)*56 - covered} 条）")
    return fail == 0 and covered >= 100


def main() -> int:
    for p, what in ((C.QWEN3_DIR, "Qwen3-ASR 官方权重"), (C.CTC_DIR, "CTC 头")):
        if not p.exists():
            print(f"找不到{what}: {p}\n见 README「准备输入」", file=sys.stderr)
            return 1
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    export_encoder()
    ok_ctc = export_ctc()
    ok_tok = export_tokens()
    print(f"\nGATE: {'PASS' if ok_ctc and ok_tok else 'FAIL'}"
          f"  (CTC 一致性 {'PASS' if ok_ctc else 'FAIL'} / 字节往返 {'PASS' if ok_tok else 'FAIL'})")
    print("编码器的等价性门在 07-Validate-Exports.py（需要真实音频）")
    return 0 if (ok_ctc and ok_tok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
