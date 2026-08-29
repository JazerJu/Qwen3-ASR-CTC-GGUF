#!/usr/bin/env python3
"""最小可运行示例：转写 + 强制对齐出字级时间戳。

    pip install torch transformers qwen-asr safetensors soundfile
    python example.py audio.wav

编码器默认从 HuggingFace 拉 Qwen/Qwen3-ASR-1.7B。离线环境把本地路径给
QWEN3_ASR_ENCODER 环境变量：

    QWEN3_ASR_ENCODER=/path/to/Qwen3-ASR-1.7B python example.py audio.wav
"""
import os
import sys

import numpy as np
import soundfile as sf
import torch

from modeling_ctc import Qwen3CtcAsr, FRAME_SHIFT_SEC


def load(path, sr=16000):
    wav, orig = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav)
    if orig != sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, orig, sr)
    return wav


def ctc_viterbi(logp, targets, blank):
    """CTC 受限格上的 Viterbi 强制对齐，返回每帧所处的扩展状态。"""
    T, L, S = logp.shape[0], len(targets), 2 * len(targets) + 1
    if T < L:
        raise ValueError(f"帧数 {T} < token 数 {L}，无合法路径")
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = targets
    emit = logp[:, ext]
    NEG = -1e30
    alpha = np.full(S, NEG)
    alpha[0] = emit[0, 0]
    if S > 1:
        alpha[1] = emit[0, 1]
    skip = np.zeros(S, dtype=bool)
    for s in range(2, S):
        if s % 2 == 1 and targets[s // 2] != targets[s // 2 - 1]:
            skip[s] = True
    bp = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        p1 = np.concatenate(([NEG], alpha[:-1]))
        p2 = np.where(skip, np.concatenate(([NEG, NEG], alpha[:-2])), NEG)
        cand = np.stack([alpha, p1, p2])
        ch = cand.argmax(axis=0)
        alpha = cand[ch, np.arange(S)] + emit[t]
        bp[t] = ch
    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    path = np.zeros(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t][s])
    return path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    wav = load(sys.argv[1])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = os.environ.get("QWEN3_ASR_ENCODER", "Qwen/Qwen3-ASR-1.7B")
    asr = Qwen3CtcAsr(".", encoder_id=encoder, device=device)

    text = asr.transcribe([wav])[0]
    print(f"转写: {text}")

    # 强制对齐：拿刚才的转写当参考序列，反查每个 token 的发射帧
    enc = asr.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    q2c = {}  # 原始 id -> 紧凑 id
    for c, q in asr.compact_to_source.items():
        q2c[q] = c
    ids, offs = [], []
    for i, t in zip(enc["input_ids"], enc["offset_mapping"]):
        if i in q2c:
            ids.append(q2c[i])
            offs.append(t)
    if not ids:
        return

    lp, lens = asr.log_probs([wav])
    logp = lp[0, :int(lens[0])].cpu().numpy()
    path = ctc_viterbi(logp, ids, asr.blank_id)

    print(f"\n字级时间戳（帧移 {FRAME_SHIFT_SEC*1000:.1f} ms）:")
    for k, (a, b) in enumerate(offs):
        idx = np.nonzero(path == 2 * k + 1)[0]
        if len(idx) == 0:
            continue
        t0, t1 = idx[0] * FRAME_SHIFT_SEC, (idx[-1] + 1) * FRAME_SHIFT_SEC
        print(f"  {text[a:b]!r:<12} {t0:6.2f} - {t1:6.2f} s")
    print("\n注意：CTC 是尖峰式发射，词起始点系统性偏晚约 100 ms、结束偏早约 80 ms"
          "（对 MFA 真值实测）。要精确时间戳请减掉这个常数偏置。")


if __name__ == "__main__":
    main()
