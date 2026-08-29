"""Qwen3-ASR CTC head — 独立推理实现（不依赖训练仓库）。

这个 CTC 头接在冻结的 Qwen3-ASR-1.7B 音频编码器后面，做首遍(first-pass)转写。
编码器本身不含在本仓库里，运行时从 Qwen/Qwen3-ASR-1.7B 加载。

三个必须照做、否则结果静默出错的点：

1. **编码器输入是时间维拼接的 2D 张量** `[128, ΣT_mel]` 加 `feature_lens`，
   不是常见的 `[B, 128, T]`。传 3D 会直接报 split_with_sizes 的错。

2. **帧率是 13 fps（76.9 ms/帧），不是 Whisper/GLM 的 50 fps。**
   编码器用 3 层 stride-2 的 conv2d 做 8 倍降采样，但官方长度公式是
   "每 100 个 mel 帧出 13 帧"。沿用 50 fps 的算法会把可用帧数高估 4 倍，
   CTC 以为容量充足，实际 log_probs 只有 1/4 长 —— loss 看着能降，对齐全错。

3. **必须打 attention mask 补丁。** qwen-asr 0.0.6 里 `_prepare_attention_mask`
   定义了却从来没被调用，`cu_seq_lens_q/k` 只有 flash_attention_2 后端认。
   不打补丁时，同一条音频单条推理与批推理的余弦相似度只有 0.81–0.88；
   打完是 0.9998+。CUDA 上同样中招，不是昇腾特有的问题。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOP_LENGTH = 160          # 16 kHz 下 10 ms 一个 mel 帧
FRAME_SHIFT_SEC = 1.0 / 13.0


# ── 编码器输出长度 ────────────────────────────────────────────────────
def qwen3_output_lengths(mel_lengths: torch.Tensor) -> torch.Tensor:
    """mel 帧数 -> encoder 输出帧数。复刻 Qwen3-ASR 官方公式，
    已对 9 次真实前向逐条核对，9/9 精确吻合。"""
    leave = mel_lengths % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (mel_lengths // 100) * 13


# ── attention mask 补丁 ───────────────────────────────────────────────
_PATCHED = False


def patch_qwen3_attention_mask() -> bool:
    """给 Qwen3ASRAudioEncoder 补上块对角 attention 掩码。幂等。

    没有它，一个 batch 里的不同语句会互相看见 —— 因为编码器把整批在时间维
    拼成一条，而 cu_seqlens 只在 flash_attention_2 路径上被消费。
    """
    global _PATCHED
    if _PATCHED:
        return False
    from qwen_asr.core.transformers_backend import modeling_qwen3_asr as M
    Layer = M.Qwen3ASRAudioEncoderLayer
    if getattr(Layer, "_mask_patched", False):
        _PATCHED = True
        return False
    orig_forward = Layer.forward

    def forward(self, hidden_states, cu_seqlens, attention_mask=None, **kw):
        if attention_mask is None and cu_seqlens is not None:
            n = hidden_states.shape[0]
            mask = torch.full((1, 1, n, n), torch.finfo(hidden_states.dtype).min,
                              device=hidden_states.device, dtype=hidden_states.dtype)
            cs = cu_seqlens.tolist()
            for i in range(1, len(cs)):
                mask[..., cs[i - 1]:cs[i], cs[i - 1]:cs[i]] = 0
            attention_mask = mask
        return orig_forward(self, hidden_states, cu_seqlens,
                            attention_mask=attention_mask, **kw)

    Layer.forward = forward
    Layer._mask_patched = True
    _PATCHED = True
    return True


# ── CTC 头 ────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=512, ffn_hidden=128, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_o = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn_w1 = nn.Linear(hidden_size, ffn_hidden)
        self.ffn_w2 = nn.Linear(ffn_hidden, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.linear_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.linear_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.head_dim ** -0.5, dim=-1)
        attn = self.dropout(attn)
        out = self.linear_o((attn @ v).transpose(1, 2).contiguous().view(B, T, C))
        x = self.norm1(x + self.dropout(out))
        ffn = self.ffn_w2(self.dropout(F.gelu(self.ffn_w1(x))))
        return self.norm2(x + self.dropout(ffn))


class CTCDecoder(nn.Module):
    def __init__(self, encoder_dim=2048, ctc_hidden=512, proj_hidden=2048,
                 num_blocks=5, num_heads=8, ffn_hidden=128,
                 vocab_size=72468, dropout=0.0, blank_id=72466):
        super().__init__()
        self.blank_id = blank_id
        self.linear1 = nn.Linear(encoder_dim, proj_hidden)
        self.linear2 = nn.Linear(proj_hidden, ctc_hidden)
        self.blocks = nn.ModuleList([
            TransformerBlock(ctc_hidden, ffn_hidden, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(ctc_hidden)
        self.ctc_lo = nn.Linear(ctc_hidden, vocab_size)

    def forward(self, encoder_out, use_blocks=True):
        x = F.gelu(self.linear1(encoder_out))
        x = F.gelu(self.linear2(x))
        if use_blocks:
            for block in self.blocks:
                x = block(x)
        return self.ctc_lo(self.layer_norm(x))


# ── 端到端封装 ────────────────────────────────────────────────────────
class Qwen3CtcAsr:
    """编码器 + CTC 头 + 紧凑词表反查，一个类跑完首遍转写。"""

    def __init__(self, repo_dir, encoder_id=None,
                 device="cuda", dtype=torch.bfloat16):
        # 离线环境把本地目录给 QWEN3_ASR_ENCODER，省得去连 HuggingFace
        encoder_id = (encoder_id or os.environ.get("QWEN3_ASR_ENCODER")
                      or "Qwen/Qwen3-ASR-1.7B")
        repo_dir = Path(repo_dir)
        self.cfg = json.loads((repo_dir / "config.json").read_text())
        self.device = torch.device(device)

        vc = json.loads((repo_dir / "vocab_compact.json").read_text())
        c2q = vc["compact_to_qwen"]
        self.compact_to_source = (dict(enumerate(c2q)) if isinstance(c2q, list)
                                  else {int(k): v for k, v in c2q.items()})
        self.blank_id = self.cfg["blank_id"]
        self.unk_id = self.cfg["unk_id"]

        from safetensors.torch import load_file
        self.head = CTCDecoder(
            encoder_dim=self.cfg["encoder_dim"], ctc_hidden=self.cfg["ctc_hidden"],
            proj_hidden=self.cfg["proj_hidden"], num_blocks=self.cfg["num_blocks"],
            num_heads=self.cfg["num_heads"], ffn_hidden=self.cfg["ffn_hidden"],
            vocab_size=self.cfg["vocab_size"], dropout=0.0, blank_id=self.blank_id,
        )
        self.head.load_state_dict(load_file(str(repo_dir / "ctc_head.safetensors")))
        self.head = self.head.to(self.device, dtype=torch.float32).eval()

        patch_qwen3_attention_mask()
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(encoder_id, dtype=dtype, device_map=None)
        self.encoder = model.model.thinker.audio_tower.to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # AutoProcessor 对这个模型返回的是 Qwen2TokenizerFast，取不到特征提取器
        from transformers import AutoTokenizer, WhisperFeatureExtractor
        self.fe = WhisperFeatureExtractor.from_pretrained(encoder_id)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_id, trust_remote_code=True)
        self.dtype = dtype

    def _features(self, waveforms):
        mels, lens = [], []
        for w in waveforms:
            f = self.fe(w.numpy(), sampling_rate=self.fe.sampling_rate,
                        padding=False, return_tensors="pt").input_features[0]
            mels.append(f)
            lens.append(f.shape[-1])
        return torch.cat(mels, dim=-1), torch.tensor(lens, dtype=torch.long)

    @torch.no_grad()
    def log_probs(self, waveforms):
        """-> (log_probs [B,T,V], input_lengths [B])，16 kHz 单声道 float32 波形。"""
        feats, feat_lens = self._features(waveforms)
        out_lens = qwen3_output_lengths(feat_lens).clamp(min=1)
        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            out = self.encoder(feats.to(self.device),
                               feature_lens=feat_lens.to(self.device))
            flat = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            pieces, off = [], 0
            for n in out_lens.tolist():
                pieces.append(flat[off:off + n])
                off += n
            hidden = nn.utils.rnn.pad_sequence(pieces, batch_first=True)
            logits = self.head(hidden.float(), use_blocks=True)
        return torch.log_softmax(logits.float(), dim=-1), out_lens

    def transcribe(self, waveforms):
        """贪心解码。返回文本列表。"""
        lp, lens = self.log_probs(waveforms)
        pred = lp.argmax(dim=-1).cpu()
        texts = []
        for b in range(pred.shape[0]):
            ids, prev = [], -1
            for t in pred[b, :int(lens[b])].tolist():
                if t != prev and t != self.blank_id:
                    ids.append(t)
                prev = t
            src = [self.compact_to_source[i] for i in ids
                   if i != self.unk_id and i in self.compact_to_source]
            texts.append(self.tokenizer.decode(src, skip_special_tokens=True))
        return texts

    def frame_time(self, frame_index):
        """帧下标 -> 秒。做时间戳时用这个，别自己乘 8*10ms。"""
        return frame_index * FRAME_SHIFT_SEC
