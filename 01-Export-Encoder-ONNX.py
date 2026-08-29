#!/usr/bin/env python3
"""Step 1: export Qwen3-ASR audio tower to ONNX.

Fixed 30s bucket (mel 3000 frames -> 390 out frames) + runtime `feature_length`
scalar input. The original forward's data-dependent ops (chunk split by
tolist, pad_sequence, boolean unpad, cu_seqlens python loop) are replaced by a
static-shape equivalent that keeps padded frames and masks them out of
attention; the caller slices output[:qwen3_output_lengths(feature_length)].

Equivalence for real frames is exact: conv padding values, chunk positions and
per-layer attention all match the original batch=1 path (validated in
02b-Validate-Encoder.py, gate cosine >= 0.999 on lengths != trace input).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
import torch.nn.functional as F

import qwen3_compat  # noqa: F401  — must precede qwen_asr import (init-order patch)
from qwen_asr import Qwen3ASRModel

from modeling_ctc import patch_qwen3_attention_mask
from qwen3_compat import repair_positional_embedding, valid_frames

QWEN3_DIR = "/data/推理框架/asr-onnx/Qwen3-ASR-HF"
MEL_FRAMES = 3000


class ExportTower(torch.nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.tower = tower

    def forward(self, input_features, feature_length):
        t = self.tower
        n_chunks = input_features.shape[-1] // 100
        x = input_features.T.reshape(n_chunks, 100, input_features.shape[0]).permute(0, 2, 1).unsqueeze(1)
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
        h = t.proj2(t.act(t.proj1(h)))
        return h


def main():
    patch_qwen3_attention_mask()
    m = Qwen3ASRModel.from_pretrained(QWEN3_DIR, dtype=torch.float32, device_map=None)
    tower = m.model.thinker.audio_tower.eval()
    repair_positional_embedding(tower)
    wrapper = ExportTower(tower).eval()

    out = HERE / "model" / "Qwen3-ASR-Encoder.fp32.onnx"
    dummy_mel = torch.randn(128, MEL_FRAMES)
    dummy_len = torch.tensor([2800], dtype=torch.long)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy_mel, dummy_len), str(out),
            input_names=["input_features", "feature_length"], output_names=["enc_output"],
            opset_version=18, do_constant_folding=True, dynamo=False,
        )
    print(f"exported {out.name} ({out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
