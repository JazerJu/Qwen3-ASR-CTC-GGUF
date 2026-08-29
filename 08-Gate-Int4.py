#!/usr/bin/env python3
"""Step 4 gate: int4 chain vs fp32 chain greedy text diff <= 1% on real audio.

Also serves as the QwenEngine prototype: mel -> pad(3000) -> encoder ->
slice(valid) -> CTC -> byte-collapse.
"""

import base64
import glob
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import soundfile as sf

from qwen3_compat import valid_frames

sys.path.insert(0, str(HERE))
import export_config as C  # noqa: E402

QWEN3_DIR = str(C.QWEN3_DIR)
BLANK, UNK = 72466, 72467


def load_tokens(path):
    id2bytes = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        b64, idx = line.rsplit("\t", 1)
        id2bytes[int(idx)] = base64.b64decode(b64)
    return id2bytes


def collapse(ids, id2bytes):
    out, prev = [], None
    for t in ids:
        t = int(t)
        if t == prev:
            continue
        prev = t
        if t not in (BLANK, UNK):
            out.append(t)
    return b"".join(id2bytes[t] for t in out).decode("utf-8", errors="replace")


class MiniEngine:
    def __init__(self, quantized: bool):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        tag = "q4" if quantized else "fp32"
        self.enc = ort.InferenceSession(str(HERE / f"model/Qwen3-ASR-Encoder.{tag}.onnx"),
                                        providers=["CPUExecutionProvider"])
        self.ctc = ort.InferenceSession(str(HERE / f"model/Qwen3-ASR-CTC.{tag}.onnx"),
                                        providers=["CPUExecutionProvider"])
        self.fe = WhisperFeatureExtractor.from_pretrained(QWEN3_DIR)
        self.id2bytes = load_tokens(HERE / "model/qwen-ctc-tokens.txt")

    def __call__(self, audio):
        mel = self.fe(audio, sampling_rate=16000, padding=False, return_tensors="np").input_features[0]
        T = mel.shape[1]
        padded = np.zeros((128, 3000), dtype=np.float32)
        padded[:, :T] = mel
        enc = self.enc.run(None, {"input_features": padded,
                                  "feature_length": np.array([T], np.int64)})[0]
        valid = int(valid_frames(torch.tensor([T]))[0])
        logits = self.ctc.run(None, {"enc_output": enc[:valid][None].astype(np.float32)})[0]
        return collapse(np.argmax(logits[0], -1), self.id2bytes)


def char_diff(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    import editdistance

    return editdistance.eval(a, b) / max(len(a), len(b), 1)


def main():
    files = sorted(glob.glob("/data/datasets/librispeech/LibriSpeech/test-clean/**/*.flac", recursive=True))[:30]
    assert files, "librispeech test-clean not found"

    fp32 = MiniEngine(quantized=False)
    q4 = MiniEngine(quantized=True)

    diffs = []
    for f in files:
        audio, _ = sf.read(f, dtype="float32")
        t1, t2 = fp32(audio), q4(audio)
        d = char_diff(t1, t2)
        diffs.append(d)
        if d > 0.01:
            print(f"[{Path(f).name}] diff={d:.3f}\n  fp32: {t1[:70]}\n  q4:   {t2[:70]}")
    mean = float(np.mean(diffs))
    worst = float(np.max(diffs))
    print(f"\nmean char diff = {mean*100:.2f}%, worst = {worst*100:.2f}%  (gate <= 1%)")
    print("sample output:", q4(sf.read(files[0], dtype="float32")[0])[:80])
    sys.exit(0 if mean <= 0.01 else 1)


if __name__ == "__main__":
    main()
