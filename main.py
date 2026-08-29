#!/usr/bin/env python3
"""Qwen3-ASR-CTC inference CLI: any audio (any ffmpeg format), CTC first-pass
with word-level timestamps.

13 fps frames; per QWEN3-CTC-导出与推理指示.md §7 the raw CTC spikes sit
~100 ms late on word starts and ~78.5 ms early on word ends — both corrected
below (measured against MFA ground truth on the training side).

The LLM second pass (decoder q5_k_m GGUF) is exported and loadable but not
yet wired into this CLI — see 06-Export-Decoder-GGUF.py.

Examples:
    python main.py clip.wav
    python main.py input.mp3 --srt out.srt --cpu
"""

import argparse
import base64
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

BLANK_ID, UNK_ID = 72466, 72467
FRAME_SEC = 1.0 / 13.0
START_BIAS = 0.100
END_BIAS = 0.0785


def to_wav16k(src: Path, tmpdir: Path) -> Path:
    dst = tmpdir / "input16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        check=True, capture_output=True,
    )
    return dst


class QwenCTCEngine:
    def __init__(self, use_gpu: bool = True, quantized: bool = True):
        import numpy as np
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        tag = "q4" if quantized else "fp32"
        providers = ["CPUExecutionProvider"]
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.enc = ort.InferenceSession(str(HERE / f"model/Qwen3-ASR-Encoder.{tag}.onnx"),
                                        providers=providers)
        self.ctc = ort.InferenceSession(str(HERE / f"model/Qwen3-ASR-CTC.{tag}.onnx"),
                                        providers=providers)
        self.fe = WhisperFeatureExtractor.from_pretrained(str(HERE / "preprocessor"))
        self.id2bytes = {}
        for line in open(HERE / "model/qwen-ctc-tokens.txt", encoding="utf-8"):
            line = line.rstrip("\n")
            if line:
                b64, idx = line.rsplit("\t", 1)
                self.id2bytes[int(idx)] = base64.b64decode(b64)
        self.np = np

    def transcribe(self, audio):
        np = self.np
        mel = self.fe(audio, sampling_rate=16000, padding=False,
                      return_tensors="np").input_features[0]
        T = mel.shape[1]
        padded = np.zeros((128, 3000), dtype=np.float32)
        padded[:, :T] = mel
        enc = self.enc.run(None, {"input_features": padded,
                                  "feature_length": np.array([T], np.int64)})[0]
        full, leave = divmod(T, 100)
        valid = full * 13 + (0 if leave == 0 else (leave - 1) // 8 + 1)
        logits = self.ctc.run(None, {"enc_output": enc[:valid][None].astype(np.float32)})[0]
        ids = np.argmax(logits[0], axis=-1)

        tokens = []
        prev = None
        for frame, t in enumerate(ids):
            t = int(t)
            if t == prev:
                continue
            prev = t
            if t not in (BLANK_ID, UNK_ID):
                tokens.append((t, frame))
        return tokens

    def decode(self, tokens):
        return b"".join(self.id2bytes[t] for t, _ in tokens).decode("utf-8", errors="replace")


def word_timestamps(engine, tokens):
    """[(unit, start_s, end_s)] — CJK per char, latin runs merged to words."""
    import re

    units = []
    for tid, frame in tokens:
        text = engine.id2bytes[tid].decode("utf-8", errors="replace")
        start = max(frame * FRAME_SEC - START_BIAS, 0.0)
        end = max((frame + 1) * FRAME_SEC - END_BIAS, start)
        for ch in text:
            units.append([ch, start, end])

    merged = []
    buf, buf_s, buf_e = [], None, None
    for ch, s, e in units:
        if re.match(r"[A-Za-z]", ch):
            if buf_s is None:
                buf_s = s
            buf.append(ch)
            buf_e = e
        else:
            if buf:
                merged.append(("".join(buf), buf_s, buf_e))
                buf, buf_s = [], None
            if ch.strip():
                merged.append((ch, s, e))
    if buf:
        merged.append(("".join(buf), buf_s, buf_e))
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="input audio/video, any ffmpeg format, <=30s")
    ap.add_argument("--srt", default=None, help="write word-timestamped SRT")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    src = Path(args.audio).expanduser()
    if not src.exists():
        raise SystemExit(f"input not found: {src}")

    eng = QwenCTCEngine(use_gpu=not args.cpu, quantized=not args.fp32)
    with tempfile.TemporaryDirectory() as td:
        wav = to_wav16k(src, Path(td))
        import soundfile as sf

        audio, sr = sf.read(wav, dtype="float32")
        duration = len(audio) / sr
        t0 = time.perf_counter()
        tokens = eng.transcribe(audio)
        wall = time.perf_counter() - t0

    text = eng.decode(tokens)
    print(f"[qwen3-ctc] {text}")
    print(f"[stats] {duration:.1f}s audio -> {wall*1000:.0f}ms (RTF {wall/duration:.3f})")

    if args.srt:
        from glm_free_srt import write_srt

        write_srt(word_timestamps(eng, tokens), args.srt)
        print(f"[srt] {args.srt}")


if __name__ == "__main__":
    main()
