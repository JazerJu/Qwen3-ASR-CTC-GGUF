#!/usr/bin/env python3
"""第 6 步：端到端验证 / CLI。

CTC 首遍转写 + 词级时间戳。任意 ffmpeg 支持的格式，单段 <= 30 秒
（编码器按 30 秒桶导出）。

    python 06-Inference.py input.mp3
    python 06-Inference.py clip.wav --srt out.srt --cpu --precision fp32

LLM 二遍（q5_k_m GGUF）已导出可加载，但尚未接进本 CLI。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import export_config as C
from qwen3_asr_ctc import create_asr_engine
from qwen3_asr_ctc.srt import write_srt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", default=str(HERE / "input.mp3"),
                    help="输入音频/视频，任意 ffmpeg 格式（默认 input.mp3）")
    ap.add_argument("--precision", default="q4", choices=["fp32", "fp16", "q4"],
                    help="用哪一档 ONNX（默认 q4）")
    ap.add_argument("--srt", default=None, help="写出词级时间戳 SRT")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU，不用 CUDA")
    args = ap.parse_args()

    src = Path(args.audio).expanduser()
    if not src.exists():
        print(f"找不到输入: {src}", file=sys.stderr)
        return 1

    enc = C.onnx(C.ENCODER, args.precision)
    ctc = C.onnx(C.CTC, args.precision)
    for p in (enc, ctc, C.TOKENS_TXT):
        if not p.exists():
            print(f"缺少 {p.name}，先跑 01–03 步", file=sys.stderr)
            return 1

    engine = create_asr_engine(
        encoder_onnx_path=str(enc), ctc_onnx_path=str(ctc),
        tokens_path=str(C.TOKENS_TXT), preprocessor_path=str(C.PREPROC_DIR),
        use_gpu=not args.cpu,
    )
    r = engine.transcribe(src)
    print(f"[qwen3-ctc/{args.precision}] {r.text}")
    print(f"[stats] {r.duration:.1f}s 音频 -> {r.elapsed*1000:.0f}ms (RTF {r.rtf:.3f})")

    if args.srt:
        write_srt(r.words, args.srt)
        print(f"[srt] {args.srt}  ({len(r.words)} 个单元)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
