#!/usr/bin/env python3
"""第 6 步：端到端验证 / CLI。

默认两遍：CTC 首遍（词级时间戳 + 音素热词）-> LLM 二遍（q5_k_m GGUF）-> NW 对齐。
model/ 里没有 Decoder GGUF、或加 --no-decoder，就只跑 CTC 首遍。
任意 ffmpeg 支持的格式；超过 30 秒按 30 秒切段（编码器按 30 秒桶导出）。

    python 06-Inference.py input.mp3
    python 06-Inference.py clip.wav --srt out.srt --hotwords hot.txt
    python 06-Inference.py clip.wav --no-decoder --cpu --precision fp32
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
    ap.add_argument("--cpu", action="store_true", help="强制 CPU（ONNX 和 llama 都不用 GPU）")
    ap.add_argument("--no-decoder", action="store_true", help="只跑 CTC 首遍")
    ap.add_argument("--hotwords", default=None, help="热词文件，一行一个，# 开头为注释")
    ap.add_argument("--context", default=None, help="上下文文本，放进 system 段")
    ap.add_argument("--language", default=None, help="指定语种，如 Chinese / English")
    ap.add_argument("--verbose", action="store_true", help="显示 llama.cpp 加载日志")
    args = ap.parse_args()
    if not args.verbose:
        from qwen3_asr_ctc import llama_cpp_bindings as _llama
        _llama.configure_logging(False)

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
    decoder = None if args.no_decoder else (C.DECODER_Q5_GGUF if C.DECODER_Q5_GGUF.exists() else None)
    if not args.no_decoder and decoder is None:
        print(f"没有 {C.DECODER_Q5_GGUF.name}（先跑 04–05 步），退化为纯 CTC", file=sys.stderr)

    hotwords = []
    if args.hotwords:
        for line in Path(args.hotwords).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                hotwords.append(line)

    engine = create_asr_engine(
        encoder_onnx_path=str(enc), ctc_onnx_path=str(ctc),
        tokens_path=str(C.TOKENS_TXT), preprocessor_path=str(C.PREPROC_DIR),
        decoder_gguf_path=str(decoder) if decoder else None,
        use_gpu=not args.cpu, llm_use_gpu=not args.cpu, hotwords=hotwords,
    )
    r = engine.transcribe(src, language=args.language, context=args.context)
    tag = f"qwen3-ctc+llm/{args.precision}" if engine.has_decoder else f"qwen3-ctc/{args.precision}"
    if engine.has_decoder:
        print(f"[ctc ] {r.ctc_text}")
        if r.hotwords:
            print(f"[热词] {r.hotwords}")
    print(f"[{tag}] {r.text}")
    print(f"[stats] {r.duration:.1f}s 音频 -> {r.elapsed*1000:.0f}ms (RTF {r.rtf:.3f})")

    if args.srt:
        write_srt(r.words, args.srt)
        print(f"[srt] {args.srt}  ({len(r.words)} 个单元)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
