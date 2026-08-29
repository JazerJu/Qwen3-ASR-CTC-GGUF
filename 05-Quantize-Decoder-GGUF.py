#!/usr/bin/env python3
"""第 5 步：用 llama-quantize 把 fp16 GGUF 量化到 q5_k_m。

q5_k_m 与 GLM、Fun 两条链路同一量化档，便于三方横向比较。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import export_config as C

QUANT_TYPE = "q5_k_m"


def main() -> int:
    if not C.DECODER_FP16_GGUF.exists():
        print(f"先跑 04-Export-Decoder-GGUF-FP16.py（缺 {C.DECODER_FP16_GGUF.name}）",
              file=sys.stderr)
        return 1
    if not C.LLAMA_QUANTIZE.exists():
        print(f"找不到 llama-quantize: {C.LLAMA_QUANTIZE}\n"
              f"Windows 上是 llama-quantize.exe；设 LLAMA_CPP_DIR 指向 llama.cpp",
              file=sys.stderr)
        return 1

    subprocess.run([str(C.LLAMA_QUANTIZE), str(C.DECODER_FP16_GGUF),
                    str(C.DECODER_Q5_GGUF), QUANT_TYPE], check=True)
    a = C.DECODER_FP16_GGUF.stat().st_size / 2**20
    b = C.DECODER_Q5_GGUF.stat().st_size / 2**20
    print(f"DONE: {C.DECODER_Q5_GGUF.name} ({b:.0f} MiB，fp16 是 {a:.0f} MiB，压到 {b/a:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
