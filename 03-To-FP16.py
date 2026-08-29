#!/usr/bin/env python3
"""fp16 ONNX for DirectML, from the RAW fp32 graph.

Never feed the 02-Optimize output here: fused com.microsoft ops
(SkipLayerNormalization/BiasGelu) break convert_float_to_float16's cast
insertion and yield mixed fp16/fp32 graphs.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16


def to_fp16(src: Path, dst: Path) -> None:
    print(f"fp16 {src.name} -> {dst.name}")
    model = onnx.load(str(src))
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,
        min_positive_val=1e-7,
        max_finite_val=65504,
        op_block_list=["LayerNormalization"],
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(dst))
    print(f"  {dst.stat().st_size/2**20:.1f} MiB")


def main() -> int:
    here = Path(__file__).resolve().parent / "model"
    for stem in ("Qwen3-ASR-CTC", "Qwen3-ASR-Encoder"):
        raw = here / f"{stem}.fp32.onnx"
        if not raw.exists():
            print(f"skip missing {stem}.fp32")
            continue
        to_fp16(raw, here / f"{stem}.fp16.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
