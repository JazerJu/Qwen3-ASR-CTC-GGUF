#!/usr/bin/env python3
"""Quantize the fp32 exports: int4 (CUDA/Linux) + fp16 (DirectML/Windows iGPU).

Both variants derive from the RAW fp32 graphs in model/. The fp16 pass must
not consume 02-Optimize output: fused com.microsoft ops break
convert_float_to_float16's cast insertion and yield mixed fp16/fp32 graphs.

The int4 text gate (fp32 vs int4 greedy transcripts, <= 1% char diff on
LibriSpeech) lives in 05-Gate-Int4.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.transformers.float16 import convert_float_to_float16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_config as C  # noqa: E402

MODEL = C.MODEL_DIR


def gemm_to_matmul(model_path: Path, out_path: Path) -> int:
    """Rewrite Gemm(A, B, C, transB=1) -> MatMul(A, B^T) + Add(C) so the
    MatMul-only int4 quantizer can reach the encoder's Linear weights."""
    from onnx import helper, numpy_helper

    m = onnx.load(str(model_path))
    inits = {i.name: i for i in m.graph.initializer}
    gemm_b = {}
    new_nodes = []
    for node in m.graph.node:
        if node.op_type != "Gemm":
            new_nodes.append(node)
            continue
        attrs = {a.name: (a.i if a.type == onnx.AttributeProto.INT else a.f) for a in node.attribute}
        assert attrs.get("alpha", 1.0) == 1.0 and attrs.get("beta", 1.0) == 1.0
        assert node.input[1] in inits, f"non-constant Gemm B: {node.input[1]}"
        transposed_name = node.input[1] + "_qt"
        gemm_b[node.input[1]] = transposed_name if attrs.get("transB", 0) else node.input[1]
        mm = helper.make_node("MatMul", [node.input[0], gemm_b[node.input[1]]], [node.output[0] + "_mm"])
        if len(node.input) > 2 and node.input[2]:
            new_nodes.append(mm)
            new_nodes.append(helper.make_node("Add", [mm.output[0], node.input[2]], [node.output[0]]))
        else:
            mm.output[0] = node.output[0]
            new_nodes.append(mm)
    converted = len(gemm_b)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    fresh = []
    for init in m.graph.initializer:
        if init.name in gemm_b:
            w = numpy_helper.to_array(init)
            name = gemm_b[init.name]
            if name != init.name:
                w = w.T.copy()
            fresh.append(numpy_helper.from_array(w, name=name))
        else:
            fresh.append(init)
    del m.graph.initializer[:]
    m.graph.initializer.extend(fresh)
    onnx.save(m, str(out_path))
    return converted


def quant_int4(src: Path, dst: Path, block_size: int, symmetric: bool):
    needs_rewrite = any(n.op_type == "Gemm" for n in onnx.load(str(src)).graph.node)
    target = src
    if needs_rewrite:
        tmp = dst.with_suffix(".pre.onnx")
        n = gemm_to_matmul(src, tmp)
        print(f"  [pre] rewrote {n} Gemm -> MatMul+Add")
        target = tmp
    quant = MatMulNBitsQuantizer(
        model=str(target), bits=4, block_size=block_size,
        is_symmetric=symmetric, op_types_to_quantize=("MatMul",),
    )
    quant.process()
    dst.parent.mkdir(parents=True, exist_ok=True)
    quant.model.save_model_to_file(str(dst))
    if target != src:
        target.unlink()
    print(f"  int4 {dst.name}: {dst.stat().st_size/2**20:.1f} MiB (was {src.stat().st_size/2**20:.1f})")


def to_fp16(src: Path, dst: Path):
    model = onnx.load(str(src))
    model_fp16 = convert_float_to_float16(
        model, keep_io_types=True,
        min_positive_val=1e-7, max_finite_val=65504,
        op_block_list=["LayerNormalization"],
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(dst))
    print(f"  fp16 {dst.name}: {dst.stat().st_size/2**20:.1f} MiB")


def to_q4f16(q4: Path, dst: Path):
    """int4 权重 + fp16 激活，给显存紧的 N 卡。

    q4 是从 fp32 图量化的，MatMulNBits 的激活是 fp32，CUDA EP 上走不带 tensor core 的 fp32 GEMM，
    比 fp16 慢一倍（5070 Ti 上 Qwen encoder 21.9 vs 9.7ms，GLM 105 vs 50ms）。MatMulNBits 本身不慢
    ——单算子 A=fp16 时 2.81ms，fp16 MatMul 2.77ms。把激活和 scale 转成 fp16，速度就和 fp16 一样，
    体积约 fp16 的 1/4，帧级 CTC 输出与 q4 完全一致。DML 跑不了 MatMulNBits，那边仍用 fp16。
    """
    model = onnx.load(str(q4))
    m16 = convert_float_to_float16(
        model, keep_io_types=True,
        min_positive_val=1e-7, max_finite_val=65504,
        disable_shape_infer=True,   # 不 block LayerNormalization：和 disable_shape_infer 一起用会让 LN 的输入/权重类型不一致，CUDA 的 fp16 LN 内部本就 fp32 累加
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m16, str(dst))
    print(f"  q4f16 {dst.name}: {dst.stat().st_size/2**20:.1f} MiB")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--symmetric", action="store_true")
    p.add_argument("--skip-int4", action="store_true")
    p.add_argument("--skip-fp16", action="store_true")
    p.add_argument("--skip-q4f16", action="store_true")
    args = p.parse_args()

    for stem in ("Qwen3-ASR-CTC", "Qwen3-ASR-Encoder"):
        fp32 = MODEL / f"{stem}.fp32.onnx"
        if not fp32.exists():
            print(f"skip missing {stem}.fp32")
            continue
        print(stem)
        if not args.skip_int4:
            quant_int4(fp32, MODEL / f"{stem}.q4.onnx", args.block_size, args.symmetric)
        if not args.skip_fp16:
            to_fp16(fp32, MODEL / f"{stem}.fp16.onnx")
        if not args.skip_q4f16 and (MODEL / f"{stem}.q4.onnx").exists():
            to_q4f16(MODEL / f"{stem}.q4.onnx", MODEL / f"{stem}.q4f16.onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
