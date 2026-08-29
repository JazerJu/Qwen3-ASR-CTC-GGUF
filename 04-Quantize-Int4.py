#!/usr/bin/env python3
"""Quantize Qwen3-ASR ONNX models to int4 (MatMulNBits, weight-only).

Unlike the GLM 01c script this also targets Gemm nodes — the Qwen encoder
exports its Linears as Gemm and would otherwise stay ~98% uncompressed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def gemm_to_matmul(model_path: Path, out_path: Path) -> int:
    """Rewrite Gemm(A, B, C, transB=1) -> MatMul(A, B^T) + Add(C) so the
    int4 quantizer (MatMul-only) can reach the encoder's Linear weights."""
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--symmetric", action="store_true")
    args = p.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    needs_gemm_rewrite = any(n.op_type == "Gemm" for n in onnx.load(str(args.input)).graph.node)
    target = args.input
    if needs_gemm_rewrite:
        tmp = args.output.with_suffix(".pre.onnx")
        n = gemm_to_matmul(args.input, tmp)
        print(f"[pre] rewrote {n} Gemm -> MatMul+Add")
        target = tmp

    quant = MatMulNBitsQuantizer(
        model=str(target),
        bits=4,
        block_size=args.block_size,
        is_symmetric=args.symmetric,
        op_types_to_quantize=("MatMul",),
    )
    quant.process()
    quant.model.save_model_to_file(str(args.output))
    if target != args.input:
        target.unlink()
    print(f"OK quantized: {args.output} "
          f"({args.output.stat().st_size/2**20:.1f} MiB, "
          f"was {args.input.stat().st_size/2**20:.1f} MiB, "
          f"ratio {args.output.stat().st_size/args.input.stat().st_size*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
