#!/usr/bin/env python3
"""ORT transformer fusion on fp32 ONNX (Fun-ASR-GGUF 02).

CTC head uses bert fusion. Encoder is a custom graph (30s bucket + block mask);
fusion is attempted and dropped if the cosine gate fails.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.transformers.optimizer import optimize_model


def optimize_one(src: Path, dst: Path, opt_level: int) -> None:
    print(f"optimize {src.name} -> {dst.name}")
    opt = optimize_model(
        str(src),
        model_type="bert",
        num_heads=0,
        hidden_size=0,
        opt_level=opt_level,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    opt.save_model_to_file(str(dst))
    model = onnx.load(str(dst), load_external_data=False)
    domain_ops = defaultdict(set)
    for node in model.graph.node:
        domain_ops[node.domain or "ai.onnx"].add(node.op_type)
    print(f"  nodes={len(model.graph.node)} size={dst.stat().st_size/2**20:.1f} MiB")
    for domain, ops in sorted(domain_ops.items()):
        print(f"  [{domain}] {', '.join(sorted(ops)[:12])}")


def cosine_gate(fp32: Path, opt: Path, feeds: dict) -> bool:
    import onnxruntime as ort

    a = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(opt), providers=["CPUExecutionProvider"])
    ya = a.run(None, feeds)[0].ravel().astype(np.float64)
    yb = b.run(None, feeds)[0].ravel().astype(np.float64)
    cos = float(ya @ yb / (np.linalg.norm(ya) * np.linalg.norm(yb) + 1e-12))
    print(f"  cosine fp32 vs opt = {cos:.7f}")
    return cos >= 0.999


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import export_config as C
    here = C.MODEL_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--opt-level", type=int, default=1)
    p.add_argument("--skip-encoder", action="store_true")
    args = p.parse_args()
    rc = 0

    ctc_src = here / "Qwen3-ASR-CTC.fp32.onnx"
    ctc_dst = here / "Qwen3-ASR-CTC.opt.fp32.onnx"
    if ctc_src.exists():
        optimize_one(ctc_src, ctc_dst, args.opt_level)
        x = np.random.randn(1, 80, 2048).astype(np.float32)
        if not cosine_gate(ctc_src, ctc_dst, {"enc_output": x}):
            print("FAIL CTC optimize gate; delete opt, Linux int4 stays on raw fp32")
            ctc_dst.unlink(missing_ok=True)
            rc = 1
    else:
        print("skip missing CTC fp32")

    enc_src = here / "Qwen3-ASR-Encoder.fp32.onnx"
    enc_dst = here / "Qwen3-ASR-Encoder.opt.fp32.onnx"
    if args.skip_encoder:
        print("skip encoder optimize")
    elif enc_src.exists():
        try:
            optimize_one(enc_src, enc_dst, args.opt_level)
            mel = np.random.randn(128, 3000).astype(np.float32)
            fl = np.array([800], np.int64)
            if not cosine_gate(enc_src, enc_dst, {"input_features": mel, "feature_length": fl}):
                print("FAIL encoder optimize gate; DML fp16 will use unoptimized fp32")
                enc_dst.unlink(missing_ok=True)
        except Exception as e:
            print(f"FAIL encoder optimize: {e}")
            enc_dst.unlink(missing_ok=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
