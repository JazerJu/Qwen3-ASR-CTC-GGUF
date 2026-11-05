# -*- coding: utf-8 -*-
"""只导 CTC 头：ctc-<name>/ -> model-<name>/Qwen3-ASR-CTC.{fp32,fp16,q4}.onnx，并建 bench 用的 models/qwen-ctc-<name>/。

编码器、decoder、tokens.txt 跨轮不变（同一份冻结编码器、同一张紧凑词表），软链到 model/，
和第一轮 model-aiinfra 的做法一致。复用 01 的 export_ctc()（含 ONNX vs PyTorch 余弦门）
和 03 的 quant_int4() / to_fp16()。
    python export_ctc_only.py aiinfra2
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(fn, mod):
    spec = importlib.util.spec_from_file_location(mod, HERE / fn)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="round name: ctc-<name>/ -> model-<name>/")
    ap.add_argument("--method", choices=["rtn", "hqq"], default="rtn")
    ap.add_argument("--bench-dir", type=Path, help="optionally link into <bench-dir>/models/")
    args = ap.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        ap.error("name must be a single directory component")
    os.environ["QWEN3_CTC_DIR"] = str(HERE / ("ctc-" + args.name))
    os.environ["QWEN3_MODEL_DIR"] = str(HERE / ("model-" + args.name))
    sys.path.insert(0, str(HERE))

    e1 = load("01-Export-ONNX-FP32.py", "exp01")
    C = e1.C
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ok = e1.export_ctc()
    print("GATE:", "PASS" if ok else "FAIL", "(CTC ONNX vs PyTorch)")
    if not ok:
        return 1
    q3 = load("03-Quantize-ONNX.py", "exp03")
    tag = "hqq" if args.method == "hqq" else "q4"
    fp32 = C.onnx(C.CTC, "fp32")
    q3.quant_int4(fp32, C.onnx(C.CTC, tag), 128, False, method=args.method)
    q3.to_fp16(fp32, C.onnx(C.CTC, "fp16"))
    q3.to_q4f16(C.onnx(C.CTC, tag), C.onnx(C.CTC, tag + "f16"))

    for f in ["Qwen3-ASR-Decoder.q5_k_m.gguf", "Qwen3-ASR-Encoder.fp16.onnx",
              "Qwen3-ASR-Encoder.q4.onnx", "tokens.txt"]:
        link = C.MODEL_DIR / f
        if not link.exists() and not link.is_symlink():
            link.symlink_to(Path("..") / "model" / f)

    if args.bench_dir:
        b = args.bench_dir.resolve() / "models" / ("qwen-ctc-" + args.name)
        b.mkdir(parents=True, exist_ok=True)
        for prec in ("fp16", tag, tag + "f16"):
            link = b / f"Qwen3-ASR-CTC.{prec}.onnx"
            if not link.exists() and not link.is_symlink():
                link.symlink_to(C.onnx(C.CTC, prec))
        (b / "config.json").write_text(C.CTC_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    print("EXPORT_DONE", args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
