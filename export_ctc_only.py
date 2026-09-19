# -*- coding: utf-8 -*-
"""只导 CTC 头：ctc-<name>/ -> model-<name>/Qwen3-ASR-CTC.{fp32,fp16,q4}.onnx，并建 bench 用的 models/qwen-ctc-<name>/。

编码器、decoder、tokens.txt 跨轮不变（同一份冻结编码器、同一张紧凑词表），软链到 model/，
和第一轮 model-aiinfra 的做法一致。复用 01 的 export_ctc()（含 ONNX vs PyTorch 余弦门）
和 03 的 quant_int4() / to_fp16()。
    python export_ctc_only.py aiinfra2
"""
import importlib.util, json, os, sys
from pathlib import Path

name = sys.argv[1]
HERE = Path("/data/推理框架/asr-onnx/Qwen3-ASR-CTC-GGUF")
BENCH = Path("/data/推理框架/asr-onnx/bench-asr-ctc/models")
os.environ["QWEN3_CTC_DIR"] = str(HERE / ("ctc-" + name))
os.environ["QWEN3_MODEL_DIR"] = str(HERE / ("model-" + name))
sys.path.insert(0, str(HERE)); os.chdir(HERE)


def load(fn, mod):
    spec = importlib.util.spec_from_file_location(mod, HERE / fn)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


e1 = load("01-Export-ONNX-FP32.py", "exp01")
C = e1.C
assert str(C.MODEL_DIR).endswith("model-" + name) and str(C.CTC_DIR).endswith("ctc-" + name), (C.MODEL_DIR, C.CTC_DIR)
C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
ok = e1.export_ctc()
print("CTC 一致性门:", "PASS" if ok else "FAIL")
if not ok:
    sys.exit(1)

q3 = load("03-Quantize-ONNX.py", "exp03")
fp32 = C.onnx(C.CTC, "fp32")
q3.quant_int4(fp32, C.onnx(C.CTC, "q4"), 128, False)
q3.to_fp16(fp32, C.onnx(C.CTC, "fp16"))

for f in ["Qwen3-ASR-Decoder.q5_k_m.gguf", "Qwen3-ASR-Encoder.fp16.onnx", "Qwen3-ASR-Encoder.q4.onnx", "tokens.txt"]:
    link = C.MODEL_DIR / f
    if not link.exists():
        link.symlink_to(Path("..") / "model" / f)

b = BENCH / ("qwen-ctc-" + name)
b.mkdir(parents=True, exist_ok=True)
for prec in ["fp16", "q4"]:
    link = b / ("Qwen3-ASR-CTC.%s.onnx" % prec)
    if not link.exists():
        link.symlink_to(C.onnx(C.CTC, prec))
(b / "config.json").write_text((C.CTC_DIR / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
for p in sorted(C.MODEL_DIR.iterdir()) + sorted(b.iterdir()):
    print("  %-40s %s" % (p.name, ("-> " + os.readlink(p)) if p.is_symlink() else "%.1f MB" % (p.stat().st_size / 1e6)))
print("EXPORT_DONE", name)
