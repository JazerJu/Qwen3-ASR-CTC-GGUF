import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("quantize", ROOT / "03-Quantize-ONNX.py")
quantize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantize)


class QuantizationTests(unittest.TestCase):
    def test_rtn_and_hqq_make_runnable_nbits_graphs(self):
        rng = np.random.default_rng(42)
        weight = rng.normal(size=(128, 32)).astype(np.float32)
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "weight"], ["y"], name="projection")],
            "test", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 128])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])],
            [numpy_helper.from_array(weight, "weight")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.onnx"
            onnx.save(model, source)
            fp16 = Path(td) / "test.fp16.onnx"
            quantize.to_fp16(source, fp16)
            onnx.checker.check_model(str(fp16))
            for method, tag in (("rtn", "q4"), ("hqq", "hqq")):
                output = Path(td) / f"test.{tag}.onnx"
                quantize.quant_int4(source, output, 128, False, method=method)
                graph = onnx.load(output)
                self.assertTrue(any(n.op_type == "MatMulNBits" for n in graph.graph.node))
                session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
                result = session.run(None, {"x": np.ones((1, 128), np.float32)})[0]
                self.assertEqual(result.shape, (1, 32))
                self.assertTrue(np.isfinite(result).all())
                f16 = Path(td) / f"test.{tag}f16.onnx"
                quantize.to_q4f16(output, f16)
                onnx.checker.check_model(str(f16))

    def test_hqq_symmetric_fails_before_loading(self):
        with self.assertRaisesRegex(ValueError, "RTN"):
            quantize.quant_int4(Path("missing.onnx"), Path("out.onnx"), 128, True, method="hqq")


if __name__ == "__main__":
    unittest.main()
