#!/usr/bin/env python3
"""Step 3: export the trained CTC head to ONNX + fp32-parity gate (cos >= 0.9999)."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from safetensors.torch import load_file

from modeling_ctc import CTCDecoder


def main():
    cfg = json.load(open(HERE / "config.json", encoding="utf-8"))
    head = CTCDecoder(
        encoder_dim=cfg["encoder_dim"], ctc_hidden=cfg["ctc_hidden"],
        proj_hidden=cfg["proj_hidden"], num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"], ffn_hidden=cfg["ffn_hidden"],
        vocab_size=cfg["vocab_size"], blank_id=cfg["blank_id"],
    )
    sd = load_file(str(HERE / "ctc_head.safetensors"))
    head.load_state_dict({k: v for k, v in sd.items() if not k.startswith("optimizer.")}, strict=True)
    head.eval()

    out = HERE / "model" / "Qwen3-ASR-CTC.fp32.onnx"
    dummy = torch.randn(1, 200, cfg["encoder_dim"])
    with torch.no_grad():
        torch.onnx.export(
            head, (dummy,), str(out),
            input_names=["enc_output"], output_names=["logits"],
            dynamic_axes={"enc_output": {0: "batch", 1: "time"}, "logits": {0: "batch", 1: "time"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    print(f"exported {out.name} ({out.stat().st_size/1e6:.1f} MB)")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ok = True
    for T in (65, 156, 325):
        x = torch.randn(1, T, cfg["encoder_dim"])
        with torch.no_grad():
            ref = head(x).numpy()
        hyp = sess.run(None, {"enc_output": x.numpy()})[0]
        cos = float(ref.ravel() @ hyp.ravel() / (np.linalg.norm(ref) * np.linalg.norm(hyp)))
        gate = cos >= 0.9999
        ok &= gate
        print(f"T={T:4d}  cosine={cos:.7f}  {'PASS' if gate else 'FAIL'}")
    print("GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
