#!/usr/bin/env python3
"""Step 6: export Qwen3-ASR LLM decoder to GGUF (fp16) then quantize q5_k_m.

Extract thinker.model.* + thinker.lm_head.weight from the full checkpoint,
rebuild a standard Qwen3ForCausalLM (QK-norm needs the native Qwen3 class,
NOT Llama), save HF format with vocab.json+merges.txt (no tokenizer.json in
this repo), convert via llama.cpp, quantize to q5_k_m — same quant grade as
the GLM and Fun decoders.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
QWEN3_DIR = Path("/data/推理框架/asr-onnx/Qwen3-ASR-HF")
CONVERT = Path("/data/推理框架/llama.cpp/convert_hf_to_gguf.py")
QUANTIZE = Path("/data/推理框架/llama.cpp/build/bin/llama-quantize")


def main():
    out_fp16 = HERE / "model" / "Qwen3-ASR-Decoder.fp16.gguf"
    out_q5 = HERE / "model" / "Qwen3-ASR-Decoder.q5_k_m.gguf"

    with tempfile.TemporaryDirectory() as td:
        hf_dir = Path(td) / "qwen3-decoder"

        print("loading full checkpoint shards...")
        full_state = {}
        for shard in sorted(QWEN3_DIR.glob("model-*.safetensors")):
            full_state.update(load_file(str(shard)))
        llm = {}
        for k, v in full_state.items():
            if k.startswith("thinker.model."):
                llm[k[len("thinker."):]] = v
            elif k == "thinker.lm_head.weight":
                llm["lm_head.weight"] = v
        del full_state
        print(f"extracted {len(llm)} decoder keys")

        from transformers import Qwen3Config, Qwen3ForCausalLM

        text_cfg = json.load(open(QWEN3_DIR / "config.json"))["thinker_config"]["text_config"]
        cfg = Qwen3Config(**text_cfg)
        model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
        model.load_state_dict(llm, strict=True)
        del llm
        model.save_pretrained(hf_dir, safe_serialization=True)
        for fname in ("vocab.json", "merges.txt", "tokenizer_config.json", "generation_config.json"):
            src = QWEN3_DIR / fname
            if src.exists():
                shutil.copy(src, hf_dir / fname)
        print(f"HF decoder saved to {hf_dir}")

        print("converting to fp16 GGUF...")
        subprocess.run([sys.executable, str(CONVERT), str(hf_dir),
                        "--outfile", str(out_fp16), "--outtype", "f16"], check=True)

    print("quantizing q5_k_m...")
    subprocess.run([str(QUANTIZE), str(out_fp16), str(out_q5), "q5_k_m"], check=True)
    print(f"DONE: {out_q5} ({out_q5.stat().st_size/2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
