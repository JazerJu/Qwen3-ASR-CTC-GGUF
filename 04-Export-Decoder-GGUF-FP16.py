#!/usr/bin/env python3
"""第 4 步：抽出 LLM 解码器 -> HF 标准格式 -> fp16 GGUF。

从整份 Qwen3-ASR 权重里取 `thinker.model.*` + `thinker.lm_head.weight`，
重建成标准 `Qwen3ForCausalLM`，再交给 llama.cpp 的 convert_hf_to_gguf.py。

两个必须注意的点：
  - **必须用 Qwen3Config / Qwen3ForCausalLM，不能照抄 GLM 那版的 Llama 类** ——
    Qwen3 有 QK-norm，用 Llama 会 load_state_dict 缺键。
  - 官方仓库**没有 tokenizer.json**，只有 vocab.json + merges.txt，
    convert_hf_to_gguf.py 认这个组合。

tie_word_embeddings 实测过：lm_head.weight 与 embed_tokens.weight 逐位相同
（sha256 都是 d7d2c2a8e14c215f），配置是对的，按默认路径走即可。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
from safetensors.torch import load_file

import export_config as C


def main() -> int:
    if not C.CONVERT_HF_TO_GGUF.exists():
        print(f"找不到 convert_hf_to_gguf.py: {C.CONVERT_HF_TO_GGUF}\n"
              f"设 LLAMA_CPP_DIR 指向 llama.cpp 目录", file=sys.stderr)
        return 1
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        hf_dir = Path(td) / "qwen3-decoder"

        print("载入完整 checkpoint 分片 ...")
        full = {}
        for shard in sorted(C.QWEN3_DIR.glob("model-*.safetensors")):
            full.update(load_file(str(shard)))
        llm = {}
        for k, v in full.items():
            if k.startswith("thinker.model."):
                llm[k[len("thinker."):]] = v
            elif k == "thinker.lm_head.weight":
                llm["lm_head.weight"] = v
        del full
        print(f"抽出 {len(llm)} 个解码器权重")

        from transformers import Qwen3Config, Qwen3ForCausalLM

        text_cfg = json.loads((C.QWEN3_DIR / "config.json").read_text())["thinker_config"]["text_config"]
        model = Qwen3ForCausalLM(Qwen3Config(**text_cfg)).to(torch.bfloat16)
        model.load_state_dict(llm, strict=True)
        del llm
        model.save_pretrained(hf_dir, safe_serialization=True)
        for fname in ("vocab.json", "merges.txt", "tokenizer_config.json",
                      "generation_config.json"):
            src = C.QWEN3_DIR / fname
            if src.exists():
                shutil.copy(src, hf_dir / fname)

        print("转 fp16 GGUF ...")
        subprocess.run([sys.executable, str(C.CONVERT_HF_TO_GGUF), str(hf_dir),
                        "--outfile", str(C.DECODER_FP16_GGUF), "--outtype", "f16"], check=True)

    size = C.DECODER_FP16_GGUF.stat().st_size / 2**20
    print(f"DONE: {C.DECODER_FP16_GGUF.name} ({size:.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
