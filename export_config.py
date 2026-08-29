"""集中配置：所有路径和产物名只在这里定义一次。

导出脚本 01–05 和验证脚本 07 都从这里取路径，改机器只改这个文件。
运行时包 qwen3_asr_ctc 不依赖它（运行时只认 model/ 下的文件名）。
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── 输入 ───────────────────────────────────────────────────────────────
# Qwen3-ASR-1.7B 官方权重（编码器 + LLM 解码器都在这一份里）
#   huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir <dir>
QWEN3_DIR = Path(os.environ.get(
    "QWEN3_ASR_DIR", "/data/推理框架/asr-onnx/Qwen3-ASR-HF"))

# 训练好的 CTC 头（从 HF 拉，或指到本地目录）
#   v1: JazerJu/qwen3-asr-ctc      48.3M, ffn_hidden=128
#   v2: JazerJu/qwen3-asr-ctc-r2   58.2M, ffn_hidden=2048
CTC_DIR = Path(os.environ.get("QWEN3_CTC_DIR", HERE / "ctc"))
CTC_CONFIG = CTC_DIR / "config.json"
CTC_WEIGHTS = CTC_DIR / "ctc_head.safetensors"
CTC_VOCAB = CTC_DIR / "vocab_compact.json"

# llama.cpp（第 4、5 步用）
LLAMA_CPP = Path(os.environ.get("LLAMA_CPP_DIR", "/data/推理框架/llama.cpp"))
CONVERT_HF_TO_GGUF = LLAMA_CPP / "convert_hf_to_gguf.py"
LLAMA_QUANTIZE = LLAMA_CPP / "build" / "bin" / "llama-quantize"

# ── 输出 ───────────────────────────────────────────────────────────────
MODEL_DIR = HERE / "model"
PREPROC_DIR = HERE / "preprocessor"

ENCODER = "Qwen3-ASR-Encoder"
CTC = "Qwen3-ASR-CTC"
DECODER = "Qwen3-ASR-Decoder"

def onnx(stem: str, precision: str) -> Path:
    """model/<stem>.<precision>.onnx —— precision ∈ {fp32, opt.fp32, fp16, q4}"""
    return MODEL_DIR / f"{stem}.{precision}.onnx"

TOKENS_TXT = MODEL_DIR / "tokens.txt"
DECODER_FP16_GGUF = MODEL_DIR / f"{DECODER}.fp16.gguf"
DECODER_Q5_GGUF = MODEL_DIR / f"{DECODER}.q5_k_m.gguf"

# ── 导出常量（改这些会让已导出的模型失效）─────────────────────────────
MEL_FRAMES = 3000        # 30 秒桶：编码器按固定形状导出，运行时传真实 feature_length
MEL_BINS = 128
SAMPLE_RATE = 16000
FRAME_RATE_HZ = 13       # 每 100 个 mel 帧出 13 帧，帧移 1/13 秒，不是 8×10ms
INT4_BLOCK_SIZE = 128
