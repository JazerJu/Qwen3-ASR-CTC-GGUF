"""Qwen3-ASR-CTC 运行时。

    from qwen3_asr_ctc import create_asr_engine

    engine = create_asr_engine(
        encoder_onnx_path="model/Qwen3-ASR-Encoder.q4.onnx",
        ctc_onnx_path="model/Qwen3-ASR-CTC.q4.onnx",
        tokens_path="model/tokens.txt",
        preprocessor_path="preprocessor",
        decoder_gguf_path="model/Qwen3-ASR-Decoder.q5_k_m.gguf",   # 不给就是纯 CTC
        hotwords=["Claude Code", "科大讯飞"],
    )
    result = engine.transcribe("input.mp3")
    print(result.text)       # LLM 二遍文本（纯 CTC 模式下 = CTC 文本）
    print(result.ctc_text)   # CTC 首遍文本
    print(result.words)      # [(词, 起, 止)]
"""
import logging
import os

logger = logging.getLogger("qwen3_asr_ctc")   # hotword 子包 `from .. import logger` 用


def setup_logging(level: int = logging.INFO, log_file: str | None = None):
    logger.setLevel(level)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        h = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(h)
    return logger


from .engine import ASREngineConfig, ASRResult, Qwen3ASREngine, Qwen3CtcEngine, create_asr_engine  # noqa: E402
from .schema import DecodeResult, RecognitionStream, Timings  # noqa: E402

__all__ = ["Qwen3ASREngine", "Qwen3CtcEngine", "ASREngineConfig", "ASRResult",
           "DecodeResult", "RecognitionStream", "Timings", "create_asr_engine",
           "logger", "setup_logging"]
