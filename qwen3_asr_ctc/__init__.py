"""Qwen3-ASR-CTC 运行时。

    from qwen3_asr_ctc import create_asr_engine

    engine = create_asr_engine(
        encoder_onnx_path="model/Qwen3-ASR-Encoder.q4.onnx",
        ctc_onnx_path="model/Qwen3-ASR-CTC.q4.onnx",
        tokens_path="model/tokens.txt",
        preprocessor_path="preprocessor",
    )
    result = engine.transcribe("input.mp3")
    print(result.text)
    print(result.words)      # [(词, 起, 止)]
"""
from .engine import ASREngineConfig, ASRResult, Qwen3CtcEngine, create_asr_engine

__all__ = ["Qwen3CtcEngine", "ASREngineConfig", "ASRResult", "create_asr_engine"]
