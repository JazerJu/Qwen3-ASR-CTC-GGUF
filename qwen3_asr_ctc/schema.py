"""数据结构：识别流 / 计时 / CTC 单元 / 解码结果。字段与 GLM-ASR-CTC-GGUF 同名，
方便 CapsWriter 侧用同一套适配代码。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class RecognitionResult:
    text: str = ""
    timestamps: List[float] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)


@dataclass
class RecognitionStream:
    """兼容 sherpa-onnx 的流对象：accept_waveform 进音频，result 出文本。"""
    sample_rate: int = 16000
    audio_data: Optional[Any] = None
    _result: Optional[RecognitionResult] = field(default=None, init=False, repr=False)

    def accept_waveform(self, sample_rate: int, audio: Any):
        self.sample_rate = sample_rate
        self.audio_data = np.asarray(audio, dtype=np.float32)

    @property
    def result(self) -> RecognitionResult:
        if self._result is None:
            self._result = RecognitionResult()
        return self._result

    def set_result(self, text: str, timestamps=None, tokens=None):
        self._result = RecognitionResult(text=text, timestamps=list(timestamps or []),
                                         tokens=list(tokens or []))


@dataclass
class Timings:
    """各阶段耗时（秒）。"""
    encode: float = 0.0        # mel + 编码器
    ctc: float = 0.0           # CTC 头 + argmax + 字节反解 + 热词匹配
    prepare: float = 0.0       # 拼 prompt embedding
    inject: float = 0.0        # LLM prefill
    llm_generate: float = 0.0  # LLM 逐 token 生成
    align: float = 0.0         # CTC 时间戳 -> LLM 文本的 NW 对齐
    total: float = 0.0


@dataclass
class CTCResult:
    """CTC 首遍的一个字符及其起始秒；score 无消费者，保留只为与 GLM 包同形。"""
    text: str
    timestamp: float
    score: float = 1.0


@dataclass
class LLMDecodeResult:
    text: str = ""
    n_gen: int = 0
    t_inject: float = 0.0
    t_gen: float = 0.0
    is_aborted: bool = False


@dataclass
class DecodeResult:
    text: str = ""
    ctc_text: str = ""
    ctc_results: List[CTCResult] = field(default_factory=list)
    aligned: List[List[Any]] = field(default_factory=list)   # [[词, 起秒], ...]
    n_audio: int = 0
    n_prefix: int = 0
    n_suffix: int = 0
    n_gen: int = 0
    hotwords: List[str] = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)
    is_aborted: bool = False
