"""ONNX 运行时引擎：mel -> 编码器 -> CTC -> 字节反解 -> 文本 + 词级时间戳。

三个必须照做、否则静默出错的点（详见 README「四个坑」）：

1. **特征不补到 30 秒**：`padding=False` 取真实长度，再自己零填到导出时的
   3000 帧桶，并把真实帧数作为 `feature_length` 传进去。用 extractor 自带的
   30 秒补齐会让 `feature_length` 对不上。
2. **帧率 13 fps**：有效输出帧数 = `full*13 + ceil(leave/8)`，不是 `T/8`。
   算错会把 logits 截断，转写被腰斩且不报错。
3. **反解走字节**：见 tokens.py。
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .tokens import decode_bytes, load_tokens

FRAME_SEC = 1.0 / 13.0
# CTC 尖峰式发射的系统性偏置，对 MFA 词级真值实测（v1：起点晚 100ms、终点早 78.5ms）
DEFAULT_START_BIAS = 0.100
DEFAULT_END_BIAS = 0.0785


@dataclass
class ASREngineConfig:
    encoder_onnx_path: str
    ctc_onnx_path: str
    tokens_path: str
    preprocessor_path: str
    blank_id: int = 72466
    unk_id: int = 72467
    mel_frames: int = 3000
    use_gpu: bool = True
    start_bias: float = DEFAULT_START_BIAS
    end_bias: float = DEFAULT_END_BIAS


@dataclass
class ASRResult:
    text: str
    words: list = field(default_factory=list)   # [(词, 起秒, 止秒)]
    duration: float = 0.0
    elapsed: float = 0.0

    @property
    def rtf(self) -> float:
        return self.elapsed / self.duration if self.duration else float("nan")


class Qwen3CtcEngine:
    def __init__(self, config: ASREngineConfig):
        self.cfg = config
        self._np = None
        self._enc = self._ctc = self._fe = None
        self.id2bytes: dict[int, bytes] = {}

    def initialize(self) -> "Qwen3CtcEngine":
        import numpy as np
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        providers = ["CPUExecutionProvider"]
        if self.cfg.use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self._enc = ort.InferenceSession(self.cfg.encoder_onnx_path, providers=providers)
        self._ctc = ort.InferenceSession(self.cfg.ctc_onnx_path, providers=providers)
        self._fe = WhisperFeatureExtractor.from_pretrained(self.cfg.preprocessor_path)
        self.id2bytes = load_tokens(self.cfg.tokens_path)
        self._np = np
        return self

    # ── 内部 ───────────────────────────────────────────────────────────
    @staticmethod
    def _valid_frames(mel_len: int) -> int:
        """真实 mel 帧数 -> 编码器有效输出帧数（每 100 帧出 13 帧）。"""
        full, leave = divmod(mel_len, 100)
        return full * 13 + (0 if leave == 0 else (leave - 1) // 8 + 1)

    def _frames(self, audio):
        np = self._np
        mel = self._fe(audio, sampling_rate=16000, padding=False,
                       return_tensors="np").input_features[0]
        t = mel.shape[1]
        padded = np.zeros((mel.shape[0], self.cfg.mel_frames), dtype=np.float32)
        padded[:, :t] = mel
        enc = self._enc.run(None, {"input_features": padded,
                                   "feature_length": np.array([t], np.int64)})[0]
        logits = self._ctc.run(
            None, {"enc_output": enc[: self._valid_frames(t)][None].astype(np.float32)})[0]
        ids = np.argmax(logits[0], axis=-1)

        out, prev = [], None
        for frame, tok in enumerate(ids):
            tok = int(tok)
            if tok == prev:
                continue
            prev = tok
            if tok not in (self.cfg.blank_id, self.cfg.unk_id):
                out.append((tok, frame))
        return out

    def _words(self, tokens):
        """CJK 一字一词，拉丁串按空白合并成词；减掉 CTC 的常数偏置。"""
        units = []
        for tid, frame in tokens:
            text = self.id2bytes[tid].decode("utf-8", errors="replace")
            start = max(frame * FRAME_SEC - self.cfg.start_bias, 0.0)
            end = max((frame + 1) * FRAME_SEC - self.cfg.end_bias, start)
            # 一个 token 覆盖多个字符时按字符位置线性内插，否则「大家」的两个字
            # 会拿到完全相同的时间戳（训练侧的强制对齐也是这么切的）
            n = len(text) or 1
            span = (end - start) / n
            units += [[ch, start + i * span, start + (i + 1) * span]
                      for i, ch in enumerate(text)]

        merged, buf, buf_s, buf_e = [], [], None, None
        for ch, s, e in units:
            if re.match(r"[A-Za-z0-9']", ch):
                buf_s = s if buf_s is None else buf_s
                buf.append(ch)
                buf_e = e
            else:
                if buf:
                    merged.append(("".join(buf), buf_s, buf_e))
                    buf, buf_s = [], None
                if ch.strip():
                    merged.append((ch, s, e))
        if buf:
            merged.append(("".join(buf), buf_s, buf_e))
        return merged

    # ── 公开 API ───────────────────────────────────────────────────────
    def transcribe(self, audio, sample_rate: int | None = None) -> ASRResult:
        """audio：文件路径（任意 ffmpeg 格式）或 16 kHz 单声道 float32 波形。"""
        if self._enc is None:
            raise RuntimeError("先调用 initialize()")
        if isinstance(audio, (str, Path)):
            wav, sr = self._read_any(Path(audio))
        else:
            wav, sr = audio, sample_rate or 16000
        duration = len(wav) / sr
        t0 = time.perf_counter()
        tokens = self._frames(wav)
        elapsed = time.perf_counter() - t0
        return ASRResult(text=decode_bytes(self.id2bytes, [t for t, _ in tokens]),
                         words=self._words(tokens), duration=duration, elapsed=elapsed)

    @staticmethod
    def _read_any(src: Path):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "in16k.wav"
            subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ar", "16000",
                            "-ac", "1", str(dst)], check=True, capture_output=True)
            wav, sr = sf.read(dst, dtype="float32")
        return wav, sr


def create_asr_engine(**kwargs) -> Qwen3CtcEngine:
    """便捷构造：参数同 ASREngineConfig，返回已 initialize 的引擎。"""
    return Qwen3CtcEngine(ASREngineConfig(**kwargs)).initialize()
