"""运行时引擎。

两层：
  * `Qwen3CtcEngine` —— mel -> 编码器 -> CTC -> 字节反解，CTC 首遍 + 词级时间戳。
  * `Qwen3ASREngine` —— 在上面套 LLM 二遍（q5_k_m GGUF，llama.cpp）+ 音素热词 +
    NW 时间戳对齐；不给 `decoder_gguf_path` 就退化成纯 CTC。

三个必须照做、否则静默出错的点（详见 README「四个坑」）：

1. **特征不补到 30 秒**：`padding=False` 取真实长度，再自己零填到导出时的
   3000 帧桶，并把真实帧数作为 `feature_length` 传进去。
2. **帧率 13 fps**：有效输出帧数 = `full*13 + ceil(leave/8)`，不是 `T/8`。
3. **反解走字节**：见 tokens.py。
第四个是加载顺序：**llama.cpp 要先于 ONNX Runtime CUDA 初始化**，反过来 SIGSEGV。
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .schema import DecodeResult, RecognitionStream
from .tokens import decode_bytes, load_tokens

FRAME_SEC = 1.0 / 13.0
# CTC 尖峰式发射的系统性偏置，对 MFA 词级真值实测（v1：起点晚 100ms、终点早 78.5ms）
DEFAULT_START_BIAS = 0.100
DEFAULT_END_BIAS = 0.0785
SEGMENT_SEC = 30.0   # 编码器按 30 秒桶导出，更长的文件按 30 秒切段


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
    ctc_argmax_in_graph: bool = True   # 加载时给 CTC 头追加 ArgMax，只回传 int64 帧级 id（见 _ctc_session）
    start_bias: float = DEFAULT_START_BIAS
    end_bias: float = DEFAULT_END_BIAS
    # ---- LLM 二遍（可选）----
    decoder_gguf_path: Optional[str] = None
    enable_ctc: bool = True           # False = 纯 decoder（不出热词、时间戳），对应 CapsWriter 的 'qwen_asr'
    llm_use_gpu: bool = True
    n_ctx: int = 4096
    n_ubatch: int = 512
    n_threads: Optional[int] = None
    n_predict: int = 256
    # ---- 热词 ----
    hotwords: List[str] = field(default_factory=list)
    similar_threshold: float = 0.72   # 进 prompt 的门槛
    replace_threshold: float = 0.85   # 直接替换的门槛（CTC 文本本身不改，只出候选）
    max_hotwords: int = 10


@dataclass
class ASRResult:
    text: str
    words: list = field(default_factory=list)   # [(词, 起秒, 止秒)]
    duration: float = 0.0
    elapsed: float = 0.0
    ctc_text: str = ""
    hotwords: list = field(default_factory=list)

    @property
    def rtf(self) -> float:
        return self.elapsed / self.duration if self.duration else float("nan")


class Qwen3CtcEngine:
    FRAME_SEC = FRAME_SEC

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
        self._ctc = self._ctc_session(self.cfg.ctc_onnx_path, providers, self.cfg.ctc_argmax_in_graph)
        self._fe = WhisperFeatureExtractor.from_pretrained(self.cfg.preprocessor_path)
        self.id2bytes = load_tokens(self.cfg.tokens_path)
        self._np = np
        return self

    # ── 内部 ───────────────────────────────────────────────────────────
    @staticmethod
    def _ctc_session(path, providers, argmax_in_graph=True):
        """加载 CTC 头；argmax_in_graph 时在图末尾追加 ArgMax，只回传 int64 的帧级 id。

        logits [1, 390, 72468] float32 在 30 秒音频上是 113MB，每次都从显存拷回主机再 numpy argmax，
        CTC 头这段要 ~19ms；ArgMax 进图后 ~8ms，逐帧结果一致（5070 Ti，CUDA EP）。
        已经是 int64 输出的模型（比如 CapsWriter 那种把 argmax 导进图的）原样加载。
        """
        import onnxruntime as ort
        if not argmax_in_graph:
            return ort.InferenceSession(path, providers=providers)
        import onnx
        from onnx import TensorProto, helper
        m = onnx.load(path)                       # 外置 .onnx.data 会一并读进来，序列化后不再依赖路径
        out = m.graph.output[0]
        if out.type.tensor_type.elem_type == TensorProto.INT64:
            return ort.InferenceSession(path, providers=providers)
        m.graph.node.append(helper.make_node("ArgMax", [out.name], ["indices"], axis=-1, keepdims=0))
        del m.graph.output[:]
        m.graph.output.append(helper.make_tensor_value_info("indices", TensorProto.INT64, ["batch", "time"]))
        return ort.InferenceSession(m.SerializeToString(), providers=providers)

    @staticmethod
    def _valid_frames(mel_len: int) -> int:
        """真实 mel 帧数 -> 编码器有效输出帧数（每 100 帧出 13 帧）。"""
        full, leave = divmod(mel_len, 100)
        return full * 13 + (0 if leave == 0 else (leave - 1) // 8 + 1)

    def encode(self, audio):
        """16 kHz 波形 -> 音频 embedding [valid, 2048]（也就是喂给 LLM 的那份）。"""
        np = self._np
        mel = self._fe(audio, sampling_rate=16000, padding=False,
                       return_tensors="np").input_features[0]
        t = mel.shape[1]
        padded = np.zeros((mel.shape[0], self.cfg.mel_frames), dtype=np.float32)
        padded[:, :t] = mel
        enc = self._enc.run(None, {"input_features": padded,
                                   "feature_length": np.array([t], np.int64)})[0]
        return enc[: self._valid_frames(t)].astype(np.float32)

    def ctc_tokens(self, audio_embd):
        """embedding -> [(紧凑 id, 帧)]，贪心折叠、去 blank/unk。"""
        np = self._np
        out = self._ctc.run(None, {"enc_output": audio_embd[None]})[0]
        ids = out[0] if out.dtype == np.int64 else np.argmax(out[0], axis=-1)   # 图里已 ArgMax 则直接是 id
        out, prev = [], None
        for frame, tok in enumerate(ids):
            tok = int(tok)
            if tok == prev:
                continue
            prev = tok
            if tok not in (self.cfg.blank_id, self.cfg.unk_id):
                out.append((tok, frame))
        return out

    def _frames(self, audio):
        return self.ctc_tokens(self.encode(audio))

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
        text = decode_bytes(self.id2bytes, [t for t, _ in tokens])
        return ASRResult(text=text, words=self._words(tokens), duration=duration,
                         elapsed=elapsed, ctc_text=text)

    @staticmethod
    def _read_any(src: Path):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "in16k.wav"
            subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ar", "16000",
                            "-ac", "1", str(dst)], check=True, capture_output=True)
            wav, sr = sf.read(dst, dtype="float32")
        return wav, sr


class Qwen3ASREngine:
    """CTC 首遍 + 可选 LLM 二遍。接口对齐 GLM-ASR-CTC-GGUF 的 GLMASREngine：
    create_stream / decode_stream / transcribe / update_hotwords / cleanup。"""

    def __init__(self, config: ASREngineConfig):
        from .pipeline import Qwen3Pipeline

        self.config = config
        self.ctc = Qwen3CtcEngine(config)
        self.pipeline = Qwen3Pipeline(self.ctc, config)
        if config.decoder_gguf_path:
            self.pipeline.load_llm()          # 先 llama
        self.ctc.initialize()                 # 后 ORT CUDA —— 顺序不能反
        self.create_stream = lambda **_: RecognitionStream()
        self.decode_stream = self.pipeline.decode_stream

    @property
    def has_decoder(self) -> bool:
        return self.pipeline.model is not None

    def update_hotwords(self, hotwords: List[str]):
        self.pipeline.update_hotwords(hotwords)

    def transcribe(self, audio, sample_rate: int | None = None, language: Optional[str] = None,
                   context: Optional[str] = None, **kw) -> ASRResult:
        """文件或波形；超过 30 秒按 30 秒切段（编码器是 30 秒桶），时间戳按段偏移。"""
        if isinstance(audio, (str, Path)):
            wav, sr = Qwen3CtcEngine._read_any(Path(audio))
        else:
            wav, sr = audio, sample_rate or 16000
        duration = len(wav) / sr
        seg = int(SEGMENT_SEC * sr)
        t0 = time.perf_counter()
        texts, words, ctc_texts, hotwords = [], [], [], []
        for i, s in enumerate(range(0, len(wav), seg)):
            stream = RecognitionStream(sample_rate=sr)
            stream.accept_waveform(sr, wav[s:s + seg])
            r: DecodeResult = self.decode_stream(stream, language=language, context=context,
                                                 timestamp_offset=i * SEGMENT_SEC, **kw)
            texts.append(r.text)
            ctc_texts.append(r.ctc_text)
            hotwords += [h for h in r.hotwords if h not in hotwords]
            for k, (tok, st) in enumerate(r.aligned):
                nxt = r.aligned[k + 1][1] if k + 1 < len(r.aligned) else st + FRAME_SEC
                words.append((tok, float(st), float(max(nxt, st))))
        elapsed = time.perf_counter() - t0
        return ASRResult(text="".join(texts), words=words, duration=duration, elapsed=elapsed,
                         ctc_text="".join(ctc_texts), hotwords=hotwords)

    def cleanup(self):
        self.pipeline.ctx = None
        self.pipeline.model = None
        self.ctc._enc = self.ctc._ctc = None


def create_asr_engine(**kwargs) -> Qwen3ASREngine:
    """便捷构造：参数同 ASREngineConfig，返回已初始化的引擎。
    给 decoder_gguf_path 就是两遍，不给就是纯 CTC。"""
    return Qwen3ASREngine(ASREngineConfig(**kwargs))
