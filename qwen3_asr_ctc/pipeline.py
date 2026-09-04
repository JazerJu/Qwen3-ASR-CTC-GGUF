"""Qwen3-ASR 两遍流水线：CTC 首遍 -> 音素热词 -> 热词进 prompt -> LLM 二遍 -> NW 对齐。

和 GLM-ASR-CTC-GGUF 的 pipeline 同构，三处 Qwen 特有：
  1. prompt 是 ChatML（<|im_start|>system/user/assistant），音频 embedding 夹在
     <|audio_start|> … <|audio_end|> 之间，assistant 段以 <asr_text> 起头；
  2. 位置编码是 4 段 M-RoPE：pos_arr = [pos, pos, pos, 0]，长度 4n；
  3. 热词/上下文放在 system 段（上游 CapsWriter 的 qwen_asr 后端就是这么放的）。
加载顺序有硬约束：**llama.cpp 必须先于 ONNX Runtime CUDA 初始化**，反过来同进程
SIGSEGV（Fun-ASR-GGUF 也有同样的记录）。
"""
from __future__ import annotations

import codecs
import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from . import llama_cpp_bindings as llama
from .ctc_aligner import CTCAligner
from .hotword.hot_phoneme import PhonemeCorrector
from .schema import CTCResult, DecodeResult, LLMDecodeResult, RecognitionStream, Timings

logger = logging.getLogger("qwen3_asr_ctc")

DEFAULT_SYSTEM = "You are a helpful assistant."


class Qwen3Pipeline:
    def __init__(self, ctc_engine, cfg):
        """ctc_engine：已 initialize 的 Qwen3CtcEngine；cfg：ASREngineConfig。

        注意这里假定 llama 已经在 ctc_engine 之前加载（见 engine.py 的顺序）。
        """
        self.cfg = cfg
        self.ctc = ctc_engine
        self.model: Optional[llama.LlamaModel] = None
        self.ctx = None
        self.table = None
        self.corrector = PhonemeCorrector(threshold=cfg.replace_threshold,
                                          similar_threshold=cfg.similar_threshold)
        self.update_hotwords(cfg.hotwords or [])

    # ── 加载 ──────────────────────────────────────────────────────────
    def load_llm(self):
        cfg = self.cfg
        self.model = llama.LlamaModel(cfg.decoder_gguf_path, use_gpu=cfg.llm_use_gpu)
        self.table = llama.get_token_embeddings_gguf(cfg.decoder_gguf_path)
        self.ctx = llama.LlamaContext(self.model, n_ctx=cfg.n_ctx, n_batch=cfg.n_ctx,
                                      n_ubatch=cfg.n_ubatch, n_threads=cfg.n_threads)
        tid = lambda s: self.model.tokenize(s, add_special=False, parse_special=True)[0]
        self.ID_IM_START = tid("<|im_start|>")
        self.ID_IM_END = tid("<|im_end|>")
        self.ID_AUDIO_START = tid("<|audio_start|>")
        self.ID_AUDIO_END = tid("<|audio_end|>")
        self.ID_ASR_TEXT = tid("<asr_text>")
        # 不能用 model.eos_token：这份 GGUF 的 eos 元数据是 11（英文逗号 ','），
        # 拿它当停止符会在第一个 "," 处截断。ChatML 的真正结束符是 <|im_end|>，
        # 再带上 <|endoftext|> 兜底。
        self.stop_ids = {self.ID_IM_END, tid("<|endoftext|>")}
        logger.info(f"[LLM] {cfg.decoder_gguf_path} n_embd={self.model.n_embd} "
                    f"specials im_start={self.ID_IM_START} audio_start={self.ID_AUDIO_START} "
                    f"asr_text={self.ID_ASR_TEXT}")
        return self

    def update_hotwords(self, hotwords: List[str]):
        clean = [h.strip() for h in (hotwords or []) if h and h.strip()]
        self.corrector.update_hotwords(clean)
        logger.info(f"[CTC] 热词已更新 (热词数: {len(clean)})")

    # ── CTC 首遍 ──────────────────────────────────────────────────────
    def _ctc_units(self, tokens: List[Tuple[int, int]]) -> List[CTCResult]:
        """(紧凑 id, 帧) -> 逐字符 CTCResult。字节走增量 utf-8 解码：一个汉字可能
        跨两三个字节 token，字符在哪个 token 上凑齐就记那个 token 的起始秒。"""
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        units: List[CTCResult] = []
        last = 0.0
        for tid, frame in tokens:
            start = max(frame * self.ctc.FRAME_SEC - self.cfg.start_bias, 0.0)
            last = start
            for ch in dec.decode(self.ctc.id2bytes.get(tid, b"")):
                units.append(CTCResult(text=ch, timestamp=start))
        for ch in dec.decode(b"", final=True):
            units.append(CTCResult(text=ch, timestamp=last))
        return units

    def _hotwords(self, ctc_text: str) -> List[str]:
        if not self.corrector.hotwords or not ctc_text:
            return []
        res = self.corrector.correct(ctc_text, k=self.cfg.max_hotwords)
        seen, out = set(), []
        for _, hw, _ in list(res.matchs) + list(res.similars):
            k = hw.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(hw)
        return out

    # ── prompt ────────────────────────────────────────────────────────
    def _prompt(self, hotwords: List[str], language: Optional[str], context: Optional[str]):
        tok = lambda s: self.model.tokenize(s, add_special=False, parse_special=False)
        system = context.strip() if context else DEFAULT_SYSTEM
        if hotwords:
            system += "\n热词列表：[" + ", ".join(hotwords) + "]"
            logger.info(f"热词列表：{', '.join(hotwords)}")
        prefix = ([self.ID_IM_START] + tok("system\n" + system) + [self.ID_IM_END]
                  + [self.ID_IM_START] + tok("user\n") + [self.ID_AUDIO_START])
        head = "assistant\n" + (f"language {language}" if language else "")
        suffix = [self.ID_AUDIO_END, self.ID_IM_END, self.ID_IM_START] + tok(head) + [self.ID_ASR_TEXT]
        return (self.table[prefix].astype(np.float32), self.table[suffix].astype(np.float32),
                len(prefix), len(suffix))

    # ── LLM ───────────────────────────────────────────────────────────
    def _llm(self, full_embd: np.ndarray, temperature: float, top_k: int, top_p: float) -> LLMDecodeResult:
        res = LLMDecodeResult()
        n = full_embd.shape[0]
        t0 = time.perf_counter()
        pos = np.arange(n, dtype=np.int32)
        pos4 = np.concatenate([pos, pos, pos, np.zeros(n, dtype=np.int32)])
        self.ctx.clear_kv_cache()
        batch = llama.LlamaBatch(max(n * 4, 8192), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos4)
        if self.ctx.decode(batch) != 0:
            raise RuntimeError("Qwen3 prefill failed")
        res.t_inject = time.perf_counter() - t0

        t0 = time.perf_counter()
        seed = int(np.random.randint(0, 2**31 - 1)) if temperature > 0 else 0
        pieces: List[int] = []
        ctx_ptr = self.ctx.ptr
        n_vocab = llama.llama_vocab_n_tokens(self.model.vocab)
        # temperature=0 时绕开 llama_sampler_sample：它每次调用都要把 n_vocab 个 logits 装成
        # llama_token_data 候选数组（59k 词表 ~0.18ms、152k ~0.21ms），greedy 只是在上面线性扫一遍；
        # 对同一块内存直接 numpy argmax 只要 ~17µs，每 token 省约 10%。llama_get_logits_ith 内部会
        # synchronize；指针指向 llama.cpp 自己的缓冲区，下一次 decode 会覆盖，必须当场取完。
        def pick_greedy():
            logits = np.ctypeslib.as_array(llama.llama_get_logits_ith(ctx_ptr, -1), shape=(n_vocab,))
            return int(logits.argmax())

        with llama.LlamaSampler(temperature=temperature, top_k=top_k, top_p=top_p, seed=seed) as smpl:
            pick = pick_greedy if temperature <= 0 else (lambda: smpl.sample(self.ctx, -1))
            for _ in range(self.cfg.n_predict):
                tid = pick()
                if tid in self.stop_ids:          # 先判停再前向，省一次无用的 decode
                    break
                if self.ctx.decode_token(tid) != 0:
                    break
                pieces.append(tid)
                if len(pieces) >= 30 and len(set(pieces[-30:])) <= 3:
                    res.is_aborted = True         # 复读熔断，同 GLM
                    break
        res.text = self.model.detokenize(pieces)
        res.n_gen = len(pieces)
        res.t_gen = time.perf_counter() - t0
        return res

    # ── 主流程 ────────────────────────────────────────────────────────
    def decode_stream(self, stream: RecognitionStream, language: Optional[str] = None,
                      context: Optional[str] = None, temperature: float = 0.0,
                      top_k: int = 1, top_p: float = 1.0, timestamp_offset: float = 0.0,
                      verbose: bool = False, **_) -> DecodeResult:
        T = Timings()
        t_all = time.perf_counter()
        audio = stream.audio_data
        if audio is None or len(audio) < 1600:
            return DecodeResult(timings=T)

        t0 = time.perf_counter()
        audio_embd = self.ctc.encode(audio)
        T.encode = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.cfg.enable_ctc or self.model is None:
            tokens = self.ctc.ctc_tokens(audio_embd)
            units = self._ctc_units(tokens)
            ctc_text = "".join(u.text for u in units)
            hotwords = self._hotwords(ctc_text)
            logger.info(f"[CTC首遍] {ctc_text!r}")
            if hotwords:
                logger.info(f"[CTC热词匹配] {hotwords}")
        else:                                     # 纯 decoder 模式：没有首遍，也就没有热词和时间戳锚点
            units, ctc_text, hotwords = [], "", []
        T.ctc = time.perf_counter() - t0

        if self.model is None:                    # 未配 decoder：CTC 结果直接出
            aligned = [[u.text, max(u.timestamp + timestamp_offset, 0.0)] for u in units]
            aligned = CTCAligner._merge_english_words(aligned)
            stream.set_result(ctc_text, [a[1] for a in aligned], [a[0] for a in aligned])
            T.total = time.perf_counter() - t_all
            return DecodeResult(text=ctc_text, ctc_text=ctc_text, ctc_results=units, aligned=aligned,
                                n_audio=int(audio_embd.shape[0]), hotwords=hotwords, timings=T)

        t0 = time.perf_counter()
        p_embd, s_embd, n_p, n_s = self._prompt(hotwords, language, context)
        full = np.concatenate([p_embd, audio_embd.astype(np.float32, copy=False), s_embd], axis=0)
        T.prepare = time.perf_counter() - t0

        llm = None
        temp = temperature
        for retry in range(4):
            llm = self._llm(full, temp, top_k, top_p)
            if not llm.is_aborted:
                break
            temp += 0.3
            logger.warning(f"[LLM] 复读熔断，重试 {retry} (temperature -> {temp:.1f})")
        text = llm.text.strip()
        T.inject, T.llm_generate = llm.t_inject, llm.t_gen

        t0 = time.perf_counter()
        aligned = CTCAligner.align(units, text, timestamp_offset=timestamp_offset) if units else []
        T.align = time.perf_counter() - t0

        stream.set_result(text, [a[1] for a in aligned], [a[0] for a in aligned])
        T.total = time.perf_counter() - t_all
        if verbose:
            print(f"[CTC] {ctc_text}\n[LLM] {text}\n[热词] {hotwords}\n"
                  f"[耗时 ms] enc {T.encode*1e3:.0f} ctc {T.ctc*1e3:.0f} inject {T.inject*1e3:.0f} "
                  f"gen {T.llm_generate*1e3:.0f} align {T.align*1e3:.0f} total {T.total*1e3:.0f}")
        return DecodeResult(text=text, ctc_text=ctc_text, ctc_results=units, aligned=aligned,
                            n_audio=int(audio_embd.shape[0]), n_prefix=n_p, n_suffix=n_s,
                            n_gen=llm.n_gen, hotwords=hotwords, timings=T, is_aborted=llm.is_aborted)
