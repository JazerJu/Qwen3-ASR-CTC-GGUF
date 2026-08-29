# AGENTS.md — Qwen3-ASR-CTC-GGUF

## What This Repo Does

Converts [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) + a trained
CTC head into ONNX (fp32 / fp16 / int4) + GGUF for offline inference. Hybrid
runtime: ONNX Runtime for Encoder + CTC, llama.cpp for the LLM decoder.

Language: Python only. Structure mirrors
[Fun-ASR-GGUF](https://github.com/HaujetZhao/Fun-ASR-GGUF).

## Export Pipeline (strict order)

Scripts run sequentially — each consumes the previous output. All output goes to
`./model/`. Paths live in `export_config.py`; nothing else hardcodes a path.

| Step | Script | What it does | Key outputs |
|------|--------|-------------|-------------|
| 1 | `01-Export-ONNX-FP32.py` | Encoder + CTC → fp32 ONNX; generate `tokens.txt` | `*.fp32.onnx`, `model/tokens.txt` |
| 2 | `02-Optimize-ONNX.py` | ORT transformer fusion; dropped if the cosine gate fails | `*.opt.fp32.onnx` |
| 3 | `03-Quantize-ONNX.py` | fp16 (DirectML) + int4 MatMulNBits (CUDA/CPU) | `*.fp16.onnx`, `*.q4.onnx` |
| 4 | `04-Export-Decoder-GGUF-FP16.py` | `thinker.model.*` → Qwen3ForCausalLM → fp16 GGUF | `Qwen3-ASR-Decoder.fp16.gguf` |
| 5 | `05-Quantize-Decoder-GGUF.py` | `llama-quantize` → q5_k_m | `Qwen3-ASR-Decoder.q5_k_m.gguf` |
| 6 | `06-Inference.py` | End-to-end CLI / verification | console + optional SRT |

Steps 7 and 8 are **gates, not pipeline steps** — optional but strongly advised:

| Gate | Script | Threshold |
|------|--------|-----------|
| 7 | `07-Validate-Encoder.py` | ONNX vs PyTorch on lengths ≠ trace input, cosine ≥ 0.999 |
| 8 | `08-Gate-Int4.py` | int4 chain vs fp32 chain, greedy text diff ≤ 1% |

**Prerequisites before step 1** — see README «准备输入»:
`QWEN3_ASR_DIR` → official weights, `QWEN3_CTC_DIR` → trained CTC head
(default `./ctc`). Steps 4–5 additionally need `LLAMA_CPP_DIR`.

## Public API (Runtime)

```python
from qwen3_asr_ctc import create_asr_engine, ASREngineConfig, Qwen3CtcEngine

engine = create_asr_engine(
    encoder_onnx_path="model/Qwen3-ASR-Encoder.q4.onnx",
    ctc_onnx_path="model/Qwen3-ASR-CTC.q4.onnx",
    tokens_path="model/tokens.txt",
    preprocessor_path="preprocessor",
    use_gpu=True,
)
result = engine.transcribe("input.mp3")   # -> ASRResult(text, words, duration, elapsed, rtf)
```

`qwen3_asr_ctc/` layout:

| Module | Role |
|---|---|
| `engine.py` | ONNX sessions, mel → encoder → CTC → bytes, word timestamps |
| `tokens.py` | byte-level detokenization (`bytes_to_unicode`, `load_tokens`) |
| `modeling_ctc.py` | PyTorch `CTCDecoder` + the attention-mask patch (export only) |
| `compat.py` | `qwen_asr` 0.0.6 patches; **import before `qwen_asr`** |
| `srt.py` | dependency-free SRT writer |

## Invariants — do not "fix" these

1. **13 fps, not 50.** `valid = full*13 + ceil(leave/8)` where
   `full, leave = divmod(mel_frames, 100)`. Using `T/8` or a 50 fps formula
   overestimates frames 4× and silently truncates the transcript.
2. **`padding=False` on the feature extractor**, then zero-pad to the 3000-frame
   bucket yourself and pass the *real* frame count as `feature_length`.
3. **`patch_qwen3_attention_mask()` before loading the encoder.** Without it
   batch-vs-single cosine is 0.81–0.88. Affects CUDA too.
4. **Detokenize through bytes**, never `"".join(pieces)` — the compact vocab
   keeps 89 byte primitives and a CJK char may span several tokens.
5. **Read CTC hyperparameters from `ctc/config.json`.** v1 `ffn_hidden=128`,
   v2 `ffn_hidden=2048`. Hardcoding breaks `load_state_dict`.
6. **fp16 conversion must consume the RAW fp32 graph, not the 02-Optimize
   output.** Fused `com.microsoft` ops (SkipLayerNormalization / BiasGelu)
   break `convert_float_to_float16`'s cast insertion and yield a mixed graph.
7. **int4 must also target `Gemm`, not only `MatMul`.** The Qwen encoder exports
   its Linears as Gemm; MatMul-only quantization leaves it ~98% uncompressed.

## Conventions

- Model filenames: `<Stem>.<precision>.onnx`, precision ∈
  `fp32 | opt.fp32 | fp16 | q4`. Build paths with `export_config.onnx()`.
- Every export step prints a `GATE: PASS/FAIL` line and exits non-zero on FAIL.
  **Do not report success without running the step and reading that line.**
- `model/*.onnx`, `*.gguf` and `ctc/` are gitignored — only `tokens.txt` is
  tracked. Large artifacts go to Releases or HF.
- Step 5 assumes a POSIX `llama-quantize`; on Windows it is `.exe`.
