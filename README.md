# Qwen3-ASR-CTC-GGUF

Qwen3-ASR 的自训 CTC 头（冻结官方 encoder）的 ONNX/GGUF 导出与推理仓库，镜像 `../GLM-ASR-CTC-GGUF` 的结构。训练侧产物：<https://huggingface.co/JazerJu/qwen3-asr-ctc>（56,916 步，val_loss 0.615）。

## 模型清单（model/）

| 文件 | 大小 | 说明 |
|---|---|---|
| Qwen3-ASR-Encoder.q4.onnx | 186 MB | 官方 audio tower（317.5M 参数，冻结），int4，固定 30s 桶 + `feature_length` 运行时输入 |
| Qwen3-ASR-CTC.q4.onnx | 26 MB | 我们的 CTC 头（48.3M 参数，72,468 类 BPE），int4 |
| Qwen3-ASR-CTC.fp32.onnx | 193 MB | fp32 对照 |
| Qwen3-ASR-Decoder.q5_k_m.gguf | 1200 MB | LLM 二遍解码器（1.72B，与 GLM/Fun 同量化档） |
| qwen-ctc-tokens.txt | 1.2 MB | 紧凑词表 → 原始字节 base64（§1.4 字节级反词表化） |

## 管线（编号脚本，每步带验证门）

```
01-Export-Encoder-ONNX.py     fp32 encoder，固定30s桶+算术mask     门: 02b 余弦≥0.999 (实测 1.000000×3)
02-Gen-Tokens.py              字节化词表 + 500句round-trip         门: 逐字一致 (实测 280/280)
02b-Validate-Encoder.py       换长真实音频 vs PyTorch
03-Export-CTC-ONNX.py         fp32 CTC 头                          门: 余弦≥0.9999 (实测 1.0×3)
04-Quantize-Int4.py           int4（含 Gemm→MatMul 重写）           门: 05 文本差异≤1% (实测 0.97%)
05-Gate-Int4.py               fp32 vs int4 全链路文本对比
06-Export-Decoder-GGUF.py     Qwen3ForCausalLM 重建 → GGUF → q5_k_m
```

推理入口：

```bash
python main.py clip.wav                    # CTC 首遍 + 词级时间戳（含 -100/-78.5ms 偏置校正）
python main.py input.mp3 --srt out.srt --cpu
```

LLM 二遍（decoder GGUF 已导出、llama.cpp 可加载）尚未接入 main.py——见 `06` 脚本头部说明。

## 本机适配层（qwen3_compat.py）

qwen_asr 0.0.6 按 transformers 4.57 编写，本机 5.12 需四个补丁：config 初始化顺序（validate_token_ids 钩子早于子类属性赋值）、`ROPE_INIT_FUNCTIONS['default']` 移除、RotaryEmbedding 的 `compute_default_rope_parameters`、以及 **from_pretrained 会污染 sinusoids 位置编码 buffer**（5.12 重新初始化钩子写入垃圾值，加载后需用闭式重算覆写）。

## 验证汇总

| 门 | 标准 | 实测 |
|---|---|---|
| Encoder ONNX vs PyTorch（3 长度换长） | 余弦 ≥ 0.999 | 1.000000 / 1.000000 / 1.000000 |
| 字节反词表化 round-trip | 逐字一致 | 280/280 |
| CTC 头 ONNX | 余弦 ≥ 0.9999 | 1.0 ×3 |
| int4 全链路 | 文本差异 ≤ 1% | 0.97%（LibriSpeech 30 条） |
| **LibriSpeech test-clean WER（对表训练侧）** | ≈ 6.93% | **6.92%（500 条）** |

## FLEURS 三方对比（bench-asr-ctc，全量 7,876 句，全部 int4 + CUDA EP）

GLM-CTC 7 胜 / Fun-ASR 2 胜 / **Qwen3-CTC 2 胜（ko_kr 20.3%、yue 30.1%，均为碾压级）**；英语三家用 18.1–18.3% 打平，中文 Fun 略优（9.3 vs 10.7/10.9）。完整表：`../bench-asr-ctc/README.md`。

Qwen encoder 635.0M 参数的一半（317.5M）、50fps 的 13fps（约 1/8 每秒计算量），ko/yue 的优势来自 Qwen3-ASR 底座的多语言预训练；zh/en 落后 GLM 的部分归因训练配方（同数据 512×56,916 vs 256×134,140，对照实验未跑）。
