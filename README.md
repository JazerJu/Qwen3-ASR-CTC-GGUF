# Qwen3-ASR-CTC-GGUF

把 [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) 转成可以在本地
高效运行的格式，实现**离线语音识别**。编码器与 CTC 头走 ONNX Runtime，
LLM 解码器走 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 GGUF。

结构参照 [Fun-ASR-GGUF](https://github.com/HaujetZhao/Fun-ASR-GGUF)。

### 核心特性

- ✅ **纯本地运行** — 无需网络
- ✅ **三档精度** — 每个 ONNX 模型都产出 `fp32` / `fp16` / `q4`(int4)，按硬件选
- ✅ **13 fps 帧率** — 编码器每秒只出 13 帧，CTC 头的计算量只有 50 fps 方案的 1/3.85
- ✅ **字级时间戳** — CTC 帧位置直接给时间戳，并减掉实测的常数偏置
- ✅ **每步有门** — 导出不是"跑完就算过"，每步都有数值门（余弦 / 字节往返 / 文本差异）

| 产物 | fp32 | fp16 | q4 (int4) |
|---|---|---|---|
| Encoder（317.5 M 参数） | 1270 MB | 636 MB | **186 MB** |
| CTC 头（v1 48.3 M） | 193 MB | 97 MB | **26 MB** |
| LLM Decoder（1.72 B） | — | 3447 MB (GGUF) | **1258 MB** (q5_k_m) |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> `ffmpeg` 需系统安装（读任意音频格式）。GPU 推理把 `onnxruntime` 换成
> `onnxruntime-gpu`（CUDA）或 `onnxruntime-directml`（Windows iGPU）。

第 4、5 步（GGUF）还需要编译好的 llama.cpp，用 `LLAMA_CPP_DIR` 指过去。

### 2. 准备输入

```bash
# 官方权重（编码器 + LLM 解码器都在这一份里）
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-HF
export QWEN3_ASR_DIR=$PWD/Qwen3-ASR-HF

# 训练好的 CTC 头（二选一）
huggingface-cli download JazerJu/qwen3-asr-ctc    --local-dir ./ctc   # v1  48.3M
huggingface-cli download JazerJu/qwen3-asr-ctc-r2 --local-dir ./ctc   # v2  58.2M
```

两版 CTC 头的差别与实测对比见
[JazerJu/qwen3-asr-ctc-r2](https://huggingface.co/JazerJu/qwen3-asr-ctc-r2)。
**超参一律从 `ctc/config.json` 读，不写死** —— v1 的 `ffn_hidden` 是 128、
v2 是 2048，写死会 `load_state_dict` 形状不匹配。

所有路径集中在 [`export_config.py`](export_config.py)，也可用环境变量覆盖。

### 3. 导出与量化（6 步走）

按顺序执行，每步吃上一步的产物，全部输出到 `model/`：

```bash
python 01-Export-ONNX-FP32.py        # 编码器 + CTC -> fp32 ONNX，生成 tokens.txt
python 02-Optimize-ONNX.py           # ORT 算子融合（DirectML 提速）
python 03-Quantize-ONNX.py           # -> fp16 / q4 三档
python 04-Export-Decoder-GGUF-FP16.py  # LLM -> fp16 GGUF
python 05-Quantize-Decoder-GGUF.py   # -> q5_k_m GGUF
python 06-Inference.py input.mp3     # 端到端验证
```

两个可选的**数值门**（不是流水线步骤，但强烈建议跑）：

```bash
python 07-Validate-Encoder.py        # ONNX 编码器 vs PyTorch，不同长度余弦 >= 0.999
python 08-Gate-Int4.py               # int4 链 vs fp32 链，贪心文本差异 <= 1%
```

### 4. 运行识别

默认两遍：CTC 首遍出词级时间戳和热词候选，LLM 二遍（q5_k_m GGUF）出最终文本，再用 NW 把 CTC 的时间戳对到 LLM 文本上。不给 `decoder_gguf_path` 就只跑 CTC 首遍。

**精度怎么选**：N 卡走 CUDA EP 用 **fp16**——q4 的 `MatMulNBits` 是给 DirectML / 核显省显存的，在 CUDA EP 上反而比 fp16 慢一倍（5070 Ti 上 encoder 21.8 vs 10.1ms）。q4 留给 Windows DML 和显存紧的机器。CTC 头加载时会在图末尾追加 ArgMax，只回传帧级 id（省掉 30 秒 113MB 的 logits 回传，19 → 8ms），`ctc_argmax_in_graph=False` 可关。

```python
from qwen3_asr_ctc import create_asr_engine

engine = create_asr_engine(
    encoder_onnx_path="model/Qwen3-ASR-Encoder.fp16.onnx",   # CUDA 用 fp16；DML/核显用 q4
    ctc_onnx_path="model/Qwen3-ASR-CTC.fp16.onnx",
    tokens_path="model/tokens.txt",
    preprocessor_path="preprocessor",
    decoder_gguf_path="model/Qwen3-ASR-Decoder.q5_k_m.gguf",   # 去掉这行就是纯 CTC
    hotwords=["Claude Code", "科大讯飞"],                        # 可选：CTC 首遍音素匹配到的热词会进 prompt
)

result = engine.transcribe("input.mp3")
print(result.text)                    # LLM 二遍文本（纯 CTC 模式下 = CTC 文本）
print(result.ctc_text)                # CTC 首遍文本
print(result.hotwords)                # 这段音频里匹配到、进了 prompt 的热词
print(result.words)                   # [(词, 起秒, 止秒)]
print(f"RTF {result.rtf:.3f}")
```

命令行：

```bash
python 06-Inference.py input.mp3                          # 两遍
python 06-Inference.py clip.wav --srt out.srt --hotwords hot.txt
python 06-Inference.py clip.wav --no-decoder              # 只 CTC
```

流式接口和 CapsWriter / Fun-ASR-GGUF 同形：`create_stream()` → `accept_waveform(16000, wav)` → `decode_stream(stream, context=..., language=...)`，结果在 `stream.result.text / tokens / timestamps`，返回的 `DecodeResult.timings` 有 encode / ctc / inject / llm_generate / align 各段耗时。

---

## 工作原理

```
音频输入（任意格式，ffmpeg 转 16 kHz 单声道；超过 30 秒按 30 秒切段）
    ↓
  mel 特征 128×T   ——  不补到 30 秒，取真实长度后零填到 3000 帧桶
    ↓
┌──────────────────────────────────────────────────┐
│  Encoder (ONNX)   30 秒桶 + feature_length 标量    │
│                   -> 每 100 mel 帧出 13 帧         │  ──┐
│  CTC 头 (ONNX)    -> 逐帧 logits -> argmax         │    │ 音频 embedding [T, 2048]
└──────────────────────────────────────────────────┘    │
    ↓                      ↓                            │
  贪心折叠 + 字节反解      帧位置 -> 起始秒（减常数偏置）    │
    ↓                                                   │
  CTC 首遍文本 ──► 音素热词匹配（PhonemeCorrector）         │
    │                      ↓ 命中的热词                  │
    │   ┌──────────────────────────────────────────────┐ │
    │   │ LLM 二遍 (llama.cpp, q5_k_m GGUF)             │◄┘
    │   │ ChatML：system(热词/上下文) user(<|audio_start|> │
    │   │ embedding <|audio_end|>) assistant <asr_text>  │
    │   │ 4 段 M-RoPE 位置 [pos,pos,pos,0]                │
    │   └──────────────────────────────────────────────┘
    │                      ↓ 最终文本
    └──► NW 对齐（CTC 字符时间戳 -> LLM 文本，逐行向量化）
                           ↓
                 识别文本 + 词级时间戳 / SRT
```

CTC 首遍文本本身不改（热词只出候选进 prompt，LLM 从音频 embedding 重新生成，不消费 CTC 文本），它只做两件事：给热词匹配当输入，给时间戳对齐当锚点。

加载顺序有硬约束：**llama.cpp 要先于 ONNX Runtime CUDA 初始化**，反过来同进程 SIGSEGV（Fun-ASR-GGUF 也有同样的记录），`Qwen3ASREngine` 已按这个顺序写。停止符用 `<|im_end|>` / `<|endoftext|>`，不能用 GGUF 元数据里的 eos——这份 GGUF 的 eos 被标成了 11（英文逗号），拿它当停止符会在第一个 "," 处截断。

---

## 四个坑

Qwen3-ASR 和 GLM-ASR / Whisper 的约定不同，**任何一处照抄都会得到
"能跑、不报错、结果是错的"**。仓库里都处理了，自己改时务必照做。

### ① 帧率是 13 fps，不是 50 fps

编码器是 3 层 stride-2 的 conv2d（8 倍降采样），但官方长度公式不是 `T/8`，
而是**每 100 个 mel 帧出 13 帧**：

```python
full, leave = divmod(mel_frames, 100)
valid = full * 13 + (0 if leave == 0 else (leave - 1) // 8 + 1)
```

有效帧移 = **1/13 秒 = 76.923 ms**（不是 8×10 ms）。
沿用 50 fps 的算法会把帧数**高估 4 倍** —— logits 只有 1/4 长，转写被腰斩，
**不报任何错**。

### ② 特征不能补到 30 秒

`WhisperFeatureExtractor` 默认补到 30 秒（`n_samples=480000`）。必须
`padding=False` 取真实长度，再自己零填到 3000 帧桶，并把**真实帧数**作为
`feature_length` 传进模型。用 extractor 自带的补齐会让 `feature_length` 对不上。

### ③ attention mask 必须显式构造

`qwen-asr` 0.0.6 的 `_prepare_attention_mask` 定义了却**从来没被调用**，
`cu_seq_lens_q/k` 只有 flash_attention_2 后端消费。不打补丁时，同一条音频
单条推理 vs 批推理的隐层余弦只有 0.81–0.88，打完 0.9998+。**CUDA 上同样中招。**

导出时这个 mask 会按追踪时的形状固化，所以 `07-Validate-Encoder.py` 用
**与追踪长度不同**的真实音频把关（cos >= 0.999）。**这是整条链路唯一有真实
失败风险的一步。**

### ④ 反词表化必须走字节

紧凑词表里保留了 89 个字节原语做兜底，一个汉字可能由多个字节 token 拼成。
按字符串 `"".join()` 拼会得到乱码。`model/tokens.txt` 存的是每个紧凑 id 的
**原始字节的 base64**，解码时拼 bytes 再统一 `decode("utf-8")`。
`01-Export-ONNX-FP32.py` 会用 500 条多语种句子验证字节路径与
`tokenizer.decode` 逐字节一致。

---

## 时间戳

CTC 是**尖峰式**发射：概率集中在词的中间，所以词起点系统性偏晚、终点偏早。
对 LibriSpeech 的 MFA 词级真值实测（1,500 句 / 29,621 词）：

| | v1 | v2 |
|---|---|---|
| 词起始 中位偏置 | +100.0 ms | +96.5 ms |
| 词结束 中位偏置 | −78.5 ms | −94.6 ms |
| 起始 去偏置后 中位\|误差\| | 50.8 ms | **40.3 ms** |
| 起始 去偏置后 ≤100 ms | 77.6% | **92.2%** |

这两个偏置是**常数**，引擎里已经减掉（`ASREngineConfig.start_bias/end_bias`）。
**做词级时间戳用起点，别用终点** —— v2 的终点精度比 v1 差。

中文没有词级真值可校，无法给出中文的偏置常数。

---

## 识别率参考

CTC 首遍（贪心解码，无 LLM 二遍），PyTorch 侧实测，ONNX 链路应当复现
（int4 偏差在 1% 相对以内）：

| 测试集 | 指标 | v1 | v2 |
|---|---|---|---|
| AISHELL-1 test | CER | 5.31% | 5.53% |
| AISHELL-1 dev | CER | 4.37% | 4.54% |
| LibriSpeech test-clean | WER | 6.93% | 6.53% |
| LibriSpeech test-other | WER | 12.40% | 11.93% |
| ASCEND test（中英混说） | MER | 14.53% | 14.47% |

**v2 不是普遍更好，是一笔交易**：英文变好、中文变差（按句配对自举 2000 次，
中文退步的 95% CI 为 [+0.07, +0.34] pp，不含 0）。中文为主的场景建议先用 v1
在自有集上对比。

> **「按句配对自举」和「95% CI」是什么**
>
> 测试集就固定那几千句，万一恰好抽到的这批句子对某个模型友好呢？**自举**就是
> 模拟「换一批句子会怎样」：从原测试集里**有放回**地随机抽同样多的句子，组成一个
> 「平行世界的测试集」，算一遍错误率，重复 2000 次，看这 2000 个结果的散布。
>
> **按句**而不是按字 —— 错误在句子内部是聚集的（一句崩了往往连着错十几个字），
> 按字当独立样本会把有效样本量高估好几倍，置信区间算得过窄。
>
> **配对** —— 每次抽出的那批句子，**两个模型都在同一批上算**。这样「这批句子难不难」
> 对两边影响相同，做差时抵消掉，剩下的才是模型的真实差异。实测在 aishell1_test 上
> 配对能把 CI 宽度压到非配对的一半（0.263pp vs 0.529pp），同一个真实差值
> +0.208pp，配对能下结论、非配对跨 0 判不出方向。
>
> **95% CI**（置信区间）就是这 2000 个差值排序后中间 95% 的范围。**不含 0** 表示
> 换哪批句子结论都一样，差异是稳的；**跨 0** 表示有些平行世界甲更好、有些乙更好，
> 方向判不出来，只能当噪声。
>
> 实现见 [glm-asr-ctc-train](https://github.com/JazerJu/glm-asr-ctc-train) 的
> `scripts/bootstrap_compare.py`。


对照：同数据同超参训练的 GLM-ASR-CTC 在这五个集上分别是
4.71 / 4.09 / 4.88 / 9.99 / 11.84%，仍全面领先。

---

## 许可

代码 Apache-2.0。模型权重按各自上游许可：Qwen3-ASR-1.7B 为 Apache-2.0，
CTC 头见对应 HF 仓库。
