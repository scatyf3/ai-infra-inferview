---
title: 语音输出：codec token、Talker 与首包延迟
status: draft
tags: [omni, tts, codec, rvq, rtf, streaming]
difficulty: 4
order: 12
related: [/inference/omni-serving, /inference/multimodal-encoder, /inference/metrics-benchmark, /inference/speculative-decoding]
stack: []
---

# 语音输出：codec token、Talker 与首包延迟

> 语音先被 codec 压成离散 token，模型生成 token，再解码回波形。每秒音频要生成多少 token、多码本怎么少走几步自回归、首包延迟和 RTF 由哪几段组成，用 Qwen3-Omni 公开的数字逐项算一遍

## 一句话结论

语音输出 = 自回归地生成 **codec token**，再用一个小的解码器把 token 还原成波形。codec 用残差向量量化（RVQ）：每一帧音频用好几个码本逐级逼近，码率 = 帧率 × 码本数 × 每个码本的 bit 数。帧率决定每秒要走几步自回归，码本数决定每一步要出几个 token，所以模型都在想办法「每步出一整帧」：Moshi 用一个小的 depth transformer 在码本维上自回归，Qwen3-Omni 用 Talker 只出第 0 层、再让一个 80M 的 MTP 模块补齐其余 15 层。serving 上看两个指标：**首包延迟**（各级首个输出的耗时之和，Qwen3-Omni 并发 1 时 234 ms）和 **RTF**（生成一帧的耗时 ÷ 这一帧的时长，必须小于 1 才不卡）。

## 推导

### 1. 语音怎么变成离散 token：RVQ

codec 的编码器把波形压成一串向量，每 $1/f$ 秒一个（$f$ 是帧率），再把每个向量量化成几个整数。**残差向量量化（RVQ）**：

1. 第 1 个码本里找离 $x$ 最近的码字 $c_1$，残差 $r_1 = x - c_1$。
2. 第 2 个码本里找离 $r_1$ 最近的 $c_2$，残差 $r_2 = r_1 - c_2$。
3. 依此类推，$N_q$ 个码本，输出 $N_q$ 个下标。还原时把选中的码字加起来：$\hat x = c_1 + c_2 + \dots + c_{N_q}$。

**一维的例子**：$x = 0.83$，三个码本 $\{-1, 0, 1\}$、$\{-0.3, 0, 0.3\}$、$\{-0.1, 0, 0.1\}$。

| 级 | 要逼近的值 | 选中 | 累计还原 | 残差 |
|---|---|---|---|---|
| 1 | 0.83 | 1.0 | 1.0 | −0.17 |
| 2 | −0.17 | −0.3 | 0.7 | 0.13 |
| 3 | 0.13 | 0.1 | 0.8 | 0.03 |

每多一级，误差小一截。前几级决定大轮廓（内容、音色），后几级补细节。所以很多 codec 只用前几级就能听，码本越多音质越好。

**码率**：每个码本有 $K$ 个码字，一个下标是 $\log_2 K$ bit：

$$
\text{码率} = f \times N_q \times \log_2 K
$$

| codec | 帧率 $f$ | 码本数 $N_q$ | 码本大小 $K$ | 码率 |
|---|---|---|---|---|
| [EnCodec](https://arxiv.org/abs/2210.13438)（24 kHz） | 75 Hz | 2 / 8 / 32 | 1024（10 bit） | 1.5 / 6 / 24 kbps |
| [Mimi](https://arxiv.org/abs/2410.00037)（Moshi） | 12.5 Hz | 8 | 2048（11 bit） | 1.1 kbps |
| Qwen3-Omni 的 codec | 12.5 Hz | 16 | 2048 | 2.2 kbps（按 config 算） |
| [GLM-4-Voice](https://arxiv.org/abs/2412.02612) | 12.5 Hz | 1 | | 175 bps |

Mimi 和 GLM-4-Voice 还把语义信息蒸馏进第一个码本（或唯一的码本），让它更像「语音版的文字」，LLM 更容易学。

### 2. 每秒音频要走几步自回归

LLM 一步出一个 token。$N_q$ 个码本有几种排法：

1. **全部拍平**：每帧 $N_q$ 个 token 依次生成。Qwen3-Omni 是 $12.5 \times 16 = 200$ 步/秒，太多。
2. **只用一个码本**：GLM-4-Voice、CosyVoice 2（25 Hz）这类单码本 tokenizer，每秒 12.5 或 25 步，再用一个 flow matching 模型把 token 补成高质量的 mel 频谱。
3. **两层自回归**：大模型每帧只走一步，帧内的 $N_q$ 个码本交给一个小模型自回归。
   - **Moshi 的 RQ-Transformer**：7B 的 temporal transformer 每帧一步（12.5 步/秒），输出一个向量；一个小的 depth transformer（6 层、hidden 1024）拿这个向量，在码本维上依次出 8 个 token。
   - **Qwen3-Omni**：Talker（3B 总参数、0.3B 激活的 MoE）每帧一步，只预测第 0 个码本；MTP 模块（80M 的 dense transformer）接着把剩下 15 个码本补齐。大模型每秒只走 12.5 步，小模型每秒走 $12.5 \times 15$ 步，但小模型的每一步便宜得多。

帧率越低，每秒的步数越少，但每帧要压进的信息越多，对 codec 和模型的要求越高。12.5 Hz（80 ms 一帧）是近两年的常见选择。

### 3. 文本和语音怎么对齐

一句话的文本 token 和语音 token 速度不一样：说一个字要好几帧。几种做法：

1. **Thinker-Talker**（[Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)）：Thinker 是正常的 LLM，出文本；Talker 拿 Thinker 的隐状态和它采样出的文本 token 当条件，自己决定出多少语音 token，不需要字级或时间戳级的对齐。
2. **Qwen3-Omni 的变化**：Talker 不再吃 Thinker 的高层文本表示，只以多模态特征和当前轮流式出来的文本为条件，所以 Thinker 和 Talker 可以用不同的 system prompt（比如 Thinker 管内容、Talker 管说话风格）。
3. **固定比例交错**：GLM-4-Voice 按 13 个文本 token、26 个语音 token 交替生成；CosyVoice 2 是 5 : 15。
4. **Moshi 的 inner monologue**：每一帧先出一个和时间对齐的文本 token（大部分是 PAD），再出这一帧的语音 token。

### 4. token 还原成波形

| 模型 | 解码器 | 能不能逐帧出声 |
|---|---|---|
| Qwen2.5-Omni | flow matching DiT 出 mel，再用 BigVGAN 出波形 | 按块：每块的感受野是 4 块（往回看 2 块、往前看 1 块），要等下一块到了才能出这一块 |
| Qwen3-Omni | 因果卷积网络 Code2Wav（200M），每帧上采样 1920 倍（24 kHz ÷ 12.5 Hz） | 能：只看左边，Talker 每出一帧就能解码这一帧 |
| Mimi（Moshi） | 因果卷积 + transformer | 能：理论延迟 160 ms，实测 200 ms |

「往前看一块」意味着首包至少要多等一块的生成时间。Qwen3-Omni 换成纯因果的解码器，就是为了把这段等待去掉。

### 5. 首包延迟和 RTF：用 Qwen3-Omni 的数字算一遍

**首包延迟**：用户说完到听到第一段声音。它是一串串行步骤的耗时之和：

$$
t_{\text{首包}} = t_{\text{预处理 + 编码}} + t_{\text{Thinker 首 token}} + t_{\text{Talker 首 token}} + t_{\text{MTP 一帧}} + t_{\text{解码一帧}}
$$

**RTF**（real-time factor）：生成一段音频的耗时 ÷ 这段音频的时长。Qwen3-Omni 按「生成一帧（80 ms 音频）要走的每一级各一步」来算：

$$
\text{RTF} = \frac{t_{\text{Thinker 一步}} + t_{\text{Talker 一步}} + t_{\text{MTP 一帧}} + t_{\text{解码一帧}}}{80\ \text{ms}}
$$

论文 Table 2 的数字（作者自测，Qwen3-Omni-30B-A3B，vLLM，MTP 和解码器用了 torch.compile 和 CUDA graph，论文称为理论值，没写 GPU 型号）。表里时间是「音频输入 / 视频输入」：

| | 并发 1 | 并发 4 | 并发 6 |
|---|---|---|---|
| 预处理 + 编码 | 72 / 160 ms | 94 / 180 | 100 / 200 |
| Thinker 首 token | 88 / 160 | 468 / 866 | 673 / 1330 |
| Talker 首 token | 57 / 210 | 145 / 450 | 376 / 734 |
| MTP 一帧 | 14 | 16 | 18 |
| 解码一帧 | 3 | 5 | 5 |
| **首包合计** | **234 / 547** | 728 / 1517 | 1172 / 2284 |
| Thinker 速度 | 75 token/s | 63 | 53 |
| Talker 速度 | 140 token/s | 125 | 110 |
| RTF | 0.47 | 0.56 | 0.66 |

**自己验一下**：

1. 首包，并发 1、音频输入：$72 + 88 + 57 + 14 + 3 = 234$ ms。
2. RTF，并发 1：Thinker 一步 $1000/75 = 13.3$ ms，Talker 一步 $1000/140 = 7.1$ ms，加上 MTP 14 ms、解码 3 ms，共 37.5 ms，$37.5 / 80 = 0.47$。并发 4 算出来 0.56，也对得上；并发 6 算出来 0.64，表里是 0.66，差一点。

**读这张表**：

1. 并发从 1 到 6，RTF 从 0.47 升到 0.66，还在 1 以下，播放不会卡；但首包从 234 ms 涨到 1172 ms，涨得最多的是 Thinker 首 token（88 → 673 ms），也就是 prefill 在排队。
2. 所以 omni serving 的并发上限通常不是 RTF 先到 1，而是首包延迟先超出可接受的范围。
3. MTP 一帧要 14 ms，比 Talker 一步（7 ms）还慢：它每帧要串行走 15 步，虽然模型小，步数多。这也是为什么 MTP 和解码器要用 CUDA graph：每一步的计算很少，kernel 启动开销占大头（见 [CUDA Graph](/framework/cuda-graph)）。

### 6. 对 serving 系统意味着什么

1. **多级、各自 batching**：Thinker、Talker、MTP、解码器的模型大小和步数都不同，各自凑 batch 才能各自跑满。vLLM-Omni 把它们拆成独立的 stage，用 connector 传中间结果（见 [Omni Serving](/inference/omni-serving)）。它报告级间传输的开销（作者自测，Qwen2.5-Omni）：同机共享内存 Thinker → Talker 5.5 ms、Talker → 解码器 0.5 ms；跨机 Mooncake 8.3 ms、3.3 ms。
2. **流式接力**：Talker 不等 Thinker 说完，拿到一段就开始 prefill；Qwen3-Omni 里 Talker prefill 第 $i$ 段时，Thinker 已经在 prefill 第 $i+1$ 段。
3. **小模型的步要快**：MTP 和解码器每一步都很小，用固定大小的 KV cache 和 CUDA graph，消掉 kernel 启动和内存分配的开销。
4. **生成得比播放快没有意义**：RTF = 0.3 和 0.7 用户听起来一样。多出来的速度应该换成更大的并发，SLO 写成「RTF < 1 且首包 < X ms」。

## 面试追问

::: details Q：codec 的码率怎么算？12.5 Hz、8 个码本、每个 2048 个码字是多少？
码率 = 帧率 × 码本数 × $\log_2$(码本大小)。$12.5 \times 8 \times 11 = 1100$ bps，就是 Mimi 的 1.1 kbps。对比 EnCodec 75 Hz、每个码本 1024 个码字（10 bit），一个码本就 750 bps，6 kbps 要 8 个码本。
:::

::: details Q：多码本的 codec，为什么不把所有码本拍平了让 LLM 一个个生成？
步数太多。12.5 Hz、16 个码本拍平是每秒 200 步，每步都要过一遍大模型，延迟和 RTF 都扛不住。常见做法是大模型每帧只走一步，帧内的码本交给一个小模型（Moshi 的 depth transformer、Qwen3-Omni 的 MTP），或者干脆用单码本 tokenizer、再用 flow matching 补细节。
:::

::: details Q：首包延迟由哪几段组成？怎么压？
预处理和编码器、Thinker 出第一个 token、Talker 出第一帧、MTP 补齐这一帧、解码器出第一段波形，串行相加。压的办法：Talker 不等 Thinker 说完、拿到一段就开始（流式接力）；解码器改成纯因果、逐帧出声，不等后面的块；小模块用 CUDA graph；控制并发，避免 Thinker 的 prefill 排队（Qwen3-Omni 并发 1 到 6，Thinker 首 token 从 88 ms 涨到 673 ms）。
:::

::: details Q：RTF 是 0.5，意味着什么？
生成 1 秒音频要 0.5 秒，生成比播放快一倍，播放不会卡。RTF 接近 1 时没有余量，任何抖动都会断音。RTF 远小于 1 也没额外收益，应该加大并发，把多出来的速度用掉。
:::

## 手撕

**RVQ 编码和解码**（码本 `codebooks[k]` 是 `[K, D]`）：

```python
import torch

def rvq_encode(x, codebooks):
    # x: [T, D]，T 帧，每帧一个 D 维向量
    residual, codes = x, []
    for cb in codebooks:
        idx = torch.cdist(residual, cb).argmin(-1)   # 每帧找最近的码字
        codes.append(idx)
        residual = residual - cb[idx]                # 下一级逼近残差
    return torch.stack(codes, -1)                    # [T, N_q]

def rvq_decode(codes, codebooks):
    return sum(cb[codes[:, k]] for k, cb in enumerate(codebooks))
```

**码率、首包和 RTF**：

```python
import math

def bitrate(frame_hz, n_q, codebook_size):
    return frame_hz * n_q * math.log2(codebook_size)

def first_packet_ms(encode, thinker_ttft, talker_ttft, mtp_frame, decode_frame):
    return encode + thinker_ttft + talker_ttft + mtp_frame + decode_frame

def rtf(thinker_tps, talker_tps, mtp_frame_ms, decode_frame_ms, frame_ms=80):
    return (1000 / thinker_tps + 1000 / talker_tps + mtp_frame_ms + decode_frame_ms) / frame_ms

bitrate(12.5, 8, 2048)                  # 1100 bps，Mimi
first_packet_ms(72, 88, 57, 14, 3)      # 234 ms，Qwen3-Omni 并发 1
rtf(75, 140, 14, 3)                     # 0.47
```

常见题：解释 RVQ 并算 codec 码率；多码本怎么减少自回归步数；拆解首包延迟；给定各级速度算 RTF、判断能撑多少并发。

## 参考

- [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
- [Moshi / Mimi](https://arxiv.org/abs/2410.00037)
- [EnCodec](https://arxiv.org/abs/2210.13438)
- [GLM-4-Voice](https://arxiv.org/abs/2412.02612) · [CosyVoice 2](https://arxiv.org/abs/2412.10117) · [Kimi-Audio](https://arxiv.org/abs/2504.18425)
- [vLLM-Omni](https://arxiv.org/abs/2602.02204) · [GitHub](https://github.com/vllm-project/vllm-omni)
