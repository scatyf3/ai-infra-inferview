---
title: Omni Serving：多模态输入输出的推理系统
status: draft
tags: [omni, multimodal, serving, tts]
difficulty: 4
order: 10
related: [/inference/batching-scheduling, /inference/kv-cache-paged-attention, /inference/metrics-benchmark]
stack: []
---

# Omni Serving：多模态输入输出的推理系统

> Preliminary：模型长什么样、每一级是什么负载、和纯文本 serving 差在哪

## 一句话结论

Omni 模型（Qwen-Omni、MiniCPM-o 这类）不是「一个 LLM」，而是**一条异构流水线**：模态编码器（compute-bound，非自回归）→ Thinker LLM（自回归出文本）→ Talker（自回归出语音 codec token）→ codec 解码器 / vocoder（非自回归出波形）。纯文本 serving 只需要调度一个自回归引擎；omni serving 要给**每一级单独 batching、单独分配 GPU**，再用**流式**把级与级接起来，指标也从 TTFT / TPOT 变成「首包音频延迟」和「生成速度是否追得上播放速度」。

## Preliminary

### 1. 模型解剖：Thinker-Talker

以 [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) / [Qwen3-Omni](https://arxiv.org/abs/2509.17765) 为代表：

```
 图像/视频 ──ViT──┐
                  ├─ projector ─> Thinker (LLM) ──文本 token──> 用户
 音频 ──音频编码器─┘                  │
                                    隐状态
                                      ↓
                                  Talker (小 LLM) ──codec token──> codec 解码器 ──波形──> 用户
```

| 级 | 干什么 | 计算形态 | 类比纯文本 serving |
|---|---|---|---|
| 模态编码器 | 图像 / 视频帧 / 音频 → embedding | 一次性前向，**compute-bound** | 像一次额外的 prefill，但不产生 KV |
| Thinker | 理解 + 生成文本 | 自回归，prefill + decode | 就是普通 LLM |
| Talker | 以 Thinker 的隐状态为条件，生成语音 codec token | 自回归，模型小、步数多 | 又一个 LLM，但输入来自上游流 |
| codec 解码器 | codec token → 波形 | 非自回归（ConvNet / DiT / flow matching） | 没有对应物，更像图像生成 |

Talker 吃的是 Thinker 的**隐状态**而不只是文本 token，所以两级之间传的是张量，不是字符串。这决定了它们之间要有一条高效的 GPU 间数据通道。

### 2. 多模态输入会变成多少 token

serving 的显存和延迟账都从 token 数算起：

- **图像**：动态分辨率的 ViT 大致是「每 $28 \times 28$ 像素一个 token」（[Qwen2-VL](https://arxiv.org/abs/2409.12191) 系：14×14 patch，再 2×2 合并）。一张 1920×1080 的图约 $1920 \times 1080 / 784 \approx 2.6\text{k}$ token。
- **视频**：再乘以采样帧数（通常还会在时间维 2 帧合 1）。几十秒的视频轻松上万 token。
- **音频**：Whisper 系编码器下采样后约 **25 token/s**（1 token ≈ 40 ms，见 [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)），1 分钟音频 ≈ 1.5k token。

结论：**多模态请求的 prefill 很长**，编码器本身也有不小的计算量。纯文本里「prompt 几百 token」的直觉在这里不成立。

### 3. 编码器为什么是新瓶颈

编码器的前向和 prefill 一样是 compute-bound，而且：

- **不产生 KV cache**，输出的 embedding 只在 prefill 时用一次。
- 和 LLM 放在同一组卡上时，一个大图请求的编码会**卡住同 batch 里所有请求的 decode**（类似长 prefill 打断 decode，见 [Continuous Batching 与 PD 分离](/inference/batching-scheduling)）。
- 同一张图被多次引用（多轮对话里反复带着同一张图）时，重复编码纯属浪费。

对应的三种优化：

1. **Encoder cache**：按多模态输入内容的 hash 缓存编码结果。vLLM 里还会把这个 hash 作为 extra key 拌进（[vLLM prefix caching 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching.html)） [prefix caching](/inference/kv-cache-paged-attention) 的 block hash，否则「同样的占位 token、不同的图」会错误命中。
2. **Encoder 的 chunk / 预算**：调度器每步限制编码的 token 预算，和 chunked prefill 一个思路。
3. **EPD 分离**：把 Encode、Prefill、Decode 拆到三组卡上，在 PD 分离之上再多拆一级。[EPD 论文](https://arxiv.org/abs/2501.05460)报告 TTFT 最多降 71%（作者自测数据）。

### 4. 输出侧：流式语音

语音输出的用户体验由两个量决定：

- **首包延迟**（first-packet latency / TTFA, time to first audio）：从用户说完到听到第一段声音。它是整条链路的 TTFT 之和：编码 + Thinker 首 token + Talker 首 codec token + codec 解码第一帧。
- **实时率 RTF**（real-time factor）= 生成耗时 / 音频时长。**RTF < 1 才不卡顿**：每秒至少要生成 1 秒的音频，否则播放会追上生成、出现断续。

所以 omni serving 的 SLO 不是「TPOT 越小越好」，而是**「保证 RTF < 1 的前提下塞进尽量多的并发」**。生成比播放快得多也没意义，多出来的速度应该换成 batch。

让首包变短的手段基本都是**流式接力**：

- Thinker 每出一个 chunk 就把隐状态交给 Talker，Talker 边收边 prefill，不等 Thinker 说完（[Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)、[Qwen3-Omni](https://arxiv.org/abs/2509.17765) 的 chunked prefill）。
- Talker 每出几个 codec 帧，codec 解码器就开始出波形。Qwen2.5-Omni 用滑动窗口 DiT 限制感受野；[Qwen3-Omni](https://arxiv.org/abs/2509.17765) 进一步把 codec 解码器换成**因果 ConvNet**，从第一个 codec 帧就能流式出声，报告的理论冷启动首包延迟是 234 ms（音频输入）。
- 多码本 codec 下，Talker 每步只自回归预测第一层码本，剩余层用一个小的 MTP 模块并行补齐，减少自回归步数（[Qwen3-Omni](https://arxiv.org/abs/2509.17765)）。

### 5. 系统层：从「一个引擎」到「stage 图」

把上面几条合起来，omni serving 系统要做的事：

| 问题 | 做法 |
|---|---|
| 各级计算形态不同 | 每级是独立的 stage，自回归 stage 用 LLM 引擎，扩散 / 解码 stage 用另一套引擎 |
| 各级 batch 规律不同 | **per-stage batching**：Thinker、Talker、解码器各自凑 batch |
| 各级算力需求不同 | 每级单独分配 GPU 数，按负载调比例 |
| 级间传隐状态 | 级间 connector（同卡共享显存、跨卡 NCCL / RDMA） |
| 流式 | 级与级 overlap，下游不等上游整段完成 |

**vLLM-Omni**（[论文](https://arxiv.org/abs/2602.02204)）就是按这个思路做的：把 any-to-any 模型拆成 stage 图，节点是 AR 或 DiT stage，边是用户定义的数据变换，每个 stage 独立 batching、独立分配 GPU，用统一的 connector 传中间数据。

### 6. 还没被很好解决的问题

做研究可以从这里找切入点：

- **全双工与打断**：用户中途插话时，正在生成的 Thinker / Talker 请求要中止，已有的 KV 要保留还是回滚？VAD（语音活动检测）放在哪一级？
- **流式输入**：用户还在说话时就边听边 prefill（音频按 chunk 进），而不是等说完再整体编码。
- **跨级调度**：Thinker 的 batch 大小会影响 Talker 的输入速率，各级的 SLO 要联合优化，不能各调各的。
- **各级 GPU 配比**：不同请求的模态组合差别很大（纯文本、看图说话、语音对话），固定配比容易让某一级成为瓶颈。

## 面试追问

::: details Q：为什么不能把 Thinker 和 Talker 当成一个大模型串行跑？
两者都是自回归，但节奏不同：Thinker 出一个文本 token 大约对应 Talker 好几个 codec 帧。串行跑意味着 Talker 要等 Thinker 整段说完才开始，首包延迟等于整段文本的生成时间，几秒起步。拆成两级流式接力，Talker 拿到第一个 chunk 就能开始，首包降到几百毫秒。另外两者的模型大小、最优 batch 都不同，分开才能各自调。
:::

::: details Q：语音输出的服务，并发上限由什么决定？
由 RTF < 1 这个约束决定。并发越高，每个请求分到的生成速度越慢，RTF 越接近 1。上限就是「最慢那一级刚好让 RTF = 1」时的并发数。这和文本 serving 在 TPOT SLO 下找最大 batch 是同一个问题，只是约束变成了播放速度。
:::

::: details Q：多模态请求的 prefix caching 有什么坑？
图像在 prompt 里通常展开成一串相同的占位 token，只看 token id 的话两张不同的图会算出相同的 hash，错误命中。所以 block hash 必须把多模态输入的内容 hash 拌进去。另一个坑是编码结果和 KV 是两层缓存：KV 命中了就不需要编码结果；KV 被淘汰了但 encoder cache 还在，可以跳过编码直接 prefill。
:::

## 参考

- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
- [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)
- [Qwen2-VL](https://arxiv.org/abs/2409.12191)
- [Efficiently Serving Large Multimodal Models Using EPD Disaggregation](https://arxiv.org/abs/2501.05460)
- [vLLM-Omni 论文](https://arxiv.org/abs/2602.02204) · [GitHub](https://github.com/vllm-project/vllm-omni) · [文档](https://docs.vllm.ai/projects/vllm-omni)
