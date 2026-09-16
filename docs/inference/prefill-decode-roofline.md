---
title: Prefill vs Decode 与 Roofline
status: draft
tags: [roofline, arithmetic-intensity]
difficulty: 3
order: 1
related: [/inference/memory-accounting, /inference/batching-scheduling, /gpu/tensor-core-gemm]
---

# Prefill vs Decode 与 Roofline

> compute-bound vs memory-bound、arithmetic intensity、roofline 模型

## 一句话结论

**prefill 和 decode 是两个完全不同的负载，跑在同一块卡上。** prefill 的 arithmetic intensity 约等于序列长度（上千），落在 roofline 的右侧，compute-bound；decode 的 AI 约等于 batch size（几十），落在左侧，memory-bound。所以 prefill 的优化是「把 tensor core 喂饱」，decode 的优化是「少读字节」，两者的手段几乎没有交集。

## 推导

### 一条链

$$
\text{shape} \to \text{bytes} + \text{FLOPs} \to \text{AI} = \frac{\text{FLOPs}}{\text{Bytes}} \to \text{AI} \lessgtr \text{ridge} \to \text{优化手段}
$$

roofline 的上界是

$$
\text{attainable FLOP/s} = \min(\text{peak}, \ \text{BW} \times \text{AI})
$$

拐点（ridge point）在 $\text{AI} = \text{peak} / \text{BW}$。H100 SXM：$989\ \text{TFLOP/s} \div 3.35\ \text{TB/s} \approx 295\ \text{FLOP/B}$。**这个数要记住**：AI 低于 295 就是浪费算力，高于 295 就是浪费带宽。

### prefill

一次处理 $S$ 个 token。FLOPs 主体是所有 GEMM，每个参数对每个 token 做一次乘加：

$$
\text{FLOPs}_{\text{prefill}} \approx 2 P B S + \underbrace{4 L d S^2 B}_{\text{attention}}
$$

访存主体是把权重读一遍：$\approx P b_w$。于是

$$
\text{AI}_{\text{prefill}} \approx \frac{2 P B S}{P b_w} = \frac{2 B S}{b_w} \sim S
$$

$S = 2048$ 时 AI 是几千，远在 ridge 右边 → **compute-bound**。结论：prefill 的时间由算力决定，TTFT $\approx$ FLOPs / (peak × MFU)。优化方向是提高 MFU：大 tile 的 GEMM、fp8 tensor core、chunked prefill 让每个 chunk 都足够大填满 SM。

注意 attention 项是 $S^2$ 的：$S$ 很长时它会反超线性项。70B、8k 时 attention 占约 15%；32k 时占约 40%。这就是长上下文 prefill 特别贵、以及 FlashAttention 对 prefill 至关重要的原因。

### decode

一次只处理 1 个 token。FLOPs：

$$
\text{FLOPs}_{\text{decode}} \approx 2 P B + 4 L d S_{ctx} B
$$

访存：权重整读一遍 **加上整个 KV cache 读一遍**：

$$
\text{Bytes}_{\text{decode}} \approx P b_w + \text{KV/token} \cdot B \cdot S_{ctx}
$$

于是

$$
\text{AI}_{\text{decode}} \approx \frac{2 P B}{P b_w + \text{KV}} \xrightarrow{\text{KV} \ll W} \frac{2B}{b_w} = B \ \ (\text{bf16})
$$

**decode 的 AI 就是 batch size**。batch 1 时 AI = 1，比 ridge 低 295 倍，意味着 tensor core 只有 0.3% 在干活，整块 H100 在等 HBM。这不是实现问题，是这个计算本身的性质：每个权重元素读进来只用一次，是 GEMV 不是 GEMM。

### 所以优化只能是

| | prefill | decode |
|---|---|---|
| 瓶颈 | 算力 | 带宽 |
| AI | ≈ S（上千） | ≈ B（几十） |
| 指标 | TTFT | TPOT / ITL |
| 有效手段 | fp8、大 tile GEMM、chunked prefill、FlashAttention | weight-only 量化、GQA/MLA、continuous batching 堆 batch、speculative decoding |
| 无效手段 | 堆 batch（已经饱和） | 单纯换更强算力的卡 |

speculative decoding 特别值得注意：它的本质是**把 memory-bound 的 decode 变成一次验证 $k$ 个 token 的小 prefill**，让 AI 从 $B$ 变成 $kB$，用本来闲置的算力换延迟。这也解释了它为什么在小 batch 下收益大、大 batch 下收益消失（batch 大了以后 AI 已经接近 ridge，没有空闲算力可换）。

### 两者混在一起的麻烦

prefill 和 decode 抢同一块卡。一个长 prompt 的 prefill 会把所有正在 decode 的请求卡住几百毫秒，表现为 ITL 尖刺。解法是 **chunked prefill**（把 prefill 切成小块，和 decode 拼在同一个 batch 里）或 **PD 分离**（prefill 和 decode 跑在不同的卡上，KV 通过网络传）。

## 交互

把 batch 从 1 调到 256，看 decode 的点沿着斜坡向右爬；调到 ridge 附近时，继续加 batch 就不再提升单卡吞吐了。

<MemoryCalculator />

## 面试追问

::: details Q：batch 加到多大，decode 就 compute-bound 了？
理论上 batch ≈ ridge point ≈ 295（bf16）。但实际到不了：batch 涨的同时 KV cache 也在涨，分母的 KV 项开始主导，AI 会饱和在一个低于 batch 的值。KV 越大（长 context、MHA）饱和得越早。这正是 GQA/MLA 除了省显存以外的第二个价值：让 AI 能随 batch 涨得更久。
:::

::: details Q：MFU 和 MBU 分别是什么，各自该看哪个？
MFU = 实际 FLOP/s ÷ 峰值 FLOP/s，衡量算力利用率，prefill 看它。MBU = 实际访存带宽 ÷ 峰值带宽，decode 看它。一个良好实现的 decode 应该有 60–80% 的 MBU；如果 MBU 很低而 MFU 也很低，说明瓶颈既不是带宽也不是算力，而是 kernel launch 开销或 CPU 调度，该上 CUDA graph。
:::

::: details Q：为什么 roofline 上 decode 的点画在斜坡上而不是峰值线上？
因为斜坡代表带宽上界。AI < ridge 时，即使算力无限，也只能达到 BW × AI 的有效算力。decode 的点落在斜坡上意味着它最好的情况就是把带宽打满，换更强算力的卡（比如同带宽但算力翻倍）对它一点帮助没有。H200 相对 H100 就是靠带宽从 3.35 涨到 4.8 TB/s 提升 decode 的，算力完全一样。
:::

::: details Q：chunked prefill 的 chunk 该设多大？
太小则每个 chunk 的 GEMM 打不满 tensor core，prefill 效率下降；太大则又会阻塞 decode，ITL 变差。实践中取 512–2048 token，让 chunk 的 AI 仍远高于 ridge 即可。vLLM 的 `max_num_batched_tokens` 就是这个旋钮，它同时决定了一个 batch 里 prefill token 加 decode token 的总预算。
:::

## 参考

- [Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785)
- [LLM Inference Performance Engineering — Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [SARATHI: chunked prefill](https://arxiv.org/abs/2308.16369)
