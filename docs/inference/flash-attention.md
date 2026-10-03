---
title: FlashAttention v1 / v2 / v3
status: todo
tags: [flash-attention, kernel]
difficulty: 4
order: 5
related: []
stack: [k-attn]
---

# FlashAttention v1 / v2 / v3

> tiling + online softmax，为什么省的是 HBM 访问而不是 FLOPs；v2 改了什么切分

## 一句话结论

标准 attention 要把 S×S 的 score 矩阵写到 HBM 再读回来做 softmax，FlashAttention 用 tiling 把 Q、K、V 分块搬进 SRAM，用 online softmax 在块之间增量更新 max 和分母，整个过程不落盘 score 矩阵。FLOPs 没少（甚至多了点重算），省的是 HBM 读写，而 attention 在长序列上恰好是 memory-bound。

## 推导

- **访存账**：标准实现读写 S×S 的 score，HBM 流量 O(S²)；FlashAttention 只读 Q、K、V 各一遍加写 O，流量 O(S·d)，S 远大于 d 时差几十倍。
- **online softmax**：维护当前最大值 m 和分母 l，新块来时把旧的累加结果乘 exp(m_old − m_new) 修正；推导见 [数值稳定 softmax](/handson/stable-softmax)。
- **v2 改了什么**：v1 外层循环遍历 K/V 块、内层遍历 Q 块，中间结果要反复读写；v2 改成外层遍历 Q 块（每个 warp 负责一段 Q 行），K/V 在内层流过，减少了 shared memory 的同步和非 matmul 的 FLOPs，速度提升约 2 倍。
- **v3**：针对 Hopper，用 TMA 异步搬数据、wgmma、warp specialization 让 softmax 和 GEMM 流水重叠，再加 FP8 支持。

## 面试追问

::: details Q：FlashAttention 对 decode 有帮助吗？
帮助有限。decode 时 Q 只有 1 行，score 矩阵是 1×S，本来就不大，瓶颈是读整个 KV cache 的带宽而不是 score 的读写。decode 用的是 FlashDecoding 这类沿 KV 长度切分、多 block 并行再合并的变体，目的是让足够多的 SM 一起读 KV。
:::

## 手撕

常见题：手写两块 K/V 的 online softmax 合并公式；或用 Triton 写一个简化的 FlashAttention 前向。概念铺垫见 [attention 变体](/inference/attention-variants)。

## 参考

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)、[FlashAttention-3](https://arxiv.org/abs/2407.08608)
