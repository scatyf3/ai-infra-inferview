---
title: 量化：GPTQ / AWQ / SmoothQuant / FP8
status: todo
tags: [quantization, fp8]
difficulty: 3
order: 8
related: []
stack: [ld-format, ld-dtype, k-gemm]
---

# 量化：GPTQ / AWQ / SmoothQuant / FP8

> W4A16 vs W8A8、per-group vs per-tensor、为什么 decode 场景 weight-only 就够

## 一句话结论

量化分两个问题：权重压到几 bit（省显存、省读权重的带宽）和激活压到几 bit（能用低精度 Tensor Core）。decode 是 memory-bound，只需要 weight-only 的 W4A16 就能提速；prefill 是 compute-bound，要 W8A8 / FP8 这种激活也量化的方案才有收益。GPTQ / AWQ 解决「4 bit 权重怎么压误差小」，SmoothQuant 解决「激活离群值怎么量化」，FP8 是硬件原生支持的新默认。

## 推导

- **W4A16**：权重 int4 + per-group scale（group 128），计算前反量化成 bf16 再乘；GPTQ 用二阶信息逐列补偿误差，AWQ 按激活幅度给重要权重通道放大后再量化。
- **W8A8 / FP8**：权重和激活都 8 bit，直接用 int8 / FP8 Tensor Core，prefill 提速约 2 倍；激活有离群通道时 per-tensor 误差大，SmoothQuant 把激活的难度按通道「平滑」到权重上。
- **粒度**：per-tensor 最省元数据但误差最大，per-channel / per-group 更准；KV cache 也能量化（FP8 KV 常见），见 [显存账](/inference/memory-accounting)。
- **选择**：显存紧张或 batch 小选 W4A16；吞吐优先、H100 上选 FP8（W8A8）；精度敏感的任务先跑 eval 再上线。

## 面试追问

::: details Q：为什么 W4A16 在大 batch 下反而可能比 bf16 慢？
batch 大了之后 GEMM 变成 compute-bound，瓶颈不再是读权重；而 W4A16 的 kernel 要先反量化再乘，多了额外的计算和 shared memory 操作，Tensor Core 利用率低于直接的 bf16 GEMM。所以 W4A16 的收益随 batch 增大而消失，大 batch 场景要换成 W8A8。
:::

## 手撕

常见题：手写 per-group 对称量化和反量化（求 scale、round、clamp）；算一下 70B 模型 W4A16 后的权重大小。

## 参考

- [GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978)、[SmoothQuant](https://arxiv.org/abs/2211.10438)
- [vLLM 文档：Quantization](https://docs.vllm.ai/en/latest/features/quantization/)
