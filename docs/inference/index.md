---
title: 推理系统核心
---

# 推理系统核心

这里的语言是 tensor 怎么算：shape → 访存量 (bytes) 和计算量 (FLOPs) → arithmetic intensity → roofline 落在哪一侧 → 所以优化手段只能是 X。所有推理优化都能顺着这条链解释。

::: tip 面试官真正在考什么
面试官真正在考的是：你能否用一套一致的账本（显存、带宽、FLOPs）解释每个优化为什么有效、在什么条件下失效，而不是背名词。
:::

## 本领域主题

- [Prefill vs Decode 与 Roofline](./prefill-decode-roofline) — compute-bound vs memory-bound、arithmetic intensity、roofline 模型
- [显存账：权重 / KV cache / 激活](./memory-accounting) — 能口算「70B bf16 + 8k context + batch 32 要多少卡」
- [KV Cache 与 PagedAttention](./kv-cache-paged-attention) — 布局、PagedAttention、prefix caching / RadixAttention、KV 量化
- [Attention 变体：MHA / MQA / GQA / MLA](./attention-variants) — KV 大小与 decode 带宽影响
- [FlashAttention v1 / v2 / v3](./flash-attention) — tiling + online softmax，为什么省的是 HBM 访问而不是 FLOPs；v2 改了什么切分
- [Continuous Batching、Chunked Prefill 与 PD 分离](./batching-scheduling) — scheduler 抢占（swap vs recompute）
- [Speculative Decoding](./speculative-decoding) — draft-target、acceptance rate、EAGLE / Medusa / MTP 的区别
- [量化：GPTQ / AWQ / SmoothQuant / FP8](./quantization) — W4A16 vs W8A8、per-group vs per-tensor、为什么 decode 场景 weight-only 就够
- [指标与 Benchmark：TTFT / TPOT / ITL / Goodput](./metrics-benchmark) — 怎么做 benchmark，SLA 下怎么调 batch
