---
title: Attention Kernel
---

# Attention Kernel

朴素 attention 会把 N × N 的 score 矩阵写回 HBM，显存和带宽开销都随序列长度平方增长。FlashAttention 用分块加 online softmax，让 score 始终留在片上，HBM 读写大幅减少，而 FLOPs 并没有变少。

- **prefill vs decode**：prefill 是 compute-bound 的大块计算；decode 每步只有 1 个 query，要在 KV 长度维度上切分并行（split-KV / Flash-Decoding）才能占满 GPU。
- **paged**：decode kernel 要能按 block table 从不连续的物理块里读 KV。
- **实现**：FlashAttention 2 / 3、FlashInfer，以及各家针对 MLA 的专用 kernel。

**延伸阅读**：[FlashAttention](https://arxiv.org/abs/2205.14135) · [FlashAttention-2](https://arxiv.org/abs/2307.08691) · [FlashInfer](https://arxiv.org/abs/2501.01005)
