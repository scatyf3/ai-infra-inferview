---
title: Paged KV Cache
---

# Paged KV Cache

连续预分配 KV cache 必须按最大长度预留，实际用不满就造成大量浪费。PagedAttention 借鉴操作系统分页，把 KV 切成固定大小的 block（vLLM 默认 16 个 token），每个请求用一张 block table 记录逻辑块到物理块的映射，用多少分配多少。

- **碎片**：只剩每个请求最后一个 block 里的少量浪费，显存利用率接近满，能放下更大的 batch。
- **共享**：多条序列可以指向同一个物理 block（beam search、并行采样），写时再复制。
- **代价**：attention kernel 要按 block table 间接寻址。

**延伸阅读**：[PagedAttention (SOSP '23)](https://arxiv.org/abs/2309.06180)
