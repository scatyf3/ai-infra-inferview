---
title: Preemption
---

# Preemption

KV cache 满了、新生成的 token 没地方放时，调度器必须挑一些请求让出显存。有两种做法：swap 把这些请求的 KV 换到 CPU 内存，恢复时再换回来；recompute 直接丢掉 KV，恢复时把 prompt 和已生成的 token 重新 prefill 一遍。

- **怎么选**：KV 多、PCIe 快时 swap 划算；序列短时 recompute 更简单，也不慢。vLLM V1 只保留了 recompute。
- **挑谁**：通常抢占最晚到达的请求，避免早到的请求饿死。
- **避免频繁抢占**：admission 时预留余量，或者限制并发数。

**延伸阅读**：[PagedAttention (SOSP '23)](https://arxiv.org/abs/2309.06180)
