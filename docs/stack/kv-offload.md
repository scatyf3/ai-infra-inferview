---
title: KV Offload
---

# KV Offload

显存装不下的 KV 往更便宜的层级放：CPU 内存、本地 SSD、远端存储。被抢占的请求、长时间不活跃的多轮会话、可复用的前缀都可以先下放，用到时再搬回来，用 PCIe 带宽换显存容量和重新计算的算力。

- **划不划算**：搬回来的时间要比重新 prefill 短。长前缀通常划算，短前缀不如直接重算。
- **分层 KV**：GPU → CPU → 磁盘 / 对象存储，按热度放置，和 prefix caching 结合实现跨请求复用。
- **跨实例**：PD 分离里 KV 从 prefill 实例传到 decode 实例，用的是同一套传输机制。

**延伸阅读**：[LMCache](https://github.com/LMCache/LMCache) · [Mooncake](https://arxiv.org/abs/2407.00079)
