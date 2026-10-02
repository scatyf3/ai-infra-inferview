---
title: SLA
---

# SLA

LLM 服务的延迟要分阶段定义：TTFT（首 token 延迟）、TPOT / ITL（后续每个 token 的间隔）、E2E（总时长）。SLA 一般写成分位数，比如 TTFT P99 < 2 s、TPOT P99 < 50 ms。

- **goodput**：只统计满足 SLA 的请求的吞吐，比单纯的 tokens/s 更能反映服务能力。
- **取舍**：batch 越大吞吐越高，但 TPOT 会变差；chunked prefill 的块大小在 TTFT 和 ITL 之间取平衡。
- **压测**：用真实的输入输出长度分布和到达率，逐步加压，找到满足 SLA 的最大负载。

**延伸阅读**：[DistServe（goodput 的定义）](https://arxiv.org/abs/2401.09670)
