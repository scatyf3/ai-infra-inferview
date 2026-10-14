---
title: "计算通信 overlap"
---

# 计算通信 overlap

通信只要能和不相关的计算同时跑就不算进 step 时间。DP 的梯度同步和 backward 重叠，FSDP 预取下一层参数，TP 拆块流水，MoE 用双 micro-batch 或 DualPipe 交错。

- **前提**：两件事互不依赖，而且通信不抢太多 SM。

拆块流水的公式，以及 DP bucket、FSDP 预取、TP 的 async TP / Flux、MoE 的 DeepEP 各自怎么藏通信，见 [计算通信 Overlap](/parallel/comm-overlap)。
