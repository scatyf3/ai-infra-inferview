---
title: "计算通信 overlap"
---

# 计算通信 overlap

通信只要能和不相关的计算同时跑就不算进 step 时间。DP 的梯度同步和 backward 重叠，FSDP 预取下一层参数，TP 拆块流水，MoE 用双 micro-batch 或 DualPipe 交错。

- **前提**：两件事互不依赖，而且通信不抢太多 SM。
