---
title: 通信
---

# 通信

并行切得越细，通信就越多。推理里关键的集合通信有三种：TP 的 all-reduce、EP 的 all-to-all、PP 的点对点 send / recv。decode 时每次通信的数据量很小，瓶颈往往是延迟而不是带宽。

- **ring all-reduce**：每张卡收发约 2(N−1)/N 倍的数据，带宽最优，但步数随卡数线性增长。
- **小消息优化**：decode 的 all-reduce 只有几十 KB，常用 one-shot / two-shot 的自定义 kernel 或 NVLS 来降低延迟。
- **overlap**：让通信和不相关的计算并行，或者把通信直接融合进 GEMM kernel。

**延伸阅读**：[NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
