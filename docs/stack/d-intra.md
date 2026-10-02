---
title: TP / PP / EP / DP
---

# TP / PP / EP / DP

一张卡放不下或者跑不快时，按不同维度切模型。TP 切每一层里的矩阵，PP 按层切成几段，EP 把 MoE 的 expert 分到不同卡上，DP 是把整份模型复制多份。推理里最常见的组合是机内 TP、MoE 层用 EP、实例之间 DP。

- **TP**：Megatron 式的 column + row parallel，每层两次 all-reduce，通信频繁，只适合 NVLink 域内。
- **PP**：只在段与段的边界通信，但有 bubble，推理里主要用来跨机放超大模型。
- **EP**：dispatch 和 combine 各一次 all-to-all，专家负载不均会拖慢整步。
- **attention DP**：MLA 这类 KV head 很少的模型，attention 部分用 DP、MoE 部分用 EP，避免 KV 在 TP 下被重复存储。

**延伸阅读**：[Megatron-LM](https://arxiv.org/abs/1909.08053)
