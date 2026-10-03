---
title: 集合通信原语与 NCCL
status: todo
tags: [nccl, all-reduce, nvlink]
difficulty: 3
order: 4
related: []
stack: [d-comm, hw-link]
---

# 集合通信原语与 NCCL

> ring all-reduce 带宽公式；NVLink / PCIe / IB 数量级

## 一句话结论

集合通信就五个常用原语：all-reduce（TP 每层两次）、all-gather / reduce-scatter（ZeRO、序列并行）、all-to-all（MoE 的 EP）、broadcast / P2P（PP）。ring all-reduce 每张卡的通信量是 2(N−1)/N × 数据量，几乎与卡数无关，所以瓶颈是互联带宽：NVLink 几百 GB/s、PCIe 几十 GB/s、IB 每端口几十 GB/s，差一个数量级就决定了 TP 只能在节点内做。

## 推导

- **ring all-reduce**：reduce-scatter + all-gather 两个阶段，各 N−1 步，每步每卡发 / 收 1/N 的数据；总量 2(N−1)/N·D，带宽最优但延迟随 N 线性增长，小消息用 tree 算法。
- **带宽数量级**：H100 NVLink 单卡 900 GB/s（双向）、PCIe 5.0 x16 约 64 GB/s、IB NDR 400 Gb/s ≈ 50 GB/s；节点内 NVSwitch 全互联，跨节点走 IB / RoCE。
- **NCCL**：自动探测拓扑选 ring / tree，通信 kernel 跑在 GPU 上占少量 SM，通过 stream 和计算重叠；`NCCL_DEBUG=INFO` 看它选了什么算法。
- **算一算**：70B 模型 TP=8，每层 hidden 8192、batch×seq = 4096 token、bf16，一次 all-reduce 数据量 64 MiB，NVLink 上约 0.2 ms，每层两次、80 层就是几十毫秒，这是 TP 的基本代价。

## 面试追问

::: details Q：为什么 TP 一般不跨节点，PP 可以？
TP 每层要两次 all-reduce，通信量大且在关键路径上，跨节点的 IB 带宽比 NVLink 低一个数量级，等通信的时间会超过计算。PP 只在 stage 边界传一次激活（P2P），数据量小得多，而且可以和计算流水重叠，所以能跨节点。
:::

## 手撕

常见题：推导 ring all-reduce 的通信量；给定模型和互联带宽估算 TP 一层的通信时间。并行方式的全景见 [并行总览](/parallel/parallelism-overview)。

## 参考

- [NCCL 文档：Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [Bringing HPC Techniques to Deep Learning（ring all-reduce 原文）](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)
