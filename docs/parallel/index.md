---
title: 并行与通信
---

# 并行与通信

模型放不下一张卡、或一张卡太慢时，切分维度只有那么几种：切权重（TP）、切层（PP）、切数据（DP）、切专家（EP）、切序列（SP/CP）。每种切法决定了通信原语、通信量和插入位置。

::: tip 面试官真正在考什么
面试官真正在考的是：给定模型和集群拓扑，你能否算出每种并行的通信量并说明为什么这样组合，以及通信如何与计算 overlap。
:::

## 本领域主题

- [TP / PP / DP / EP / SP / CP 总览](./parallelism-overview) — 各自切什么、通信量、通信插在哪一层
- [Megatron Tensor Parallel](./megatron-tp) — column / row parallel 组合为什么能只要两次 all-reduce
- [ZeRO 1 / 2 / 3 与 FSDP](./zero-fsdp) — 通信-显存 tradeoff
- [集合通信原语与 NCCL](./collective-comm) — ring all-reduce 带宽公式；NVLink / PCIe / IB 数量级
- [MoE 与 Expert Parallel](./moe-ep) — 路由、all-to-all、专家负载不均、EP 与 TP 混用
- [计算通信 Overlap](./comm-overlap) — 几种做法：async collective、micro-batch 交错、kernel 级融合
