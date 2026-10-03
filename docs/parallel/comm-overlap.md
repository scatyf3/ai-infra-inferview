---
title: 计算通信 Overlap
status: todo
tags: [overlap]
difficulty: 3
order: 6
related: []
stack: [d-comm]
---

# 计算通信 Overlap

> 几种做法：async collective、micro-batch 交错、kernel 级融合

## 一句话结论

通信和计算跑在不同硬件单元上（NCCL kernel 占少量 SM + NVLink / NIC），所以理论上可以重叠；做法从粗到细是：把 collective 放到另一个 stream 上异步发起、把 batch 切成 micro-batch 让第 i 块的通信和第 i+1 块的计算交错、把 GEMM 和 all-reduce 融合成一个 kernel 按 tile 粒度边算边发。

## 推导

- **async collective**：`dist.all_reduce(async_op=True)` 返回 handle，先发起通信再去算不依赖它的部分，最后 `wait()`；前提是真有不依赖的计算可做。
- **micro-batch 交错**：PP 的 1F1B 调度就是典型；TP 里把一层的输入切成几块，块 1 的 all-reduce 和块 2 的 GEMM 同时跑，代价是 GEMM 变小效率下降。
- **kernel 级融合**：GEMM 算完一个 tile 就把结果发给其他卡（Megatron 的 TP overlap、Flux、Triton-distributed），通信完全藏在 GEMM 里；实现复杂，依赖 NVLink 的 P2P 写。
- **MoE**：all-to-all 和专家计算交错（DeepEP + 双 micro-batch），是 DeepSeek V3 训练吞吐的关键。

## 面试追问

::: details Q：为什么 overlap 之后 GEMM 本身可能变慢？
NCCL 的通信 kernel 要占用一部分 SM（默认几个到十几个），和 GEMM 抢 SM；而且通信也走 L2 和 HBM，和 GEMM 抢带宽。所以 overlap 的净收益要实测，通信时间远小于计算时，强行 overlap 可能得不偿失。
:::

## 手撕

常见题：画出 TP 一层里 GEMM 和 all-reduce 的依赖图，指出哪些能重叠；通信量的算法见 [集合通信](/parallel/collective-comm)。

## 参考

- [Megatron-LM：Reducing Activation Recomputation（含 TP 通信 overlap）](https://arxiv.org/abs/2205.05198)
- [DeepEP：MoE all-to-all 通信库](https://github.com/deepseek-ai/DeepEP)
