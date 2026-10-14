---
title: "GEMM / FP8 GEMM"
---

# GEMM / FP8 GEMM

一个 Linear 层训练时有三个 GEMM：forward $Y = XW$、对输入的梯度 $dX = dY \cdot W^\top$、对权重的梯度 $dW = X^\top dY$，合起来就是每 token $6N$ FLOPs（$N$ 为参数量：每个权重每 token 在每个 GEMM 里做一次乘加，记 2 FLOPs；forward $2N$，两个 backward GEMM 各 $2N$；Kaplan et al. 2020）。

训练里 GEMM 吞吐主要看两件事：
- **精度（FP8）**：三个 GEMM 的输入 cast 成 fp8，配每张量或每块的 scale，累加仍在 fp32；Tensor Core 峰值约为 BF16 的 2 倍。
- **形状**：TP 切小、micro-batch 小、MoE 专家细，GEMM 都会变窄，算术强度下降、tile 填不满 SM，跑不满峰值；对策是 grouped GEMM、维度对齐、控制 TP 度。
