---
title: "GEMM / FP8 GEMM"
---

# GEMM / FP8 GEMM

一个 Linear 层训练时有三个 GEMM：forward $Y = XW$、对输入的梯度 $dX = dY \cdot W^\top$、对权重的梯度 $dW = X^\top dY$，合起来就是每 token $6N$ FLOPs。

- **FP8**：三个 GEMM 的输入 cast 成 fp8，配每张量或每块的 scale，累加仍在 fp32。
- **形状**：TP 切小、micro-batch 小、MoE 专家细，GEMM 都会变窄，跑不满峰值。
