---
title: "第 2 层：tensor → kernel"
---

# 第 2 层：tensor → kernel

单个算子在 GPU 上怎么跑。LLM 推理的时间几乎都花在少数几类 kernel 上：线性层的 GEMM、attention，以及一批小的逐元素算子。这一层的目标是让每个 kernel 逼近它的 roofline 上限：compute-bound 的喂饱 Tensor Core，memory-bound 的跑满 HBM 带宽。

- **GEMM**：占了大部分 FLOPs，decode 时退化成带宽受限。
- **attention**：开销随序列长度增长，prefill 和 decode 的优化方向完全不同。
- **小算子**：单个不贵，加起来很可观，靠融合解决。
