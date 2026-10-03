---
title: Tensor Core 与 GEMM Tiling
status: todo
tags: [tensor-core, gemm]
difficulty: 3
order: 2
related: []
stack: [k-gemm, hw-gpu]
---

# Tensor Core 与 GEMM Tiling

> 为什么 GEMV 打不满

## 一句话结论

Tensor Core 一条指令算一个小矩阵乘（如 16×8×16），峰值算力比 CUDA core 高一个数量级，但前提是有足够的数据复用：GEMM 靠 tiling 把 A、B 的块留在 shared memory / 寄存器里反复用；decode 的 GEMV（M = batch 很小）每个权重只用一次，没有复用，带宽先打满，Tensor Core 大部分时间在等数据。

## 推导

- **算术强度**：tile 为 BM×BN×BK 时，一次 tile 计算做 2·BM·BN·BK FLOPs，读 (BM+BN)·BK 个元素；BM、BN 越大复用越多。GEMV 相当于 BM = batch，M 很小时强度退化到约 batch 次 FLOP / 字节。
- **三级 tiling**：block tile 放 shared memory，warp tile 放寄存器，mma 指令级 tile 对应 Tensor Core 形状；Hopper 上用 TMA 异步搬 tile、wgmma 直接从 shared memory 读。
- **为什么 GEMV 打不满**：H100 的 ridge point 约 300 FLOP/B（bf16），decode 的强度约等于 batch size，batch 不到几百就 memory-bound；详见 [prefill / decode roofline](/inference/prefill-decode-roofline)。
- **decode 的对策**：加大 batch（continuous batching）、减少权重字节（W4A16 量化）、split-K 让更多 SM 参与读权重。

## 面试追问

::: details Q：为什么量化到 W4A16 对 decode 有效，对 prefill 几乎没用？
decode 是 memory-bound，时间由读权重的字节数决定，权重从 16 bit 变 4 bit 读取量减四分之三，直接提速。prefill 是 compute-bound，时间由 FLOPs 决定，W4A16 在计算前还要反量化回 bf16，FLOPs 不减反增。要加速 prefill 得用 W8A8 / FP8 这种 Tensor Core 原生支持的低精度计算。
:::

## 手撕

常见题：手写 shared memory tiled matmul 并解释每级 tile 的复用；见 [CUDA Tiled Matmul](/handson/cuda-tiled-matmul)。

## 参考

- [CUTLASS：Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
