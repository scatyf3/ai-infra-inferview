---
title: Triton 编程模型
status: todo
tags: [triton]
difficulty: 3
order: 3
related: []
stack: [k-lang]
---

# Triton 编程模型

> autotune；什么时候写 Triton、什么时候直接调 cuBLAS / CUTLASS

## 一句话结论

Triton 让你以「block」而不是「thread」为单位写 kernel：一个 program 处理一个 tile，`tl.load` / `tl.store` 带 mask 做边界，编译器负责线程映射、shared memory 和访存合并。用 Triton 写 memory-bound 的融合算子（softmax、RMSNorm、attention 变体）；标准 GEMM 直接调 cuBLAS / CUTLASS，Triton 的 matmul 一般只能到它们的 80–90%。

## 推导

<TritonProgramViz mode="matmul" />

- **编程模型**：`tl.program_id` 决定本 program 负责哪个 tile；`tl.arange` + 偏移生成一块指针，`mask` 处理尾部；block 内的 reduce 用 `tl.sum` / `tl.max`。
- **autotune**：`@triton.autotune` 对 BLOCK 大小、`num_warps`、`num_stages` 做网格搜索，按输入 shape 作为 key 缓存结果；第一次调用会慢。
- **什么时候写 Triton**：需要融合多个算子、形状特殊（GQA、paged KV）、cuBLAS 没有对应 kernel 时；什么时候不写：标准 dense GEMM、已有高度优化库（FlashAttention 官方实现）时。
- **和 CUDA 的差别**：没有 `__syncthreads`、不直接管 shared memory，损失一些极限性能，换来几十行代码写完一个 fused kernel。

## 面试追问

::: details Q：Triton 的 BLOCK_SIZE 为什么必须是 2 的幂？
`tl.arange` 和 block 内的 reduce、broadcast 都按 2 的幂做树形展开和线程映射，编译器依赖这个假设生成代码。实际长度不是 2 的幂时，取 `next_power_of_2` 再用 mask 屏蔽越界元素。
:::

## 手撕

常见题：用 Triton 写 row softmax 或 RMSNorm；原语速查见 [Triton 原语](/handson/triton_primitives)，入门从 [vector add](/handson/vector-add) 开始。

## 参考

- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Triton: an intermediate language and compiler for tiled neural network computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
