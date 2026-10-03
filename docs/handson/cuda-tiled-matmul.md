---
title: CUDA Tiled Matmul
status: todo
tags: [cuda, gemm, handson]
difficulty: 3
order: 10
related: []
stack: [k-gemm]
leetgpu: [2, 22]
---

# CUDA Tiled Matmul

> shared memory tiling、bank conflict

## 一句话结论

朴素 matmul 每个线程算 C 的一个元素，要从全局内存读 2K 次；tiled 版本让一个 block 负责 C 的一个 BM×BN 块，沿 K 维每次把 A 的 BM×BK 和 B 的 BK×BN 块搬进 shared memory，block 内所有线程复用这两块，全局访存量除以 tile 大小。再进一步是每个线程算多个元素（寄存器 tile）和 Tensor Core。

## 推导

- **复用比**：一次 tile 迭代读 (BM + BN)·BK 个元素，做 BM·BN·BK 次乘加，算术强度 ∝ BM·BN/(BM+BN)；tile 越大越接近 compute-bound，受 shared memory 和寄存器限制。
- **边界处理**：M、N、K 不是 tile 整数倍时加载要判界补零，写回要判界。
- **访存合并**：加载 A、B 到 smem 时让相邻线程读相邻地址；B 按行读天然合并，A 的列访问要注意布局或转置存进 smem。
- **再优化**：每线程算 TM×TN 个元素把 smem 读取量再除以 TM；double buffering 让下一块的加载和当前块的计算重叠；最终形态是 CUTLASS，见 [Tensor Core 与 GEMM tiling](/gpu/tensor-core-gemm)。

## 面试追问

::: details Q：tile 大小为什么不能一直加大？
smem 容量有限（每 SM 100–228 KB），tile 越大每个 block 占的 smem 越多，SM 上能驻留的 block 越少，occupancy 下降藏不住延迟；寄存器同理。而且 BM、BN 变大时 M、N 不整除造成的浪费也变大。实际是在 occupancy 和复用之间调出一个甜点，这正是 autotune 在做的事。
:::

## 手撕

对应 [LeetGPU #2、#22](https://leetgpu.com/challenges)。框架：

```cpp
template <int BM, int BN, int BK>
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK], Bs[BK][BN];
    int row = blockIdx.y * BM + threadIdx.y, col = blockIdx.x * BN + threadIdx.x;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        // 协作加载 A[row, k0:k0+BK] 和 B[k0:k0+BK, col] 到 smem，越界补 0
        __syncthreads();
        for (int kk = 0; kk < BK; ++kk) acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
```
