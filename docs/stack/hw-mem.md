---
title: 显存层次
---

# 显存层次

从快到慢：寄存器 → shared memory / L1（H100 每个 SM 最多 228 KB shared）→ L2（50 MB）→ HBM（80 GB，H100 SXM 约 3.35 TB/s）。越往下越大也越慢。decode 每生成一个 token 都要把全部权重和 KV 从 HBM 读一遍，所以 decode 的上限通常由 HBM 带宽决定，而不是算力。

- **ridge point**：H100 上 989 TFLOPS ÷ 3.35 TB/s ≈ 295 FLOP/byte，算术强度低于它就是 memory-bound。
- **复用**：tiling 的本质就是把数据搬到更快的层级，然后多用几次。
- **容量**：权重加 KV cache 装不装得下，决定了要几张卡。

## 从架构图上认出每一层

下面两张图来自 NVIDIA 的 [Hopper 架构介绍](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)。先看整颗芯片：

![GH100 整芯片框图：中间是两块 L2，四周是 GPC/SM，左右两侧是 HBM3 和内存控制器，底部是 NVLink](/figures/gh100-full-chip.png)

*图 1：完整的 GH100 芯片（NVIDIA，Figure 3）。*

从外往里读：

1. **左右两侧的 HBM3**：显存颗粒堆叠在芯片旁边，经 Memory Controller 接进来。这就是 `cudaMalloc` 拿到的「显存」，所有 SM 共享。
2. **中间的蓝色 L2 Cache**：所有 SM 读写 HBM 都要经过它。它分成左右两个分区，每个分区优先缓存直接连到它的那些 GPC 的访问（NVIDIA 原文：partitioned crossbar）。
3. **上下八块 GPC**：每块里有很多 TPC，每个 TPC 两个 SM。寄存器、L1 和 shared memory 都在 SM 内部。
4. **底部 NVLink、顶部 PCIe**：通往别的 GPU 和 CPU 内存，比 HBM 再慢一级，见 [互联](/stack/hw-link)。

注意这张图是**完整的** GH100：144 个 SM、6 组 HBM3、60 MB L2。卖的 H100 SXM5 只启用了一部分：**132 个 SM、5 组 HBM3（80 GB）、50 MB L2**（同一篇文章的规格列表）。下文的数字都按 H100 SXM5 算。

再放大到一个 SM：

![H100 SM 框图：四个分区，每个分区有 16384×32-bit 寄存器文件和一个 Tensor Core；底部是 Tensor Memory Accelerator 和 256 KB L1 Data Cache / Shared Memory](/figures/h100-sm.png)

*图 2：H100 的一个 SM（NVIDIA，Figure 4）。*

1. **四个分区**：每个分区有自己的 warp 调度器、一块 **Register File（16,384 × 32-bit = 64 KB）** 和一个 Tensor Core。四块合起来每个 SM 256 KB 寄存器，132 个 SM 共约 33 MB，比 L2 还大。
2. **256 KB L1 Data Cache / Shared Memory**：同一块 SRAM，一部分划给 shared memory（最多 228 KB，由程序员管理），剩下的当 L1（由硬件管理）。
3. **Tensor Memory Accelerator（TMA）**：Hopper 新加的搬运单元，把一整块 tile 从 HBM / L2 异步搬进 shared memory，搬运时 warp 可以去算别的。

## 每一层的数字

| 层级 | 谁能访问 | 容量（H100 SXM5） | 延迟（时钟周期） | 带宽 | 谁来管 |
|---|---|---|---|---|---|
| 寄存器 | 单个线程 | 256 KB / SM，每线程最多 255 个 | 指令直接读 | — | 编译器分配 |
| shared memory | 同一个 block 的线程 | 最多 228 KB / SM | ≈ 29 | 128 B / 时钟 / SM | 程序员显式读写 |
| L1 | 同一个 SM | 256 KB 里 shared 剩下的部分 | ≈ 41 | ≈ 125 B / 时钟 / SM | 硬件 |
| L2 | 整张卡 | 50 MB | ≈ 263 | 约 HBM 的 4 倍 | 硬件（可设驻留策略） |
| HBM | 整张卡 | 80 GB | ≈ 479 | 3.35 TB/s | 程序员（`cudaMalloc`） |

容量来自 NVIDIA 的规格表。延迟和 L1 / shared / L2 带宽是 Luo et al. 在 H800（同为 Hopper 架构）上用 pointer-chasing 微基准测的（[arXiv:2402.13499](https://arxiv.org/abs/2402.13499) Table IV、V，作者自测）：L2 的延迟约是 L1 的 6.5 倍，HBM 又是 L2 的 1.9 倍；L2 的带宽是 HBM 的 4.23 倍。

两个换算，方便和 HBM 比：

1. **shared memory 的总带宽**：128 B/时钟 × 132 个 SM × 约 1.8 GHz ≈ **30 TB/s**，约是 HBM 的 9 倍。时钟频率随负载变化，这只是量级估算。
2. **延迟换成时间**：479 个时钟周期在 1.8 GHz 下约 270 ns。一个 warp 发出 HBM 读请求后要等这么久，所以 SM 要靠同时驻留很多 warp 轮流执行来掩盖它，见 [GPU 架构](/gpu/gpu-architecture) 的 occupancy 一节。

## 推理时每一层装什么

以 Llama-3-70B、bf16、TP = 4 为例，每张卡上：

1. **HBM**：权重 140 GB ÷ 4 = 35 GB，加 KV cache。decode 每一步都要把这 35 GB 读一遍，35 × 10⁹ ÷ 3.35 × 10¹² ≈ **10 ms**，这就是每步时间的下限。
2. **L2**：50 MB 只是 35 GB 权重的 0.14%，权重放不进去，每一步都只能从 HBM 流过。L2 的作用是让**同一个 kernel 里**被多个 block 用到的数据只从 HBM 读一次，比如 GEMM 里相邻 tile 共用的 A 行条带和 B 列条带。Triton matmul 的 `GROUP_M` 就是为了提高 L2 命中，见 [Triton](/gpu/triton)。
3. **shared memory**：放当前正在算的 tile。GEMM 取 BM = BN = 128、BK = 64，一个 stage 存 A、B 各一块：(128 × 64 + 64 × 128) × 2 B = 32 KB，4 级流水 128 KB，放得进 228 KB。FlashAttention 放的是 Q、K、V 的 tile，score 矩阵从不写回 HBM，见 [FlashAttention](/inference/flash-attention)。
4. **寄存器**：放累加器。128 × 128 的 fp32 累加器是 64 KB，正好占一个 SM 寄存器的四分之一，所以大 tile 的 GEMM kernel 一个 SM 上只能驻留很少的 block。

## 一个 tile 的数据怎么走

以 GEMM 为例，每一行注释标出数据在哪一层：

```python
acc = zeros(BM, BN, fp32)                 # 寄存器：累加器，算完才写回
for k in range(0, K, BK):
    A_s = load_tile(A, rows, k)           # HBM（经 L2）→ shared memory，TMA 异步搬
    B_s = load_tile(B, k, cols)
    for kk in range(0, BK, 16):           # shared memory → 寄存器 → Tensor Core
        acc += mma(A_s[:, kk:kk+16], B_s[kk:kk+16, :])
store(C, rows, cols, acc.to(bf16))        # 寄存器 → HBM，只写一次
```

A 的每个元素从 HBM 读进来一次，在 shared memory 里被这个 tile 的 BN = 128 列复用；B 的每个元素被 BM = 128 行复用。复用次数越多，每读 1 字节做的运算越多，算术强度就越高，具体推导见 [Tensor Core 与 GEMM](/gpu/tensor-core-gemm)。

decode 的 GEMV 里 M 就是 batch，比如 B = 1 时 A 只有一行，B 的每个元素只被用 1 次。没有可复用的东西，tiling 也帮不上忙，只能等 HBM，这就是 [roofline](/inference/prefill-decode-roofline) 里 decode 落在斜线上的原因。

**延伸阅读**：[CUDA C++ Best Practices Guide：Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)、[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
