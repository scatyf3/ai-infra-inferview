---
title: "第 2 层：tensor → kernel"
---

# 第 2 层：tensor → kernel

单个算子在 GPU 上怎么跑。一层 decoder 的 forward 拆开只有三类 kernel：线性层的 GEMM、attention，以及一批逐元素或按行的小算子。这一层只问一件事：每个 kernel 离它的 roofline 上限还有多远。

边界：这一层的单位是**一个 kernel**，它内部怎么切块、怎么读写 HBM。多个 kernel 怎么串成一次 forward、CPU 怎么把它们发射出去（CUDA Graph、torch.compile 抓图），属于 [第 3 层](/stack/layer-3)。

## 先分类：每个 kernel 落在 roofline 哪一边

判断标准见 [roofline](/inference/prefill-decode-roofline)：算术强度 AI = FLOP ÷ 字节，和 ridge（H100 bf16 是 $989 / 3.35 \approx 295$ FLOP/B）比。AI 大于 ridge 是 compute-bound，时间下界是 FLOP ÷ 峰值；小于 ridge 是 memory-bound，时间下界是字节 ÷ 带宽。

记号：$T$ 是 prefill 的 token 数，$B$ 是 decode 的 batch（每个请求 1 个 token），$S$ 是序列长度，$g$ 是 GQA 里共用一组 KV 的 query head 数，bf16 每个数 2 字节。

| kernel | prefill 的 AI | decode 的 AI | 怎么来的 |
|---|---|---|---|
| 线性层 GEMM | $\approx T$，上千，compute-bound | $\approx B$，几十，memory-bound | $M = T$ 或 $B$，$K$、$N$ 是隐藏维；$M$ 小时字节几乎全是权重 $2KN$，FLOP 是 $2MKN$ |
| attention | $\approx S / 2$，compute-bound（前提是不把 $S \times S$ 的分数写回 HBM） | $\approx g$，memory-bound | prefill 每个 head 做 $4S^2 d_h$ FLOP，读写 Q、K、V、O 共 $4 S d_h \times 2$ 字节；decode 每组 KV 被 $g$ 个 query 各用一次，FLOP $4 g S d_h$，读 K、V $2 S d_h \times 2$ 字节 |
| RMSNorm、RoPE、SiLU × up、残差加、量化 | $\approx 1$ | $\approx 1$，而且字节少到比 launch 开销还小 | 每个元素读一次、写一次，只做几次运算 |

数字代一下：Llama-3-70B（$g = 8$）、H100 上，decode 的 GEMM 在 batch 64 时 AI ≈ 64，attention 不管 batch 多大都 ≈ 8，都远在 295 左边。prefill 2048 个 token 时 GEMM 和 attention 都在右边。所以同一个算子在两个阶段要用不同的手段优化。

## compute-bound：把 Tensor Core 喂饱

prefill 的 GEMM 和 attention 属于这一类。手段是 [GEMM tiling](/gpu/tensor-core-gemm)：把 $C$ 切成 $BM \times BN$ 的块，每块读进片上的 $A$、$B$ 被另一维复用，每读 1 个数做 $\frac{BM \cdot BN}{BM + BN}$ 次 MAC；同样的切法在 block（shared memory）、warp（寄存器）、指令（`mma`）三级各做一次。

在 RTX 5070 Ti 上把这些手段用 Triton 一级一级加上去（bf16，$M = N = K = 4096$，单位 TFLOP/s，这张卡的峰值约 101，[实测](/gpu/tensor-core-gemm#实测-triton-逐级优化-rtx-5070-ti)）：

| 版本 | 改动 | TFLOP/s |
|---|---|---|
| v0 | 每个 program 算一个输出 | 0.11（n = 2048，再大太慢） |
| v1 | block tile，CUDA core | 28 |
| v2 | 换 Tensor Core | 70 |
| v4 | 加多级流水 | 90 |
| v5 | autotune 挑 tile | 95（cuBLAS 90；n = 16384 时 98 对 99） |

复用（v1）和 Tensor Core（v2）决定数量级，流水（v4）把 Tensor Core 等数据的时间藏掉，autotune（v5）主要救小矩阵和不整齐的形状（wave quantization）。

同一套 tiling 的几种变体：

- **attention**：[FlashAttention](/inference/flash-attention) 就是把 $QK^\top$、softmax、$PV$ 放进同一个 tile 循环，靠 online softmax 不把分数矩阵写回 HBM，AI 从约 63 升到约 4000。
- **低精度**：[FP8 GEMM](/gpu/quant-gemm#fp8-直接进低精度-tensor-core) 直接进 FP8 Tensor Core，峰值翻倍；scale 不随 $k$ 变就放在 epilogue 乘。
- **MoE**：[grouped GEMM](/gpu/moe-grouped-gemm) 把各 expert 大小不一的 GEMM 摊成一个 tile 列表，一个 kernel 跑完；tile 内部还是普通 tiling。

## memory-bound：少搬、搬满

decode 的几乎全部 kernel 和所有小算子属于这一类。时间下界是字节 ÷ 带宽，只有两个方向（[Memory-bound Kernel 与 Kernel Fusion](/gpu/memory-bound-kernels)）：

1. **少搬**：
   - **融合**：几个相邻算子合成一个 kernel，中间结果留在寄存器里不落 HBM。残差加 + RMSNorm 省 20%，SiLU × up 省 40%，norm + FP8 量化省 57%；attention 融合（FlashAttention）省的是 $O(S^2)$，这是数量级。
   - **压缩权重**：[W4A16](/gpu/quant-gemm#w4a16-格式转换放到最后一步) 让 decode 的 GEMM 读的权重字节 ÷ 4，转换放在寄存器里、`mma` 之前做。
   - **压缩 KV**：GQA / MLA、KV 量化让 decode attention 读的 KV 变少。
2. **搬满**：访存合并、向量化（每线程一次 16 B）、在途字节够多、block 够多。decode 的 GEMV 只有一行 tile，block 不够时用 [split-K](/gpu/split-k-stream-k)；decode attention 序列长、batch 小时用 split-KV（[FlashDecoding](/inference/flash-attention#decode-flashdecoding-split-kv)）。

decode 时小算子只有一两 MiB，不到 1 μs 就搬完，比一次 kernel launch 还短。这时逐元素融合省的字节可以忽略（70B、batch 64 时残差加 + RMSNorm 一共省约 50 μs，权重读取要 42 ms），收益主要是 kernel 个数减半。融合之后剩下的发射开销不是单个 kernel 能解决的，交给第 3 层的 [CUDA Graph](/framework/cuda-graph)。

## 用什么写、怎么确认

- **写**：[Triton](/gpu/triton) 以 tile 为单位写，适合融合算子和形状特殊的 kernel，标准形状的 GEMM 也能接近 cuBLAS（上面的 v5）；要用 Hopper / Blackwell 最新的硬件特性（TMA、warp specialization、`wgmma` / `tcgen05`），一般先在 CUTLASS 和手写 CUDA 里出现，这些指令见 [普通 SIMT 之外的指令](/gpu/special-instructions)。
- **确认**：先用 torch profiler / nsys 找「GPU 在等谁」，再用 ncu 看单个 kernel 的带宽和 Tensor Core 利用率，判断它是 compute-bound 还是 memory-bound、离上限多远（[Profiling](/gpu/profiling)）。

## 建议阅读顺序

1. [GPU 执行模型](/gpu/gpu-architecture)、[计算单元](/gpu/compute-units)：warp、shared memory、occupancy，峰值从哪来。
2. [GEMM Tiling](/gpu/tensor-core-gemm)：本层的骨架，后面的页面都在它上面改。
3. [量化 GEMM](/gpu/quant-gemm)、[MoE Grouped GEMM](/gpu/moe-grouped-gemm)：tiling 的两种变体。
4. [FlashAttention](/inference/flash-attention)：tiling + online softmax。
5. [Memory-bound Kernel 与 Kernel Fusion](/gpu/memory-bound-kernels)：剩下的小算子，融合、带宽、访存模式。
6. [Triton](/gpu/triton)、[Profiling](/gpu/profiling)：写和测。
