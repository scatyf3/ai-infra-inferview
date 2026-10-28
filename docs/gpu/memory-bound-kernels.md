---
title: Memory-bound Kernel 总览
status: draft
tags: [kernel, bandwidth, fusion]
difficulty: 3
order: 5
related: [/gpu/gpu-architecture, /gpu/cuda-graph-fusion, /gpu/tensor-core-gemm, /gpu/profiling, /handson/cuda-reduce, /handson/rmsnorm, /handson/online-softmax, /handson/top-p-sampling, /gpu/quant-gemm, /gpu/moe-grouped-gemm]
stack: [k-fused]
---

# Memory-bound Kernel 总览

> 算得再快也没用，时间等于字节除以带宽

## 一句话结论

推理里除了 GEMM 和 attention，剩下的 kernel 几乎都是 memory-bound：norm、激活、RoPE、残差加、量化、采样、写 KV cache、MoE 的排队……每个元素只做几次运算，算术强度在 1 FLOP/B 上下，离 H100 的 ridge（295 FLOP/B）差两个数量级。它们的时间下界就是

$$
t \ge \frac{\text{必须读写的字节}}{\text{HBM 带宽}}
$$

优化只有两个方向：

1. **少搬**：中间结果不落 HBM（融合），或者用更小的 dtype，把分子变小。
2. **搬满**：让每个字节都以接近峰值的带宽搬，靠四件事：访存合并、向量化、在途字节够多、block 够多。

衡量写得好不好，看的不是 FLOP 利用率，而是**有效带宽 = 必须读写的字节 ÷ 实际耗时**，拿它和同样字节数的 device-to-device memcpy 比。memcpy 是这块卡上搬数据的实际上限。

难度主要取决于访存模式：

| 类型 | 例子 | 一个输出要读什么 | 难点 |
|---|---|---|---|
| elementwise | 残差加、SiLU · up、RoPE、类型转换、per-tensor 量化 | 同一位置的几个输入 | 访存合并 + 向量化就够 |
| 行归约 | RMSNorm、LayerNorm、softmax、per-token 量化 | 整行 | 线程之间要归约；一行放不放得进片上，决定读一遍还是两遍 |
| 全局归约 / 选择 | 求和、argmax、top-k / top-p 采样 | 整个张量或整个词表 | 跨 block 汇总；确定性 |
| gather / scatter | embedding 查表、写 paged KV cache、MoE permute | 按索引找到的行 | 地址由索引决定，只能保证行内连续 |
| 重排 | 转置、权重 repack | 别处的元素 | 读和写总有一边不连续，要过 shared memory |

## 先算账

RMSNorm，$X$ 是 $[T, d] = [4096, 8192]$ 的 bf16：

- 字节：读 $X$ 64 MiB，写 $Y$ 64 MiB，$\gamma$ 只有 16 KiB，共 128 MiB。按 3.35 TB/s 至少要 **40 μs**。
- FLOP：每个元素平方、累加、乘 rsqrt、乘 $\gamma$，约 4 次，共 $1.3 \times 10^8$。就算全在 FP32 CUDA core 上（67 TFLOP/s）也只要 **2 μs**。

算术强度约 1 FLOP/B，时间完全由字节决定。把计算再优化一倍，耗时不变；少读写一份 $X$，耗时就少四分之一。

decode 时 $T$ 很小，问题又变了：$T = 64$ 时同一个 RMSNorm 只有 2 MiB，0.6 μs 就搬完了，比 kernel launch 本身的开销还小。这时瓶颈是 kernel 的个数，要靠融合和 CUDA Graph（见 [CUDA Graph 与 Kernel Fusion](/gpu/cuda-graph-fusion)）。

## 少搬：融合

几个最常见的融合，每元素读写的字节（bf16 记 2 B，FP8 记 1 B）：

| 融合 | 不融合 | 融合后 | 省多少 |
|---|---|---|---|
| 残差加 + RMSNorm | add 读 2 写 1，norm 读 1 写 1：10 B | 读 $x$、residual，写新 residual、norm 输出：8 B | 20% + 1 次 launch |
| SiLU(gate) · up | 两个 elementwise 各自读写 | 读 gate、up，写一次 | 40% + 1 次 launch |
| RMSNorm + per-token FP8 量化 | norm 读 2 写 2，量化读 2 写 1：7 B | 读 2，写 1：3 B | 57% + 1 次 launch |
| elementwise 融进 GEMM 的 epilogue | GEMM 写 $C$，elementwise 再读写 $C$ | 在寄存器里做完再写 | 省 $C$ 的一次完整读写 |

前两行的推导、Inductor 和 vLLM 里谁来做融合、什么时候融不了，见 [CUDA Graph 与 Kernel Fusion](/gpu/cuda-graph-fusion)。第三行能融，是因为 per-token 量化要的 amax 也是按行算的，和 RMSNorm 的行归约在同一个 block 里，一行读进寄存器之后两件事一起做完。

融合的边界是**归约的范围**：后一步需要的东西，必须是同一个 block 能自己算出来的。RMSNorm 需要整行的平方和，而 GEMM 的一个 tile 只有一行里的 $BN$ 列，所以 RMSNorm 一般融不进前一个 GEMM 的 epilogue，只能和残差加、量化这些同样按行处理的步骤融在一起。

## 搬满：四件事

### 1. 访存合并

一个 warp 的 32 个线程同时发 load，地址挨在一起时合成几个完整的 32 B sector，否则每个线程各占一个 sector，带宽浪费几倍。规则是**让相邻线程读相邻地址**，原理见 [GPU 执行模型：coalescing](/gpu/gpu-architecture#基本功一-coalescing-读-hbm-时让-32-个地址挨在一起)。

### 2. 向量化：每个线程一次读 16 B

带宽是一样的，为什么一次读 16 B（`uint4`，8 个 bf16）比读 2 B 快？因为同样多的字节，load 指令少了 8 倍：

- 发射槽：memory-bound 的 kernel 每条 load 之后只有几条计算，指令少了，LSU 和发射槽都不再是瓶颈。
- 在途字节：每条 load 带 16 B，同样的 warp 数下路上的字节多了 8 倍，更容易满足下一条（[CUDA Pro Tip: Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)）。

```cuda
// out = silu(gate) * up，bf16；每个线程每次处理 8 个元素（16 B）
// 假设 n 是 8 的倍数、指针 16 B 对齐，尾部另行处理
__global__ void silu_mul(const bf16* gate, const bf16* up, bf16* out, int n) {
    int stride = gridDim.x * blockDim.x * 8;
    for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8; i < n; i += stride) {
        uint4 g = *(const uint4*)(gate + i);   // 一条 16 B 的 load
        uint4 u = *(const uint4*)(up + i);
        uint4 o;
        // 拆成 8 个 bf16，转 fp32 算 silu(g) * u，再打包回 bf16
        ...
        *(uint4*)(out + i) = o;                // 一条 16 B 的 store
    }
}
```

前提是地址 16 B 对齐：每行的长度是 8 的倍数（bf16），起始地址对齐。不对齐的尾巴单独用标量处理。

### 3. 在途字节够多

按 Little 定律，要让 HBM 一直忙，每个 SM 同时要有约 7–13 KB 的请求在路上（推导和几种设计的对照见 [GPU 执行模型：Little 定律](/gpu/gpu-architecture#_4-需要多少-little-定律)）。在途字节 = 驻留 warp 数 × 每个 warp 同时在等的字节。两种凑法：

- 多挂 warp（occupancy 高）；
- 每个线程连着发几条互不依赖的 16 B load，全部发出去之后再用（ILP），occupancy 不高也够。

### 4. block 够多，别留尾巴

grid 至少要让每个 SM 都有活，最好是好几波。元素太少、block 不够的时候，前三件事都做对了也跑不满，这正是 decode 下小 kernel 的处境，只能靠融合减少 kernel 数。常见写法是 grid-stride loop：开「SM 数 × 每 SM 能驻留的 block 数」个 block，每个线程循环处理多段，block 数不随数据量变。

## 行归约：一行放不放得进片上

RMSNorm $y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$ 要先有整行的平方和，才能算每个输出。最省字节的做法是**一行只读一遍**：读进寄存器，归约出平方和，再从寄存器里算输出写回。能不能做到，看一行放不放得进一个 block 的寄存器：

| 一行多大 | 怎么分 | 例子 |
|---|---|---|
| 小（几百个元素） | 一个 warp 一行，一个 block 处理好几行 | 每个 head 的 QK-norm，$d = 128$ |
| 中（几千到一两万） | 一个 block 一行，整行留在寄存器 | RMSNorm，$d = 8192$ |
| 大（几十万） | 读两遍，或一行拆给多个 block | 对 128K 词表的 logits 做 softmax / 采样 |

$d = 8192$ 的 bf16 一行 16 KB。256 个线程，每个线程拿 32 个元素，就是 4 条 16 B 的 load。归约分两层：warp 内用 shuffle 做 5 步，8 个 warp 的结果经 shared memory 再归约一次，最后把 rsqrt 广播给所有线程。完整写法见 [手撕 RMSNorm](/handson/rmsnorm)。

一行太大放不下时只能多读几遍。softmax 朴素写法要读三遍（max、sum、归一化），[online softmax](/handson/online-softmax) 把 max 和 sum 合成一遍，总共两遍。

## 全局归约：跨 block 怎么汇总

整个张量求和、argmax 这类，每个 block 先算自己那段的部分结果，再汇总。三种汇总方式：

| 方式 | 做法 | 代价 |
|---|---|---|
| 两个 kernel | 第一个写每个 block 的部分和，第二个读回来加 | 多一次 launch；结果确定 |
| atomic | 每个 block 直接 `atomicAdd` 到结果 | 一个 kernel；但浮点加法顺序随调度变，结果每次可能不同 |
| 最后一个 block 汇总 | 每个 block 写完部分和后计数，最后到的那个负责加 | 一个 kernel，结果确定，写法复杂一点 |

推导和代码见 [手撕 CUDA Reduce](/handson/cuda-reduce)。结果不确定在 RL 里会变成训推不一致，见 [训推不一致](/posttrain/rl-train-infer-mismatch)。采样（top-k / top-p）是对每一行词表做选择，属于行归约里「一行很大」的那一档，见 [手撕 Top-p 采样](/handson/top-p-sampling)。

## gather / scatter：让不规则只发生在「哪一行」

embedding 查表 `out[t] = table[ids[t]]`、写 paged KV cache（按 `slot_mapping` 把每个 token 的 K、V 写进对应的 page）、MoE 按 expert 排队（[MoE Grouped GEMM](/gpu/moe-grouped-gemm#_1-排队-token-按-expert-分段)），地址都由一个索引数组决定。

做法是**让一个 warp（或一个 block）负责一整行**：索引只决定去哪一行，行内的 $d$ 个元素仍然连续，照样能合并、能向量化。反过来，如果让相邻线程处理不同的行，每个线程的地址都跳到别处，合并就没了。

## 重排：经过 shared memory 转一道

转置这类 kernel 按行读、按列写（或反过来），总有一边不连续。做法是一个 block 先按行合并地读一个 tile 进 shared memory，再按列从 shared memory 取、合并地写出去。shared memory 里列方向的访问会撞 bank，要加 padding 或 swizzle，见 [GPU 执行模型：手撕矩阵转置](/gpu/gpu-architecture#手撕-矩阵转置)。权重 repack（[量化 GEMM](/gpu/quant-gemm#权重离线重排)）也是重排，但在加载模型时只做一次，不在推理路径上。

## 怎么判断写得好不好

1. **先算下界**：必须读写的字节 ÷ 带宽，和实测时间比，得到有效带宽。
2. **和 memcpy 比**：同样字节数的 device-to-device 拷贝能跑到多快，就是这块卡上实际能达到的上限。离它很近，就没什么可优化的了。
3. **用 ncu 找原因**：SOL 里看 DRAM Throughput（不是 Memory Throughput）；Memory Workload Analysis 里看实际读写的字节是否比公式多（多出来的是浪费），以及每个请求用了几个 sector（访存是否合并）。见 [Profiling：ncu](/gpu/profiling#_2-实测判断-compute-bound-还是-memory-bound)。

## 面试追问

::: details Q：带宽是固定的，为什么向量化 load 能变快？
同样的字节，16 B 一条的 load 比 2 B 一条的少 8 倍指令。memory-bound 的 kernel 里 load 指令本身就占了大部分发射槽，指令少了不再卡在发射上；而且每条 load 带的字节多，同样多的 warp 能让更多字节在路上，更容易满足 Little 定律。
:::

::: details Q：RMSNorm 能融进前一个 GEMM 的 epilogue 吗？
一般不能。RMSNorm 要整行的平方和，GEMM 的一个 tile 只有这一行的 $BN$ 列，算不出整行的结果，除非一个 block 负责整行的所有列。所以实践里 RMSNorm 和同样按行处理的残差加、per-token 量化融在一起，而不是和 GEMM 融。
:::

::: details Q：decode 时这些小 kernel 的瓶颈是什么？
batch 小时每个 kernel 只搬几 MB，不到 1 μs 就搬完了，比 launch 的开销还小，GPU 大部分时间在等下一个 kernel 发过来。解决办法是减少 kernel 数（融合）和减少 launch 次数（CUDA Graph）。
:::

::: details Q：求和用 atomicAdd 有什么问题？
浮点加法不满足结合律，atomic 的到达顺序由调度决定，每次运行的求和顺序不同，结果的低位可能不一样。要逐位可复现，就用两个 kernel 或「最后一个 block 汇总」，顺序固定。
:::

::: details Q：gather kernel 里，线程该怎么对应到数据？
一个 warp 或一个 block 负责一整行，线程在行内按列分。这样不规则的只是「去哪一行」，行内地址连续，访存能合并、能用 16 B 向量化。如果相邻线程负责不同的行，每个线程的地址都跳开，合并就没了。
:::

## 参考

- [CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
- [CUDA C++ Best Practices Guide：Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Mark Harris：Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
