---
title: Split-K 与 Stream-K
status: draft
tags: [gemm, scheduling]
difficulty: 3
order: 2.5
related: [/gpu/tensor-core-gemm, /gpu/quant-gemm, /gpu/gpu-architecture, /inference/flash-attention, /gpu/triton, /posttrain/rl-train-infer-mismatch]
stack: [k-gemm]
train: [tr-k-gemm]
---

# Split-K 与 Stream-K

> 按输出 tile 分活不够用时，沿 $K$ 再切

## 一句话结论

普通 GEMM 一个 block 算一个输出 tile，沿 $K$ 从头累加到尾（data-parallel）。这样分活有两种不够用的情况，都靠沿 $K$ 再切来解决，代价都是多一次部分和的合并：

| | 解决什么 | 怎么切 | 典型场景 |
|---|---|---|---|
| **Split-K** | tile 太少，SM 喂不满 | 每个 tile 的 $K$ 固定切 $S$ 段，一段一个 block | decode 的 GEMV、输出小而 $K$ 长 |
| **Stream-K** | tile 数不是 SM 数的整数倍（wave quantization） | 所有 tile 的 K 迭代排成一条队，平均分给每个 SM | 中等形状、tile 数落在两波之间 |

![同一个例子（4 个 SM、5 个 tile、每 tile 8 轮）下三种分法的时间线：data-parallel 排 2 波共 16 轮、效率 62.5%；Split-K S=2 排 3 波共 12 轮加合并、效率 83%；Stream-K 每个 CTA 连续领 10 轮、效率 100%，被切开的 tile 由起头的 CTA 收部分和](/figures/split-k-stream-k.svg)

两者都让浮点加法的顺序随形状和 SM 数变，结果不能逐位复现，见 [确定性](#确定性)。

## 约定

$C = AB$，$A$ 是 $M \times K$，$B$ 是 $K \times N$，输出 tile $BM \times BN$，主循环每轮沿 $K$ 走 $BK$。一个 tile 的主循环要 $K / BK$ 轮，下面叫一个 tile 的**迭代数**。H100 SXM：132 个 SM，bf16 稠密峰值 989 TFLOP/s，HBM 3.35 TB/s。部分和用 fp32，每个数 4 B。

## 问题：一个 tile 一个 block 什么时候不够

data-parallel 的 grid 是 $\lceil M/BM \rceil \times \lceil N/BN \rceil$ 个 block，每个 SM 同时跑 1 个（大 tile GEMM 常见）。按波上 SM，[效率](/gpu/tensor-core-gemm#wave-quantization-tile-数和-sm-数对不齐) 是

$$
\text{效率} = \frac{\text{tile 数}}{\lceil \text{tile 数} / 132 \rceil \times 132}
$$

两个例子：

1. **decode 的 GEMV**：$M = 1$，$N = K = 8192$，$BN = 128$，只有 $8192 / 128 = 64$ 个 tile，一半 SM 没活干。这里 kernel 是 memory-bound，SM 闲着的后果是在途字节不够、HBM 带宽打不满（[Little 定律](/gpu/gpu-architecture#基本功三-occupancy-挂够多的-warp-来藏延迟)）。
2. **两波之间的形状**：$M = N = 1536$，$128 \times 128$ 的 tile 有 144 个，排 2 波，第二波只有 12 个 tile，效率 $144 / 264 = 55\%$。

两个例子的共同点是：一个 tile 的 $K$ 方向只能由一个 block 串行做完，分活的最小单位太大。

## Split-K：每个 tile 的 $K$ 固定切 $S$ 段

### 怎么切

把 $K$ 切 $S$ 段，每段长 $K / S$。grid 多一维，变成 $\text{tile 数} \times S$ 个 block，每个 block 只累加自己那一段，得到一个部分和 $P_s$，最后 $C = \sum_s P_s$。

```python
# grid = (num_tiles, S)
tile, s = program_id(0), program_id(1)
k_lo, k_hi = s * K // S, (s + 1) * K // S
acc = zeros(BM, BN, fp32)
for k0 in range(k_lo, k_hi, BK):              # 只走 1/S 的 K
    acc += A[tile.m, k0:k0+BK] @ B[k0:k0+BK, tile.n]
merge(acc)                                     # 三种合并方式见下表
```

decode 的例子取 $S = 4$：64 → 256 个 block，每个 block 读 $2048 \times 128$ 的权重。

### 部分和怎么合并

| 方式 | 做法 | 代价 | 结果可复现吗 |
|---|---|---|---|
| 单独的 reduce kernel | 每段写到 workspace 的第 $s$ 层（$S \times M \times N$ 个 fp32），再起一个 kernel 沿 $s$ 加起来；CUTLASS 叫 parallel split-K（[CUTLASS: Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)） | 多一次 kernel launch，多一趟 workspace 读写 | 能（加法顺序固定） |
| atomic | 每段直接 `atomicAdd` / `tl.atomic_add` 进 $C$，$C$ 预先清零 | 最简单；$C$ 得是 fp32，或者再转一次 | 不能（谁先加完不固定） |
| 串行 fixup | 第 0 段等其余段写完部分和（每段一个 flag），读进来加上，写 $C$（[Osama et al., 2023, arXiv:2301.03598](https://arxiv.org/abs/2301.03598) 里的 fixed-split） | 一个 kernel 内完成，但第 0 段要等 | 能 |

Triton 里写 atomic 版时注意 autotune 会把 kernel 跑很多遍，$C$ 要放进 `reset_to_zero`，见 [Triton：autotune](/gpu/triton#autotune)。

### 什么时候划算

合并多出的流量是 $S$ 份部分和，写一遍、读一遍：$2 \times S \times M \times N \times 4$ B。拿它和 GEMM 本身要读的 $A$、$B$ 比：

- **decode GEMV**，$M = 1$、$N = K = 8192$、$S = 4$：部分和 $2 \times 4 \times 8192 \times 4 = 256$ KB，权重 $8192^2 \times 2 = 128$ MB，多 0.2%，换来两倍多的 block。划算。
- **大方阵**，$M = N = K = 4096$、$S = 2$：部分和 $2 \times 2 \times 4096^2 \times 4 = 256$ MB，比 $A + B$ 的 64 MB 还多；而且 tile 本来就有 1024 个，不缺并行度。不划算。

规律：输出小（$MN$ 小）而 $K$ 长时用 split-K；输出大时 tile 本来就够，切了只是多搬数据。$S$ 一般由 heuristic 或 autotune 按形状选。

## Stream-K：按迭代平均分，不按 tile 分

### 怎么分

split-K 的 $S$ 是对每个 tile 一样切，tile 数 × $S$ 仍然很难正好是 SM 数的整数倍。Stream-K 换一个分活单位：把所有 tile 的所有迭代排成一条队，总长 $\text{tile 数} \times \text{每 tile 迭代数}$，启动 $g$ 个常驻 CTA（一般等于 SM 数），每个 CTA 连续领 $\lceil \text{总迭代} / g \rceil$ 轮（[Osama et al., 2023](https://arxiv.org/abs/2301.03598)）。

拿上面 1536 的例子，$K = 4096$、$BK = 64$，每个 tile 64 轮：

$$
\text{总迭代} = 144 \times 64 = 9216, \qquad \lceil 9216 / 132 \rceil = 70\ \text{轮 / CTA}
$$

| CTA | 领到的迭代 | 落在哪些 tile |
|---|---|---|
| 0 | 0–69 | tile 0 整个（局部 0–63）+ tile 1 的局部 0–5 |
| 1 | 70–139 | tile 1 的局部 6–63 + tile 2 的局部 0–11 |
| … | … | … |
| 131 | 9170–9215 | tile 143 的局部 18–63（只有 46 轮） |

最慢的 CTA 跑 70 轮，data-parallel 要 2 波 × 64 = 128 轮，理想值 $9216 / 132 \approx 69.8$ 轮。效率从 55% 到 $9216 / (70 \times 132) = 99.7\%$，再扣掉合并的开销。

### 谁来合并

一个 tile 被切给了多个 CTA，比如上表的 tile 1：CTA 0 算了 $k$ 的前 6 轮，CTA 1 算了后 58 轮。论文的规则是**算了这个 tile 第 0 轮的 CTA 负责写 $C$**，其他 CTA 把部分和存进 workspace，再置 flag：

```python
# Stream-K，一个 CTA（论文 Algorithm 5 的结构）
it, it_end = x * iters_per_cta, min((x + 1) * iters_per_cta, total_iters)
while it < it_end:
    tile       = it // iters_per_tile
    tile_begin = tile * iters_per_tile
    tile_end   = tile_begin + iters_per_tile
    acc = mac_loop(tile, it - tile_begin, min(it_end, tile_end) - tile_begin)
    if it != tile_begin:                       # 没算第 0 轮：交部分和
        store_partials(partials[x], acc)
        signal(flags[x])
    else:                                      # 算了第 0 轮：收齐再写 C
        if it_end < tile_end:                  # 后半段在后面的 CTA 手里
            for peer in ctas_covering(tile, after=x):
                wait(flags[peer])
                acc += load_partials(partials[peer])
        store_tile(C, tile, acc)
    it = tile_end
```

为什么让起头的那个 CTA 收：每个 CTA 按顺序走自己的区间，一个 tile 的开头总是落在某个 CTA 区间的**末尾**，它的后半段落在下一个 CTA 区间的**开头**。上例中 CTA 1 在第 0–57 轮就把 tile 1 的部分和交了，CTA 0 到第 64–69 轮才碰 tile 1，等它算完那 6 轮，部分和早就在了，基本不用等。

合并的开销：被切开的 tile 最多 $g - 1 = 131$ 个，每个多一份 $128 \times 128 \times 4 = 64$ KB 的部分和，写一遍读一遍。一个 CTA 主循环 70 轮，每轮读 $(128 + 128) \times 64 \times 2 = 32$ KB 的操作数，共约 2.2 MB，部分和只占约 3%，而且 131 × 64 KB ≈ 8 MB 基本留在 L2（H100 有 50 MB）。

### hybrid：只让最后一两波用 Stream-K

纯 Stream-K 有两个问题：每个 CTA 起点错开，同时在算的 tile 不再整齐，[L2 复用](/gpu/tensor-core-gemm#l2-让同时跑的-block-读同一份数据) 变差；tile 少到每个 CTA 分不到一个 tile 时，一个 tile 要从 3 个以上 CTA 收部分和，等待藏不住。论文的做法是前面整波照常 data-parallel，只把尾巴交给 Stream-K。设 tile 数 $t$，SM 数 $p$：

| 方案 | data-parallel | Stream-K | 问题 |
|---|---|---|---|
| DP + one-tile SK | $\lfloor t/p \rfloor$ 整波 | 剩下 $t \bmod p$ 个 tile | 每个 CTA 分到不到 1 个 tile，一个 tile 可能跨 3 个 CTA，同步等待藏不住 |
| **two-tile SK + DP** | 少一波：$\lfloor t/p \rfloor - 1$ 整波 | 剩下 $p + (t \bmod p)$ 个 tile | 每个 CTA 分到 1–2 个 tile，起头的 CTA 只需要收**一个**别人的部分和 |

4096³ 的例子：$t = 1024$，$p = 132$，$1024 = 7 \times 132 + 100$。two-tile 方案是 6 波 data-parallel（792 个 tile），剩下 232 个 tile 交给 Stream-K，每个 CTA 约 $232 / 132 \approx 1.76$ 个 tile。

论文报告单一 tile 配置的 Stream-K 在 32,824 个形状上平均性能比 CUTLASS 和 cuBLAS 更高、更稳，峰值加速最高 14 倍（作者自测，[arXiv:2301.03598](https://arxiv.org/abs/2301.03598)）。卖点不只是快，而是**一种 tile 配置覆盖所有形状**，不用再靠 heuristic 在几十个 kernel 变体里挑。

## 同一个思路在别处

| 地方 | 沿哪一维切 | 合并 |
|---|---|---|
| [Flash-Decoding / split-KV](/inference/flash-attention#decode-flashdecoding-split-kv) | KV 长度 | 按 lse 加权求和，不是直接相加 |
| [Marlin 的 striped partitioning](/gpu/quant-gemm#小-batch-时的并行度) | (K 段, N 段) 平均分给 SM | 同一列的部分和最后合并，思路和 Stream-K 相同 |
| sliced-K（block 内） | 一个 block 里的几个 warp 各算一段 $K$ | 经 shared memory 在 warp 间归约（[CUTLASS: Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)） |

## 确定性

浮点加法不满足结合律，加法顺序一变结果就可能差最后几位。三种情况：

1. **atomic 合并**：同一次调用、同一个形状，跑两遍都可能不一样。
2. **reduce kernel、串行 fixup、Stream-K**：给定形状、给定 $S$ 或 SM 数，顺序固定，可以复现。但切在哪里取决于形状：split-K 的 $S$ 由 heuristic 按 $M$ 选，Stream-K 的切点由 $\text{总迭代} / g$ 决定，$M$ 一变（也就是 batch 一变）切点就挪。
3. 所以推理服务里，同一个请求和不同的请求拼进不同大小的 batch，matmul 的加法顺序跟着变，输出可能不同。Thinking Machines 的 batch-invariant matmul 就是关掉 split-K / Stream-K、所有形状用同一套配置（[He, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)），代价和对 RL 的意义见 [训推不一致](/posttrain/rl-train-infer-mismatch)。

## 面试追问

::: details Q：split-K 和 Stream-K 有什么区别？
split-K 每个 tile 的 $K$ 固定切 $S$ 段，解决「tile 太少」，分活单位还是 tile 的 $1/S$，总 block 数仍然可能和 SM 数对不齐。Stream-K 把所有 tile 的所有迭代排成一条队、平均分给常驻 CTA，解决「tile 数不是 SM 数的整数倍」，任何形状都能分到每个 SM 差不多一样多。论文里的一个例子：fixed-split 要切到 32 段才能和 Stream-K 一样均衡，但合并开销是 8 倍。
:::

::: details Q：decode 时 GEMV 已经是 memory-bound 了，为什么还要 split-K 提并行度？
memory-bound 不等于带宽打满。要打满 3.35 TB/s，同时在途的字节要够多（Little 定律），只有 64 个 block 时一半的 SM 没有发请求，HBM 带宽用不满。split-K 把 block 数提到 256，部分和只多 0.2% 的流量。
:::

::: details Q：大矩阵为什么不开 split-K？
tile 数已经远多于 SM 数，不缺并行度；而部分和的流量是 $2SMN \times 4$ B，$M = N = K = 4096$、$S = 2$ 时有 256 MB，比读 $A$、$B$ 的 64 MB 还多。
:::

::: details Q：Stream-K 里一个 tile 被两个 CTA 分着算，谁来写 C？会不会互相等很久？
算了这个 tile 第 0 轮迭代的 CTA 写。一个 tile 的开头在前一个 CTA 区间的末尾，后半段在下一个 CTA 区间的开头，所以后半段先算完、先交部分和，起头的 CTA 最后才碰这个 tile，到时部分和已经在了。只有每个 CTA 分不到一个 tile、一个 tile 跨 3 个以上 CTA 时才容易等，所以实际用 two-tile SK + DP。
:::

::: details Q：为什么开了 split-K 结果不能逐位复现？
atomic 合并时加法顺序取决于谁先算完。就算用固定顺序的 reduce，$S$ 也由 heuristic 按形状选，batch 一变 $S$ 可能变，加法顺序跟着变。浮点加法不满足结合律，最后几位就会不同。
:::

## 参考

- [Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU (arXiv:2301.03598)](https://arxiv.org/abs/2301.03598)：Stream-K、fixed-split 和 hybrid 方案的伪代码与实测
- [CUTLASS：Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)：parallel split-K（两个 kernel）和 sliced-K
- [Defeating Nondeterminism in LLM Inference (Thinking Machines, 2025)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)：split-K 和 batch-invariant matmul
- [MARLIN (arXiv:2408.11743)](https://arxiv.org/abs/2408.11743)：decode 时的 striped partitioning
