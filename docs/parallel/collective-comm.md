---
title: 集合通信原语与 NCCL
status: draft
tags: [nccl, all-reduce, nvlink]
difficulty: 3
order: 4
related: [/parallel/parallelism-overview, /parallel/zero-fsdp, /parallel/megatron-tp, /parallel/moe-ep, /parallel/comm-overlap]
stack: [d-comm, hw-link]
---

# 集合通信原语与 NCCL

> ring all-reduce 带宽公式；α-β 模型；busbw vs algbw；NVLink / IB 拓扑

## 一句话结论

分布式训练和推理里的通信，绝大部分都是几个固定的「集合通信原语」：all-reduce（DP 梯度、TP）、reduce-scatter / all-gather（ZeRO、SP）、all-to-all（MoE 的 EP）、P2P（PP、CP）。最重要的是 **ring all-reduce**：它拆成 reduce-scatter + all-gather 两个阶段，每个阶段 $N-1$ 步，每步每张卡只发 $1/N$ 的数据，所以每张卡总共发送 $\frac{2(N-1)}{N} S$（$S$ 是消息大小）。这个量**几乎和卡数无关**，并且已经是理论下界（带宽最优）。代价是步数 $2(N-1)$ 随 $N$ 线性增长，所以小消息、大规模时延迟占主导，这时 NCCL 改用 tree 算法。

## 推导

### 第 0 步：几个名词

- **rank**：参与通信的一个进程，通常一张 GPU 一个。$N$ 表示 rank 总数。
- **消息大小 $S$**：一次集合通信操作涉及的完整数组的字节数。比如 all-reduce 一份 256 MiB 的梯度，$S = 256$ MiB。
- **reduce**：把多份数据逐元素合并成一份，最常用的是求和（sum），也可以是 max / min / 求平均。
- **集合通信（collective）**：一组 rank **都**调用同一个操作，大家协作完成；与之相对的 **P2P** 只涉及一对 rank（send / recv）。
- **NCCL**：NVIDIA 的 GPU 集合通信库，PyTorch `torch.distributed` 的 `nccl` 后端就是它。它会自动探测拓扑，挑选算法（ring / tree 等）和协议，通信 kernel 直接跑在 GPU 上。

### 七个原语：4 张卡上的具体例子

设 4 张卡，每张卡上有一个长度 4 的向量：

- GPU0 = `[1, 2, 3, 4]`
- GPU1 = `[10, 20, 30, 40]`
- GPU2 = `[100, 200, 300, 400]`
- GPU3 = `[1000, 2000, 3000, 4000]`

逐元素求和是 `[1111, 2222, 3333, 4444]`。下面每个原语的语义都和 [NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) 一致。

| 原语 | 操作前 | 操作后 | 一句话 |
|---|---|---|---|
| **broadcast**（root=0） | 只有 GPU0 有 `[1,2,3,4]` | 4 张卡都是 `[1,2,3,4]` | 一份拷给所有人 |
| **reduce**（root=0，sum） | 各卡持有自己的向量 | 只有 GPU0 得到 `[1111,2222,3333,4444]` | 求和，结果只给 root |
| **all-reduce**（sum） | 各卡持有自己的向量 | 4 张卡**都**是 `[1111,2222,3333,4444]` | 求和，结果给所有人 |
| **reduce-scatter**（sum） | 各卡持有自己的向量 | GPU0 `[1111]`，GPU1 `[2222]`，GPU2 `[3333]`，GPU3 `[4444]` | 求和后切成 $N$ 块，第 $k$ 块给 rank $k$ |
| **all-gather** | GPU0 `[1111]`，GPU1 `[2222]`，GPU2 `[3333]`，GPU3 `[4444]` | 4 张卡都是 `[1111,2222,3333,4444]` | 每人一块，拼起来发给所有人 |
| **all-to-all** | GPU $k$ 的第 $j$ 个元素是要发给 GPU $j$ 的 | GPU $j$ 收到每张卡的第 $j$ 个元素：GPU0 `[1,10,100,1000]`，GPU1 `[2,20,200,2000]`… | 相当于把「卡 × 块」矩阵转置 |
| **P2P send/recv** | GPU0 有 `[1,2,3,4]` | GPU1 收到 `[1,2,3,4]` | 一对一，不是集合操作 |

注意上表中 reduce-scatter 的输出正好是 all-gather 的输入，两者接起来就是 all-reduce。这就是 [NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) 里写的：**all-reduce = reduce-scatter + all-gather**（也等于 reduce + broadcast）。[ZeRO](/parallel/zero-fsdp) 和 [Sequence Parallel](/parallel/megatron-tp) 正是利用这个等式，把 all-reduce 拆成两半，中间插入别的计算。

### 最朴素的 all-reduce 为什么慢

最直接的做法是 reduce + broadcast：所有人把数据发给 GPU0，GPU0 求和后再发回去。

- GPU0 要**接收** $(N-1)S$，再**发送** $(N-1)S$。
- 其他卡几乎闲着，只发 $S$、收 $S$。

GPU0 的链路成了瓶颈，耗时随 $N$ 线性增长：8 卡就是 7 倍于单份数据的传输时间。我们想要的是**让每张卡的链路都同样忙，且每张卡收发的数据尽量少**，ring all-reduce 就做到了这一点。

### ring all-reduce：一步步走

把 $N$ 张卡连成一个逻辑环：rank $r$ 只发给右邻居 $r+1$，只从左邻居 $r-1$ 收（下标都对 $N$ 取模）。每张卡把自己的数据切成 $N$ 块（chunk），编号 $0..N-1$。整个过程分两个阶段（[Baidu ring allreduce 讲解](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)）：

**阶段 1：reduce-scatter，$N-1$ 步。** 第 $s$ 步（$s = 0, 1, \dots, N-2$），rank $r$ 把自己的第 $(r-s) \bmod N$ 块发给右邻居，右邻居把它**加**到自己同编号的块上。

用上面 4 卡的例子（每块就是一个元素，**粗体**表示已经是 4 张卡之和的块）：

| | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|
| 初始 | [1, 2, 3, 4] | [10, 20, 30, 40] | [100, 200, 300, 400] | [1000, 2000, 3000, 4000] |
| 第 0 步：$r$ 发块 $r$ | [1, 2, 3, 4004] | [11, 20, 30, 40] | [100, 220, 300, 400] | [1000, 2000, 3300, 4000] |
| 第 1 步：$r$ 发块 $r-1$ | [1, 2, 3303, 4004] | [11, 20, 30, 4044] | [111, 220, 300, 400] | [1000, 2220, 3300, 4000] |
| 第 2 步：$r$ 发块 $r-2$ | [1, **2222**, 3303, 4004] | [11, 20, **3333**, 4044] | [111, 220, 300, **4444**] | [**1111**, 2220, 3300, 4000] |

以第 0 步为例：GPU0 把块 0（值 1）发给 GPU1，GPU1 的块 0 变成 $10+1=11$；同时 GPU3 把块 3（值 4000）发给 GPU0，GPU0 的块 3 变成 $4+4000=4004$。每一块沿环走，每经过一张卡就累加一次，走 $N-1$ 步后恰好累加了全部 $N$ 份。阶段 1 结束时，**rank $r$ 持有完整求和的第 $(r+1) \bmod N$ 块**，这正是一次 reduce-scatter（只是块编号相对 rank 错开了一位）。

**阶段 2：all-gather，$N-1$ 步。** 第 $s$ 步，rank $r$ 把第 $(r+1-s) \bmod N$ 块发给右邻居，右邻居**覆盖**（不再相加）自己同编号的块。每个完整块沿环转一圈，$N-1$ 步后所有卡都拿到全部 $N$ 块。

| | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|
| 阶段 1 结束 | [1, **2222**, 3303, 4004] | [11, 20, **3333**, 4044] | [111, 220, 300, **4444**] | [**1111**, 2220, 3300, 4000] |
| 第 0 步 | [**1111**, **2222**, 3303, 4004] | [11, **2222**, **3333**, 4044] | [111, 220, **3333**, **4444**] | [**1111**, 2220, 3300, **4444**] |
| 第 1 步 | [**1111**, **2222**, 3303, **4444**] | [**1111**, **2222**, **3333**, 4044] | [111, **2222**, **3333**, **4444**] | [**1111**, 2220, **3333**, **4444**] |
| 第 2 步 | 全部 [1111, 2222, 3333, 4444] | 同左 | 同左 | 同左 |

### 通信量推导：$\frac{2(N-1)}{N} S$

数一下每张卡发了多少：

- 每步发 1 块，每块大小 $S/N$。
- 阶段 1 有 $N-1$ 步，阶段 2 有 $N-1$ 步，共 $2(N-1)$ 步。
- 每张卡总发送量（接收量相同）：

$$
2(N-1) \cdot \frac{S}{N} = \frac{2(N-1)}{N} S
$$

代入数字：$N=2$ 时是 $1 \times S$，$N=8$ 时是 $1.75 S$，$N \to \infty$ 时趋近 $2S$。所以**卡数从 8 加到 1024，每张卡要传的数据只多了约 14%**，这就是「ring all-reduce 通信量与卡数几乎无关」的意思。

而且这是下界：[Patarasuk & Yuan (2009)](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf) 证明任何 all-reduce 算法中每个进程至少要发送和接收 $\frac{2(N-1)}{N}S$ 的数据，ring 恰好达到，所以称为**带宽最优**。直观理解：结果的每一块都需要另外 $N-1$ 份贡献，reduce-scatter 阶段每卡至少收 $\frac{N-1}{N}S$；之后每卡缺另外 $N-1$ 块结果，all-gather 阶段至少再收 $\frac{N-1}{N}S$。

同理可得其他原语每卡的发送量（ring 实现）：reduce-scatter 和 all-gather 各是 $\frac{N-1}{N}S$（正好是 all-reduce 的一半），broadcast 和 reduce 是 $S$（流水线式沿链传递）。all-to-all 每卡要把 $\frac{N-1}{N}$ 的本地数据发给别人，它不走环，而是两两直接发送。

### α-β 模型：带宽项与延迟项

估算通信时间的标准模型：发送一条 $n$ 字节的消息耗时

$$
T = \alpha + n\beta
$$

- $\alpha$：**延迟**，每发一次消息的固定开销（kernel 启动、同步、网络往返），和大小无关，量级是微秒。
- $\beta = 1/B$：每字节的传输时间，$B$ 是**单方向**链路带宽。

ring all-reduce 有 $2(N-1)$ 步，每步发 $S/N$，所以：

$$
T_{\text{ring}} = \underbrace{2(N-1)\,\alpha}_{\text{延迟项，随 } N \text{ 线性增长}} + \underbrace{\frac{2(N-1)}{N}\, S \beta}_{\text{带宽项，几乎不随 } N \text{ 变}}
$$

两项哪个占主导，取决于消息大小：

- **大消息**（MiB 级以上，比如 DP 梯度、TP 激活）：带宽项主导，ring 最优。
- **小消息**（KiB 级，比如推理 decode 阶段 TP 的 all-reduce，每次只有 batch × hidden 个元素）或 **$N$ 很大**（上千卡）：延迟项主导，$2(N-1)\alpha$ 很难看。

[Thakur, Rabenseifner & Gropp (2005)](https://web.cels.anl.gov/~thakur/papers/mpi-coll.pdf) 的结论就是：没有一个算法通吃，要按消息大小和进程数选算法。常见 all-reduce 算法对比（$N$ 取 2 的幂）：

| 算法 | 延迟项 | 带宽项 | 适合 |
|---|---|---|---|
| ring | $2(N-1)\alpha$ | $\frac{2(N-1)}{N} S\beta$ | 大消息 |
| recursive doubling（两两交换，$\log_2 N$ 轮） | $\log_2 N \cdot \alpha$ | $\log_2 N \cdot S\beta$ | 小消息 |
| Rabenseifner（recursive halving 做 RS + recursive doubling 做 AG） | $2\log_2 N \cdot \alpha$ | $\frac{2(N-1)}{N} S\beta$ | 中大消息 |
| double binary tree（NCCL） | $O(\log N)\,\alpha$ | 接近满带宽 | 小 / 中消息、大规模 |

**double binary tree** 是 NCCL 2.4 引入的（[NVIDIA 博客](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)）。单棵二叉树做 reduce + broadcast 的延迟是 $O(\log N)$，但叶子节点只发不收、内部节点要收两个孩子的数据，带宽用不满。观察到二叉树里约一半节点是叶子，于是再建第二棵树，把第一棵的叶子当内部节点，反之亦然；数据分两半，各走一棵树。这样每个 rank 在两棵树里的负载加起来是均衡的，带宽接近满，同时保留对数延迟。博客指出 ring 的延迟随 GPU 数线性增长，限制了扩展到几百卡以上。

NCCL 的算法可以用 `NCCL_ALGO` 指定，取值包括 `Ring`、`Tree`、`CollnetDirect` / `CollnetChain`（交换机内归约，如 IB SHARP）、`NVLS` / `NVLSTree`（NVSwitch 内归约）、`PAT` 等；协议用 `NCCL_PROTO` 指定 `LL` / `LL128` / `Simple`（见 [NCCL 环境变量文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)）。默认是自动选，`NCCL_DEBUG=INFO` 可以看它选了什么。协议的细节可参考 [Demystifying NCCL](https://arxiv.org/abs/2507.04786)。

### 算一算：8 卡 NVLink 上 all-reduce 256 MiB

这正是 [并行总览](/parallel/parallelism-overview) 中 70B、batch 8、2k token、bf16 时一次 TP all-reduce 的大小。

**先搞清 900 GB/s 是什么。** H100 SXM 的 NVLink 标称 900 GB/s，[NVIDIA 明确说明](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) 这是**双向**带宽，即每个方向约 450 GB/s。ring 里每张卡同时向右发、从左收，两个方向各跑一份流量，所以 α-β 公式里的 $B$ 要取**单方向**的 450 GB/s。

- 每卡发送量：$\frac{2 \times 7}{8} \times 256 \text{ MiB} = 448 \text{ MiB} \approx 470 \text{ MB}$
- 带宽项：$470 \text{ MB} / 450 \text{ GB/s} \approx 1.04$ ms
- 延迟项：$2 \times 7 = 14$ 步，假设 $\alpha \approx 5\,\mu s$（仅为说明量级的假设值），约 $0.07$ ms，可以忽略

所以理论下限约 **1 ms**。[并行总览](/parallel/parallelism-overview) 里「约 0.5 ms」是直接拿 900 GB/s 除，算的是乐观下界，实际要按单方向带宽算。真实测出来还会更慢一些，因为链路利用率达不到 100%。

**换成小消息**：decode 时 batch 64、hidden 8192、bf16，一次 all-reduce 只有 1 MiB。带宽项 $\frac{2\times7}{8} \times 1\text{ MiB} / 450 \text{ GB/s} \approx 4\,\mu s$，和 14 步的延迟项 $70\,\mu s$ 相比，延迟反而占大头。这就是推理里 TP all-reduce 要用 tree、NVLS、或者自定义 one-shot all-reduce kernel 的原因。

**通用公式**：给定 $N$、$S$、单方向带宽 $B$ 和 $\alpha$：

$$
T \approx 2(N-1)\alpha + \frac{2(N-1)}{N} \cdot \frac{S}{B}
$$

### busbw 与 algbw：读懂 nccl-tests

[nccl-tests](https://github.com/NVIDIA/nccl-tests) 是测集合通信性能的标准工具，它报两个带宽，定义见 [PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)：

- **algbw（algorithm bandwidth）** $= S / t$：消息大小除以耗时，就是「用户视角」的速度。
- **busbw（bus bandwidth）** $= \text{algbw} \times \text{因子}$：把算法本身的数据放大量乘回去，反映每张卡链路上实际跑出来的速度。

| 原语 | busbw 因子 |
|---|---|
| all-reduce | $\frac{2(N-1)}{N}$ |
| reduce-scatter | $\frac{N-1}{N}$ |
| all-gather | $\frac{N-1}{N}$ |
| broadcast | $1$ |
| reduce | $1$ |
| all-to-all | $\frac{N-1}{N}$ |

注意 nccl-tests 里 all-gather / reduce-scatter 的 $S$ 指的是**完整数组**大小（每卡份数 × $N$），不是每卡那一块（[PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)）。

**为什么要有 busbw**：algbw 和 $N$ 有关。同样的硬件，2 卡 all-reduce 的 algbw 等于链路带宽，8 卡只有 $1/1.75$。乘上因子之后，busbw 和 $N$ 无关，可以**直接和硬件单方向峰值带宽比**，看链路用满了没有。

套用上面的例子：8 卡 all-reduce 256 MiB 用时 1.04 ms，algbw $= 268 \text{ MB} / 1.04 \text{ ms} \approx 258$ GB/s，busbw $= 258 \times 1.75 \approx 450$ GB/s，正好等于单方向 NVLink 带宽。

一个反常现象：开启 NVLS 后，all-reduce 的 busbw 可能**超过**单方向链路带宽。原因是 NVSwitch 内带 SHARP 归约引擎（[NVIDIA NVLink 页面](https://www.nvidia.com/en-us/data-center/nvlink/)），每张卡只需把 $S$ 发给交换机、再收回 $S$，实际流量比 ring 的 $\frac{2(N-1)}{N}S$ 小，但 busbw 仍按 ring 的因子换算。

### 拓扑：节点内 NVLink，跨节点 IB

| 链路 | 典型带宽 | 用在 |
|---|---|---|
| NVLink 4（H100 SXM） | 900 GB/s 双向，即单方向 450 GB/s（[NVIDIA](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)） | 节点内 GPU 之间，通过 NVSwitch 全互联 |
| PCIe Gen5 x16 | 128 GB/s 双向（[NVIDIA H100 规格](https://www.nvidia.com/en-us/data-center/h100/)） | GPU 与 CPU / 网卡 |
| IB NDR | 400 Gb/s ≈ 50 GB/s 单方向 / 网卡 | 跨节点，通常每 GPU 配一张网卡 |

节点内带宽比跨节点高约一个数量级（450 vs 50 GB/s）。所以**通信量大、在关键路径上的 TP 放在节点内**，跨节点放 PP / DP（见 [并行总览](/parallel/parallelism-overview)）。

**hierarchical all-reduce**：设 $M$ 个节点、每节点 $G$ 张卡，all-reduce 大小 $S$ 的数据：

1. **节点内 reduce-scatter**（NVLink）：每张卡得到本节点求和后的 $S/G$ 一块。
2. **跨节点 all-reduce**（IB）：每个节点里的第 $g$ 号卡和其他节点的第 $g$ 号卡组成一个环，只 all-reduce 这 $S/G$。$G$ 个环并行，各走自己的网卡。
3. **节点内 all-gather**（NVLink）：把 $G$ 块拼回完整结果。

慢速的 IB 上每卡只需传 $\frac{2(M-1)}{M} \cdot \frac{S}{G}$，即只负责 $1/G$ 的数据；大头的 $\frac{2(G-1)}{G}S$ 留在快速的 NVLink 上。

算一笔：8 节点 × 8 卡，梯度 $S = 1$ GiB。

- 节点内 RS + AG：每卡发 $2 \times \frac{7}{8} \times 1\text{ GiB} \approx 1.88$ GB，按 450 GB/s 约 4.2 ms。
- 跨节点：每卡发 $\frac{2 \times 7}{8} \times 128 \text{ MiB} \approx 235$ MB，按 50 GB/s 约 4.7 ms。

两段量级相当，没有哪一段严重拖后腿。实践中 NCCL 自己构造的 ring / tree 就是拓扑感知的：ring 在节点内走 NVLink，跨节点只跳一次；tree 在节点间建树、节点内走链（[NVIDIA 博客](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)）。上面的三步分解主要用于理解和估算。FSDP 的 HSDP 也是同样的思路（见 [ZeRO / FSDP](/parallel/zero-fsdp)）。

### 每种并行用哪个原语

| 并行 | 原语 | 什么时候、传什么 | 详见 |
|---|---|---|---|
| DP / DDP | all-reduce | 每 step 一次，梯度（大小 = 参数量 × 字节数）；初始化时 broadcast 参数 | [ZeRO / FSDP](/parallel/zero-fsdp) |
| ZeRO-1 / 2 | reduce-scatter + all-gather | 梯度 RS → 更新自己那段 → 参数 AG，总量同 DDP | [ZeRO / FSDP](/parallel/zero-fsdp) |
| ZeRO-3 / FSDP | all-gather ×2 + reduce-scatter | 每层前向、反向各 AG 一次参数，梯度 RS，总量是 DDP 的 1.5 倍 | [ZeRO / FSDP](/parallel/zero-fsdp) |
| TP | all-reduce | 每层前向 2 次、反向 2 次，激活 $B S d \cdot b$ | [Megatron TP](/parallel/megatron-tp) |
| SP | reduce-scatter + all-gather | 替换 TP 的 all-reduce，总量不变 | [Megatron TP](/parallel/megatron-tp) |
| EP | all-to-all | 每个 MoE 层 dispatch + combine 两次 | [MoE / EP](/parallel/moe-ep) |
| PP | P2P send / recv | stage 边界传激活（反向传梯度） | [并行总览](/parallel/parallelism-overview) |
| CP | 环形 P2P | attention 内环形轮转 K/V | [并行总览](/parallel/parallelism-overview) |

怎么让这些通信和计算重叠，见 [通信计算重叠](/parallel/comm-overlap)。

## 面试追问

::: details Q：ring all-reduce 每卡通信量是多少？为什么说和卡数无关？
每卡发送 $\frac{2(N-1)}{N}S$，接收同样多。两阶段各 $N-1$ 步，每步发 $S/N$。$N=8$ 时是 $1.75S$，$N\to\infty$ 时趋近 $2S$，所以加卡几乎不增加每卡的数据量，总耗时的带宽项基本不变。但步数 $2(N-1)$ 线性增长，延迟项会变大。
:::

::: details Q：为什么不直接 reduce 到一张卡再 broadcast？
root 要收 $(N-1)S$、再发 $(N-1)S$，它的链路是瓶颈，其他卡的链路闲着，耗时随 $N$ 线性增长。ring 让每张卡同时收发，每张卡链路都在工作，且每卡流量只有 $\frac{2(N-1)}{N}S$，已经达到下界。
:::

::: details Q：ring 既然带宽最优，为什么 NCCL 还要 tree？
α-β 模型里 ring 的延迟项是 $2(N-1)\alpha$。小消息时带宽项很小，延迟项主导；上千卡时延迟项本身就很大。double binary tree 的延迟是 $O(\log N)$，同时用两棵互补的树把带宽补回到接近满。NCCL 会按消息大小和拓扑自动在两者之间选。
:::

::: details Q：nccl-tests 的 algbw 和 busbw 有什么区别？拿哪个和硬件带宽比？
algbw $= S/t$，busbw $=$ algbw × 原语因子（all-reduce 是 $\frac{2(N-1)}{N}$）。busbw 与 $N$ 无关，反映每卡链路实际跑出的速度，所以拿 **busbw** 和硬件单方向峰值带宽比。algbw 用来估算「我这个大小的消息要多久」。
:::

::: details Q：H100 NVLink 900 GB/s，all-reduce 256 MiB 要多久？
900 GB/s 是双向，单方向 450 GB/s。8 卡 ring：每卡发 $1.75 \times 256$ MiB $\approx 470$ MB，$470 / 450 \approx 1.04$ ms，这是理论下限。直接用 900 算会得到 0.5 ms，低估一半。
:::

::: details Q：all-to-all 和 all-reduce 的流量有什么不同？为什么 EP 跨节点更难？
all-reduce 可以组织成环，每卡只和两个邻居通信，流量规则。all-to-all 每卡要和所有其他卡直接交换数据，每卡发出 $\frac{N-1}{N}$ 的本地数据，跨节点时大部分流量要过 IB，受限于网络的二分带宽；而且每对之间发多少由路由结果决定，负载不均时会出现热点。见 [MoE / EP](/parallel/moe-ep)。
:::

::: details Q：为什么 ZeRO-1/2 的通信量和 DDP 一样？
DDP 的梯度 all-reduce 在 ring 里本来就是 reduce-scatter + all-gather，各 $\frac{N-1}{N}\Psi$。ZeRO-1/2 只是把两半拆开，中间插入优化器更新，前一半作用在梯度上，后一半作用在更新后的参数上，大小不变。
:::

## 手撕

### ring all-reduce 模拟

用 Python 模拟 $N$ 个 rank，每个 rank 是一个列表。同一步里所有 rank「同时」发送，所以先把要发的内容拷出来再写入。

```python
def ring_all_reduce(data):
    """data[r]: rank r 的输入，长度可被 N 整除。原地变成所有 rank 的逐元素和。"""
    N = len(data)
    L = len(data[0]) // N                          # 每块长度 = S / N
    def chunk(r, c):
        return data[r][c * L:(c + 1) * L]

    # 阶段 1：reduce-scatter，N-1 步
    for s in range(N - 1):
        msgs = [(r, (r - s) % N, list(chunk(r, (r - s) % N))) for r in range(N)]
        for r, c, payload in msgs:                 # rank r 发块 c 给右邻居
            dst = (r + 1) % N
            for i, v in enumerate(payload):
                data[dst][c * L + i] += v          # 累加
    # 此时 rank r 持有完整求和的第 (r+1) % N 块

    # 阶段 2：all-gather，N-1 步
    for s in range(N - 1):
        msgs = [(r, (r + 1 - s) % N, list(chunk(r, (r + 1 - s) % N))) for r in range(N)]
        for r, c, payload in msgs:
            dst = (r + 1) % N
            data[dst][c * L:(c + 1) * L] = payload  # 覆盖

data = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400], [1000, 2000, 3000, 4000]]
ring_all_reduce(data)
# 4 个 rank 都是 [1111, 2222, 3333, 4444]
# 每个 rank 发送 2(N-1) 次、每次 L 个元素，共 2(N-1)/N × 总长度
```

要点：

- 每步每个 rank 只发 1 块、只收 1 块，所有链路同时工作。
- 阶段 1 是 `+=`（归约），阶段 2 是 `=`（覆盖）。
- 发送的块编号 $(r-s) \bmod N$ 保证同一步里没有两个 rank 发同一块，每块沿环走 $N-1$ 步后恰好累加了所有 rank 的贡献。
- 实际实现（NCCL）还会把每块再切成小片做流水线，并开多个环（channel）并行，充分用满多条链路。

### α-β 估算

```python
def ring_allreduce_time(S_bytes, N, B_unidir, alpha):
    return 2 * (N - 1) * alpha + 2 * (N - 1) / N * S_bytes / B_unidir

ring_allreduce_time(256 * 2**20, 8, 450e9, 5e-6)   # ≈ 1.11e-3 s（带宽项 1.04 ms + 延迟项 0.07 ms）
ring_allreduce_time(1 * 2**20,   8, 450e9, 5e-6)   # ≈ 7.4e-5 s，延迟项占九成以上
```

常见题：推导 $\frac{2(N-1)}{N}$；给定模型和互联估 TP 一层的通信时间；解释 nccl-tests 输出的 busbw；画 4 卡 ring all-reduce 每步每卡持有什么。

## 参考

- [NCCL 文档：Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [NCCL 文档：Environment Variables（NCCL_ALGO / NCCL_PROTO / NCCL_DEBUG）](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [nccl-tests PERFORMANCE.md（algbw / busbw 定义）](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- [Bringing HPC Techniques to Deep Learning（Baidu ring allreduce 讲解）](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)
- [Patarasuk & Yuan, Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations, JPDC 2009](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf)
- [Thakur, Rabenseifner & Gropp, Optimization of Collective Communication Operations in MPICH, IJHPCA 2005](https://web.cels.anl.gov/~thakur/papers/mpi-coll.pdf)
- [Massively Scale Your Deep Learning Training with NCCL 2.4（double binary tree）](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)
- [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](https://arxiv.org/abs/2507.04786)
- [NVIDIA Hopper 架构（NVLink 900 GB/s 双向）](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
- [NVIDIA NVLink / NVSwitch（SHARP 归约引擎）](https://www.nvidia.com/en-us/data-center/nvlink/)
