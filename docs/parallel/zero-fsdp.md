---
title: ZeRO 1 / 2 / 3 与 FSDP
status: draft
tags: [zero, fsdp]
difficulty: 3
order: 3
related: [/posttrain/training-memory, /parallel/collective-comm, /parallel/megatron-tp]
stack: []
---

# ZeRO 1 / 2 / 3 与 FSDP

> 通信-显存 tradeoff

## 一句话结论

普通数据并行里，N 张卡各存一份**一模一样**的模型状态，冗余了 N 倍。ZeRO 的思路是**去掉冗余**：每张卡只负责 $1/N$，用到别人那份时再临时通信要过来。ZeRO-1 / 2 / 3 依次把优化器状态、梯度、参数切开。前两级几乎不增加通信，第三级多 50%。FSDP 是 PyTorch 原生的 ZeRO-3。

## 从头讲起

### 第 0 步：训练时一张卡上到底存了什么

以 Adam + 混合精度为例，每个参数要存：

| 东西 | 精度 | 字节 | 干什么用 |
|---|---|---|---|
| 参数 | bf16 | 2 | forward / backward 计算用 |
| 梯度 | bf16 | 2 | backward 算出来，喂给优化器 |
| 主权重 | fp32 | 4 | 优化器更新用的高精度副本 |
| Adam 一阶动量 $m$ | fp32 | 4 | 优化器状态 |
| Adam 二阶动量 $v$ | fp32 | 4 | 优化器状态 |
| **合计** | | **16** | |

后三项（12 字节）统称**优化器状态**，占大头。一个 7B 模型光这些就是 $7 \times 16 = 112$ GB，一张 80 GB 的卡放不下，这还没算激活。详见 [训练显存账](/posttrain/training-memory)。

### 第 1 步：普通数据并行（DDP）怎么做

N 张卡，每张卡放一份**完整**的上表，各自吃不同的数据：

1. 每卡 forward + backward，得到**自己那批数据**的梯度。
2. 所有卡做一次 **all-reduce**，把梯度求平均，现在每张卡梯度都一样。
3. 每卡用同样的梯度、同样的优化器状态，做同样的更新，参数依然一致。

问题很明显：第 3 步是 **N 张卡在重复做完全相同的计算，存完全相同的东西**。8 张卡就存了 8 份一模一样的 112 GB。

### 第 2 步：关键观察：优化器更新是逐元素的

Adam 更新第 $i$ 个参数，只需要第 $i$ 个参数自己的梯度、$m$、$v$、主权重，**和其他参数无关**。

所以完全可以分工：把参数编号均分成 N 段，**卡 $k$ 只负责更新第 $k$ 段**，也只需要存第 $k$ 段的优化器状态。更新完再把各自那段新参数广播给大家。

这就是 ZeRO 的全部核心思想。三个阶段只是「切到多彻底」的区别。

### ZeRO-1：切优化器状态

- 每卡存：完整参数 (2) + 完整梯度 (2) + **$1/N$ 的优化器状态** (12/N)
- 流程：backward 后对梯度做 **reduce-scatter**（每卡只拿到自己那段的平均梯度）→ 更新自己那段 → **all-gather** 新参数

> 一次 all-reduce 本来就等于 reduce-scatter + all-gather（见 [集合通信](/parallel/collective-comm)）。ZeRO-1 只是把这两半拆开，**中间插进优化器更新**。通信量和 DDP 一样，显存白省。

### ZeRO-2：再切梯度

ZeRO-1 里，reduce-scatter 之后别人那段的梯度其实已经没用了，但还占着显存。ZeRO-2 直接把它丢掉：

- 每卡存：完整参数 (2) + **$1/N$ 梯度** (2/N) + **$1/N$ 优化器状态** (12/N)
- 通信量：依然和 DDP 一样。

### ZeRO-3：连参数也切

参数也只存 $1/N$。但 forward / backward 计算时**需要完整的层参数**，所以：

1. forward 到第 $\ell$ 层前：**all-gather** 拼出第 $\ell$ 层完整参数 → 算 → **立刻丢掉**
2. backward 到第 $\ell$ 层前：再 **all-gather** 一次 → 算梯度 → 丢掉参数
3. 梯度 **reduce-scatter**，每卡只留自己那段

- 每卡存：**全部 $16/N$**，外加「当前正在算的那一层」的完整参数（临时）
- 通信量：参数 all-gather 两次 + 梯度 reduce-scatter 一次 = $3\Psi$，DDP 是 $2\Psi$，所以**多 50%**

可以理解成「参数也按需借用，用完就还」。为了不让 GPU 干等通信，会在算第 $\ell$ 层时**提前 prefetch** 第 $\ell+1$ 层的参数。

### 算一笔：7B 模型，8 张卡（不含激活）

| | 每参数字节 | 每卡显存 | 通信量（相对 DDP） |
|---|---|---|---|
| DDP | $2 + 2 + 12 = 16$ | 112 GB | 1× |
| ZeRO-1 | $2 + 2 + 12/8 = 5.5$ | 38.5 GB | 1× |
| ZeRO-2 | $2 + (2+12)/8 = 3.75$ | 26.3 GB | 1× |
| ZeRO-3 | $16/8 = 2$ | 14 GB | 1.5× |

ZeRO-1 一步就省掉三分之二，因为优化器状态本来就占 12/16。

### FSDP

PyTorch 的 FSDP（Fully Sharded Data Parallel）就是 ZeRO-3 的原生实现。它以「wrap 单元」（通常是一个 Transformer block）为粒度做 all-gather / 丢弃。`ShardingStrategy.SHARD_GRAD_OP` 对应 ZeRO-2，`NO_SHARD` 退化为 DDP。**HSDP** 是折中：节点内做 ZeRO-3（走 NVLink，通信便宜），节点间做普通 DP（只同步梯度）。

### 怎么选

- 显存够：ZeRO-2，通信和 DDP 一样，没理由不开。
- 放不下：ZeRO-3 / FSDP，用 50% 额外通信换显存。
- 再大：ZeRO-3 和 TP、PP 组合，或用 HSDP 把 all-gather 限制在节点内。

## 面试追问

::: details Q：ZeRO-3 和 TP 都是把参数切到多卡，区别在哪？
ZeRO-3 切的是「存储」，计算时还要把整层参数 gather 回来，每张卡算完整的层；TP 切的是「计算」，每张卡只算自己那一片，用 all-reduce 合并结果。ZeRO-3 通信的是参数（和 batch 无关），TP 通信的是激活（和 batch 成正比），所以小 batch 用 TP 划算，大 batch 用 ZeRO 划算。
:::

::: details Q：ZeRO-1 / 2 为什么说通信量「不变」？
DDP 的梯度 all-reduce 在 ring 实现里本来就是 reduce-scatter + all-gather 两步，每步每卡收发约 $\Psi$。ZeRO-1/2 把它拆成「梯度 reduce-scatter」和「更新后参数 all-gather」，两步的大小和原来一样，只是中间插了优化器更新。
:::

::: details Q：ZeRO 能省激活显存吗？
不能。ZeRO 切的是模型状态（参数、梯度、优化器状态），激活和 batch、序列长度成正比，要靠 activation checkpointing、sequence parallel 或 CP 来省。
:::

## 手撕

常见题：给定模型参数量、卡数和优化器，算 ZeRO 各阶段每卡显存（照上面 7B 的表算）；画出 FSDP 一层的 all-gather → compute → 丢弃 → reduce-scatter 时序。

## 参考

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP 论文](https://arxiv.org/abs/2304.11277)
