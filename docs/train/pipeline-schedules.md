---
title: Pipeline 并行调度：GPipe、1F1B、Interleaved、Zero Bubble、DualPipe
status: draft
tags: [pipeline-parallel, 1f1b, interleaved, zero-bubble, dualpipe]
difficulty: 3
order: 3
related: [/parallel/parallelism-overview, /parallel/comm-overlap, /parallel/moe-ep, /posttrain/training-memory, /parallel/megatron-tp]
stack: []
train: [tr-p-pp]
---

# Pipeline 并行调度：GPipe、1F1B、Interleaved、Zero Bubble、DualPipe

> 同样是把层切到 $p$ 张卡上，调度顺序决定了 bubble 有多大、要存多少份激活、要传多少次

## 一句话结论

PP 把模型按层切成 $p$ 个 stage，一个 global batch 切成 $m$ 个 micro-batch 流过去。流水线开头要灌满、结尾要排空，每个 stage 都要空等 $(p-1)$ 个 micro-batch 的时间，这就是 bubble，占总时间 $\frac{p-1}{m+p-1}$。五种调度各解决一件事：

1. **GPipe**：先全部 forward 再全部 backward，bubble 如上，但要存 $m$ 份激活。
2. **1F1B**：bubble 不变，在途 micro-batch 不超过 $p$，激活从 $O(m)$ 降到 $O(p)$。
3. **Interleaved 1F1B**：每卡放 $v$ 段不连续的层，bubble 除以 $v$，P2P 通信乘以 $v$。
4. **Zero Bubble**：把 backward 拆成对输入（B）和对权重（W）两半，W 不在关键路径上，挪去填空；ZB-H1 显存不变、bubble 降到 1F1B 的 1/3。
5. **DualPipe**：两头同时灌数据，并把一个 forward chunk 和一个 backward chunk 配对，让一个的 all-to-all 和另一个的计算重叠；代价是每卡存两份参数。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $p$ | PP stage 数（一条流水线上的卡数） | 4 |
| $m$ | 一个 global batch 切出的 micro-batch 个数 | 8 |
| $t_f$ | 一个 micro-batch 在一个 stage 上的 forward 时间 | 1 |
| $t_b$ | backward 里对**输入**求梯度（B）的时间 | 1 |
| $t_w$ | backward 里对**权重**求梯度（W）的时间 | 1 |
| $v$ | interleaved 下每卡持有的 model chunk 数（virtual stage） | 2 |
| $M_a$ | 一个 micro-batch 在一个 stage 上要存的激活 | 见 [训练显存账](/posttrain/training-memory) |
| $b, s, h$ | micro-batch 大小（样本数）、序列长度、hidden size | 1, 8192, 8192 |

**为什么 $t_f \approx t_b \approx t_w$**：线性层 $Y = XW$ forward 一次矩阵乘；backward 两次同样大小的矩阵乘，$\partial X = \partial Y\, W^\top$（B，要传给上一个 stage）和 $\partial W = X^\top \partial Y$（W，只给优化器用）。这就是「每 token $6N$ FLOPs = forward 2 + backward 4」。[Zero Bubble 论文](https://arxiv.org/abs/2401.10241) Table 1 按矩阵乘算每层：F 为 $sbh(24h+4s)$，B 为 $sbh(24h+8s)$，W 为 $24sbh^2$，$s \ll h$ 时三者接近。下文取 $t_f = t_b = t_w = 1$，完整 backward 是 $t_b + t_w = 2$。

记一个 micro-batch 在一个 stage 上的全部计算为 $T = t_f + t_b + t_w$。

### 1. GPipe：bubble 从哪来

[GPipe](https://arxiv.org/abs/1811.06965) 的顺序：每个 stage 先把 $m$ 个 micro-batch 的 forward 全做完，再做全部 backward。$p=4$，$m=8$ 的时序（数字 = 第几个 micro-batch 的 forward，占 1 格；字母 a–h = 第 1–8 个的 backward，占 2 格；`.` = 空等）：

```
S0 12345678.........aabbccddeeffgghh
S1 .12345678......aabbccddeeffgghh..
S2 ..12345678...aabbccddeeffgghh....
S3 ...12345678aabbccddeeffgghh......
```

每个 stage 空 9 格：开头等前面 stage 的 forward 传过来 $(p-1)t_f$，结尾等后面 stage 的 backward 传回来 $(p-1)(t_b+t_w)$，合计 $(p-1)T = 3 \times 3 = 9$。总时间 $(m+p-1)T = 33$。

bubble 有两种写法，分母不同：

$$
\underbrace{\frac{(p-1)T}{(m+p-1)T} = \frac{p-1}{m+p-1}}_{\text{占总时间}}
\qquad
\underbrace{\frac{(p-1)T}{mT} = \frac{p-1}{m}}_{\text{相对理想时间（Megatron 论文的定义）}}
$$

记 $x = \frac{p-1}{m}$，则 $\frac{p-1}{m+p-1} = \frac{x}{1+x}$，$m \gg p$ 时两者相等。例子：$\frac{3}{11} = 27.3\%$ 的时间在空等；或者说比理想多花 $\frac{3}{8} = 37.5\%$ 的时间。面试时先说清用的是哪个分母。

降 bubble 只能加大 $m$：global batch 固定时，就是把 micro-batch 切得更小、更多，这会让 GEMM 变小、效率变低（见 [并行总览](/parallel/parallelism-overview)）。GPipe 作者的经验是 $m \ge 4p$ 时 bubble 可以忽略（作者自测，部分原因是重算可以提前排进空档）。

**显存**：backward 开始前，每个 stage 要同时存 $m$ 份激活，即 $m \cdot M_a$。GPipe 配合重算，只存每个 stage 的输入 $2bsh$ 字节，backward 时重算 stage 内部。

### 2. 1F1B（PipeDream-Flush）：bubble 不变，激活降到 $O(p)$

[Narayanan et al. 2021](https://arxiv.org/abs/2104.04473) 用的是 PipeDream-Flush：stage $i$（从 0 数）先做 $p-1-i$ 个 forward 预热，之后每做一个 forward 就做一个 backward（1F1B），最后把剩下的 backward 做完。

```
S0 1234......aa5bb6cc7dd8ee.ff.gg.hh
S1 .123....aa4bb5cc6dd7ee8ff.gg.hh..
S2 ..12..aa3bb4cc5dd6ee7ff8gg.hh....
S3 ...1aa2bb3cc4dd5ee6ff7gg8hh......
```

1. **bubble 一样**：总时间还是 33，每个 stage 还是空 9 格。灌满和排空的过程没变，只是空档挪了位置。
2. **在途 micro-batch 变少**：在途 = 已做 forward、还没做 backward、激活必须留着的 micro-batch。stage $i$ 最多 $p - i$ 个（上图 S0 到 S3 分别是 4、3、2、1），GPipe 是每个 stage 都 $m$ 个。峰值激活从 $m \cdot M_a$ 降到 $p \cdot M_a$。$p=8$、$m=64$ 时少 8 倍。

**为什么第一个 stage 显存压力最大**：

1. 它预热最多（$p-1$ 个），在途最多（$p$ 个）；最后一个 stage 只有 1 个。
2. embedding 表在它上面。
3. 有了 1F1B，$m$ 就可以放心调大来压 bubble，激活不再跟着 $m$ 涨。

Llama 3 的报告也指出 PP 的第一个 stage 因为 embedding 和预热的 micro-batch 占显存更多（[Llama 3 §3.3.2](https://arxiv.org/abs/2407.21783)）。

### 3. Interleaved 1F1B：bubble ÷ $v$，通信 × $v$

[Narayanan et al. 2021](https://arxiv.org/abs/2104.04473) §2.2.2：每张卡不再放一段连续的层，而是放 $v$ 段（model chunk）。论文的例子：4 张卡、每卡 2 个 chunk、共 16 层，卡 1 放第 1、2、9、10 层，卡 2 放第 3、4、11、12 层。一个 micro-batch 要绕流水线 $v$ 圈。

每个 chunk 的计算是原来的 $1/v$，开头灌满、结尾排空的时间也缩成 $1/v$：

$$
\text{bubble 时间} = \frac{(p-1)T}{v}, \qquad \text{相对理想} = \frac{1}{v} \cdot \frac{p-1}{m}
$$

例：$p=4$，$m=8$，$v=2$，从 $37.5\%$ 降到 $18.75\%$。

代价：

1. **通信 × $v$**：一个 micro-batch 跨过的 stage 边界从 $p-1$ 个变成 $pv-1$ 个，论文说通信量增加 $v$ 倍。论文用 scatter/gather 优化（见第 7 节）来抵消，作者自测吞吐最多提升 10%。
2. **对 $m$ 有要求**：论文的调度要求 $m$ 是 $p$ 的整数倍（4 张卡时 $m$ 必须是 4 的倍数）。现在的 Megatron-Core 放宽了：`--microbatch-group-size-per-virtual-pipeline-stage`（记为 $N$，默认 $p$）个连续 micro-batch 为一组轮流过各个 chunk，要求 $p \le N \le m$，且 $m \bmod N$ 为 0 或 $\ge p$（[model_parallel_config.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/model_parallel_config.py)、[schedules.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)）。Llama 3 也把这个 $N$ 做成了可调参数。
3. **激活略多**：Megatron-Core 里 rank 0 预热 $2(p-1) + (v-1)p$ 个 chunk 的 forward，加上稳态的 1 个，每个 chunk 的激活是 $M_a/v$，峰值约 $\left(p + \frac{p-1}{v}\right) M_a$。$p=4$、$v=2$ 时是 $5.5 M_a$，1F1B 是 $4 M_a$。

### 4. Zero Bubble：把 W 挪去填空

[Qi et al. 2023](https://arxiv.org/abs/2401.10241) 的观察：B 和 W 的依赖不一样。

1. **B** 算出 $\partial X$，上一个 stage 要等它才能开始自己的 backward，在关键路径上。
2. **W** 算出 $\partial W$，只有本 stage 的优化器要用，同一 micro-batch 的 B 做完以后什么时候做都行。

所以先做 B、把 $\partial X$ 发走，W 留着去填流水线里的空档。代价是 W 要用的 $X$ 和 $\partial Y$ 得多留一会儿，但 W 需要的激活比 B 少（论文 Table 1：B 要 $sb(34h + 5as)$ 字节，W 只要 $32sbh$）。

论文 Table 2（$M_B$ = 一次 B 要的激活）：

| 调度 | bubble 时间 | 峰值激活 |
|---|---|---|
| 1F1B | $(p-1)(t_f + t_b + t_w)$ | $p M_B$ |
| ZB-H1 | $(p-1)(t_f + t_b - t_w)$ | $p M_B$ |
| ZB-H2 | $(p-1)(t_f + t_b - 2t_w)$ | $(2p-1) M_B$ |

代入 $t_f = t_b = t_w = 1$：

1. **ZB-H1**：bubble 是 1F1B 的 $\frac{1+1-1}{1+1+1} = \frac{1}{3}$，峰值显存和 1F1B 一样。$p=4$、$m=8$ 时空 3 格，总时间 $8 \times 3 + 3 = 27$，bubble $3/27 = 11\%$（1F1B 是 27%）。
2. **ZB-H2**：bubble 为 0，激活接近 1F1B 的两倍（预热阶段多塞 forward 填满开头）。前提是去掉优化器 step 处的跨 stage 同步：梯度裁剪要全局梯度范数、混合精度要全局查 NaN/Inf，这些 all-reduce 会打破流水线。论文改成「先按部分归约的结果更新，下一轮再拿到全局值校验，不对就回滚重做」。

作者自测：1.5B 模型、$p=8$、$m=24$，bubble 占比 1F1B 24.3%、ZB-H1 15.9%、自动搜索的 ZB-2p 4.3%；显存相近时吞吐比 1F1B 最多高 23%，放宽显存最多高 31%。

### 5. DualPipe：双向流水 + 计算通信配对

[DeepSeek-V3 技术报告 §3.2.1](https://arxiv.org/abs/2412.19437)。背景：DeepSeek-V3 用 16 路 PP、跨 8 节点的 64 路 EP、ZeRO-1 DP，不用 TP；跨节点 EP 的 all-to-all 让计算 : 通信约为 1 : 1，通信不藏起来就白白浪费一半时间。

做法：

1. **chunk 拆四段**：attention、all-to-all dispatch、MLP、all-to-all combine。backward chunk 里的 attention 和 MLP 再像 Zero Bubble 一样拆成对输入、对权重两部分。另有 PP 通信。
2. **一对 forward / backward chunk 交错**：两个 chunk 来自不同 micro-batch，互不依赖。重排这些段，让一个 chunk 的 all-to-all 和 PP 通信在另一个 chunk 计算时进行，并手动调分给通信和计算的 SM 比例。作者称 all-to-all 和 PP 通信都能完全藏住。
3. **双向**：micro-batch 从流水线两端同时灌入。每个 rank 持有两个 stage（rank $i$ 放第 $i$ 和第 $p-1-i$ 个 stage），所以**参数存两份**。

论文 Table 2（$F$：一个 forward chunk；$B$：一个完整 backward chunk；$W$：一个对权重的 backward chunk；$F\&B$：一对互相重叠的 forward 和 backward chunk，均为时间）：

| 方法 | bubble | 参数 | 激活 |
|---|---|---|---|
| 1F1B | $(p-1)(F+B)$ | 1× | $p$ |
| ZB1P | $(p-1)(F+B-2W)$ | 1× | $p$ |
| DualPipe | $(\frac{p}{2}-1)(F\&B + B - 3W)$ | 2× | $p+1$ |

算一笔：$p=8$，$F=1$，$B=2$，$W=1$，假设通信完全藏住、$F\&B \approx F + B = 3$：1F1B $7 \times 3 = 21$，ZB1P $7 \times 1 = 7$，DualPipe $3 \times 2 = 6$。这个比较其实低估了 DualPipe：没有配对重叠时，1F1B 的 $F$、$B$ 里还含着暴露的 all-to-all，在 1 : 1 的通信比下会接近翻倍。

代价和约束（论文原文）：

1. 参数两份。DeepSeek-V3 的 EP 很大，每卡的专家参数少，所以总显存增加不多。
2. 激活峰值只多 $1/p$。
3. 只要求 stage 数和 micro-batch 数都能被 2 整除，不要求 $m$ 是 $p$ 的倍数；bubble 和激活都不随 $m$ 增长。

和 MoE 通信的关系见 [MoE 与 EP](/parallel/moe-ep) 和 [计算通信 Overlap](/parallel/comm-overlap)。

### 6. 对比

$M_a$ 为一个 micro-batch 在一个 stage 上的激活，激活一列是 stage 0 的峰值；P2P 次数是一个 micro-batch 在 forward 里跨过的 stage 边界数（backward 同样多）。

| 调度 | bubble 时间 | 激活峰值 | P2P 次数 | 实现复杂度 |
|---|---|---|---|---|
| GPipe | $(p-1)T$ | $m M_a$ | $p-1$ | 低 |
| 1F1B | $(p-1)T$ | $p M_a$ | $p-1$ | 低 |
| Interleaved | $(p-1)T / v$ | $\approx (p + \frac{p-1}{v}) M_a$ | $pv-1$ | 中：$m$ 有整除约束 |
| ZB-H1 | $(p-1)(t_f+t_b-t_w)$ | $p M_B$ | $p-1$ | 中：拆 B/W |
| ZB-H2 | $(p-1)(t_f+t_b-2t_w)$ | $(2p-1) M_B$ | $p-1$ | 高：去掉优化器同步、回滚 |
| DualPipe | $(\frac{p}{2}-1)(F\&B+B-3W)$ | $(p+1)$，参数 2× | 双向 | 高：拆 chunk、手调 SM |

### 7. 现实问题

**首尾 stage 不均衡**。按层数平均切时：

1. 第一个 stage 多了 embedding 表：$V \times h$ 个参数。Llama 3 8B（$V = 128{,}000$，$h = 4096$）是 5.2 亿，相当于 2.4 层的参数量（每层约 2.18 亿），按 16 字节/参数是 8.4 GB 模型状态，外加最多的在途激活。
2. 最后一个 stage 多了 LM head 和 loss：LM head 的计算等于一个 $V \times h$ 的线性层。70B（$h = 8192$，每层约 8.56 亿参数）的 LM head 有 10.5 亿参数，约 1.2 层的计算。80 层 $p=8$ 时最后一个 stage 是 $10 + 1.2 = 11.2$ 层的活，整条流水线按最慢的 stage 走，慢约 12%。

调法：给首尾 stage 少分层。Llama 3 405B 从第一个和最后一个 stage 各拿掉一层 Transformer，使第一个 stage 的第一个 chunk 只有 embedding、最后一个 stage 的最后一个 chunk 只有输出投影和 loss（[Llama 3 §3.3.2](https://arxiv.org/abs/2407.21783)）。可以这样理解：126 层加上 embedding、LM head 两个位置共 128 个，正好 16 个 rank 各 8 个。Megatron 里对应 `--decoder-first-pipeline-num-layers`、`--decoder-last-pipeline-num-layers`，或者用 `--pipeline-model-parallel-layout` 直接写每个 stage 放什么（[arguments.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py)）。

**PP 通信算一笔**。每个 micro-batch 每跨一个 stage 边界，forward 传一份激活、backward 传回一份同样大的梯度，各 $bsh$ 个元素（[Narayanan et al.](https://arxiv.org/abs/2104.04473) §3.2），bf16 下 $2bsh$ 字节。70B，$b=1$，$s=8192$，$h=8192$：

1. 一次传 $2 \times 8192 \times 8192 = 134$ MB，IB 每卡 50 GB/s 要 2.7 ms。
2. 这个 micro-batch 在一个 stage（10 层）上的 forward：$2 \times 10 \times 8.56\times10^8 \times 8192 = 1.4\times10^{14}$ FLOP，按 H100 bf16 dense 峰值 989 TFLOP/s、MFU 50% 算约 284 ms。
3. 通信不到计算的 1%，而且可以和其他 micro-batch 的计算重叠。PP 的问题是 bubble，不是通信量。

有 TP 时，相邻 stage 上 TP 组内的 8 张卡要发同一份激活。scatter/gather 优化让每张卡只发 $1/8$，收端再用 NVLink all-gather 拼回来，跨节点流量降为 $1/8$（[Narayanan et al.](https://arxiv.org/abs/2104.04473) §4.1）。

## 面试追问

::: details Q：1F1B 没有减少 bubble，为什么大家都用它？
bubble 只由灌满、排空决定，两种调度都是 $(p-1)T$。1F1B 的收益在显存：在途 micro-batch 从 $m$ 降到不超过 $p$，激活峰值和 $m$ 无关。于是 $m$ 可以调大，用 $\frac{p-1}{m+p-1}$ 压 bubble，而不用担心 OOM。
:::

::: details Q：PP 的 micro-batch 和 gradient accumulation 是什么关系？
同一回事。一个 global batch 切成 $m$ 个 micro-batch，各自 forward/backward，梯度在本 stage 上累加，$m$ 个全做完才做一次优化器 step（这就是 flush）。所以 global batch = $m \times b \times$ DP 度。想加 $m$ 又不改 global batch，就只能减小 $b$ 或减小 DP 度。
:::

::: details Q：interleaved 能把 $v$ 调得很大吗？
不能。通信次数 $pv - 1$ 随 $v$ 线性涨，跨节点时会暴露出来；每个 chunk 只有 $L/(pv)$ 层，层数要能整除；激活也略增。实际 $v$ 常取 2 到 4。Megatron 论文的实验里 batch 变大时 interleaved 的优势缩小，原因正是 bubble 本来就小了、而通信多了。
:::

::: details Q：Zero Bubble 为什么可以推迟 W，推迟的代价是什么？
W 算 $\partial W = X^\top \partial Y$，结果只给本 stage 的优化器用，没有别的 stage 等它；B 算 $\partial X$，上一个 stage 要等它。所以 B 先做、W 推后填空。代价是 $X$ 和 $\partial Y$ 要多留到 W 做完，但这比 B 要的激活少（$32sbh$ 对 $sb(34h+5as)$）。ZB-H1 控制 W 的起点，峰值显存和 1F1B 一样；ZB-H2 要把 bubble 压到 0，激活增加到约 $(2p-1) M_B$，还要去掉优化器 step 的跨 stage 同步。
:::

::: details Q：DualPipe 参数存两份，为什么还划算？
DeepSeek-V3 是 MoE，跨 8 节点做 64 路 EP，每张卡上的专家参数本来就少，两份也不多。换来的是：forward 和 backward chunk 配对，跨节点 all-to-all 和 PP 通信能藏在另一个 chunk 的计算后面（通信比约 1 : 1 时，这相当于省回近一半的时间），bubble 也比 1F1B、ZB1P 小。稠密模型没有这么大的通信，参数又多，DualPipe 的性价比就低得多。
:::

::: details Q：第一个 stage OOM，怎么办？
1. 给第一个 stage 少分几层（`--decoder-first-pipeline-num-layers`），抵掉 embedding 和更多的在途激活。
2. 只给前几个 stage 开 activation checkpointing，它们在途最多。
3. 减小 micro-batch 大小 $b$，靠加大 $m$ 维持 global batch。
4. 换 ZB-H1 这类不增加峰值的调度，别用 ZB-H2。
:::

## 手撕

写一个事件模拟器：给 $p$、$m$ 和调度类型，按依赖关系（$F$ 要等上一个 stage 的同一个 $F$，$B$ 要等下一个 stage 的同一个 $B$）排出每个 stage 的时间表，打印时序图，算 bubble 和每个 stage 的峰值在途数。

```python
def schedule(p, m, kind, tf=1, tbw=2):
    """每个 stage 的 [(op, mb, start, end)]；B 含对输入和对权重两部分，耗时 tbw。"""
    orders = []
    for s in range(p):
        if kind == "gpipe":
            order = [("F", i) for i in range(m)] + [("B", i) for i in range(m)]
        else:                                        # 1F1B：预热 p-1-s 个 forward
            warm = min(p - s - 1, m)
            order = [("F", i) for i in range(warm)]
            f, b = warm, 0
            while b < m:
                if f < m:
                    order.append(("F", f)); f += 1
                order.append(("B", b)); b += 1
        orders.append(order)

    done, free, pos = {}, [0] * p, [0] * p
    out = [[] for _ in range(p)]
    while any(pos[s] < len(orders[s]) for s in range(p)):
        for s in range(p):
            if pos[s] == len(orders[s]):
                continue
            op, i = orders[s][pos[s]]
            if op == "F":
                dep = ("F", s - 1, i) if s > 0 else None
            else:
                dep = ("B", s + 1, i) if s < p - 1 else ("F", s, i)
            if dep is not None and dep not in done:
                continue                             # 依赖没算完，先看别的 stage
            start = max(free[s], done.get(dep, 0))
            end = start + (tf if op == "F" else tbw)
            done[(op, s, i)] = free[s] = end
            out[s].append((op, i, start, end))
            pos[s] += 1
    return out

def draw(out):
    T = max(e for row in out for *_, e in row)
    for s, row in enumerate(out):
        cells = ["."] * T
        for op, i, st, en in row:
            for t in range(st, en):
                cells[t] = str(i + 1) if op == "F" else "abcdefghij"[i]
        print(f"S{s} " + "".join(cells))

def bubble(out):
    T = max(e for row in out for *_, e in row)
    busy = sum(e - st for row in out for *_, st, e in row)
    return 1 - busy / (T * len(out))

def peak_inflight(row):
    cur = best = 0
    for op, *_ in row:
        cur += 1 if op == "F" else -1
        best = max(best, cur)
    return best

for kind in ("gpipe", "1f1b"):
    out = schedule(4, 8, kind)
    draw(out)
    print(kind, round(bubble(out), 3), [peak_inflight(r) for r in out])
# gpipe 0.273 [8, 8, 8, 8]    （= 3/11，时序图同第 1 节）
# 1f1b  0.273 [4, 3, 2, 1]    （bubble 不变，在途数 p - s）
```

常见变体：把 B 拆成 `tb`、`tw` 两种 op 写 ZB-H1；加一个通信延迟参数，看 $m$ 小时 bubble 怎么变；用它验证 $\frac{p-1}{m+p-1}$。

## 参考

- [GPipe](https://arxiv.org/abs/1811.06965)（Huang et al., 2019）
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)（Narayanan et al., 2021）：PipeDream-Flush、interleaved、scatter/gather
- [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)（Qi et al., 2023）· [代码](https://github.com/sail-sg/zero-bubble-pipeline-parallelism)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) §3.2.1 · [DualPipe 代码](https://github.com/deepseek-ai/DualPipe)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) §3.3.2：PP 的显存与计算不均衡
- [Megatron-Core schedules.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)
