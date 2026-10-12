---
title: 并行配置怎么选：从显存约束到 MFU 调优
status: draft
tags: [parallelism, tp, pp, cp, dp, mfu]
difficulty: 3
order: 4
related: [/posttrain/training-memory, /train/compute-mfu, /train/pipeline-schedules, /train/long-context, /parallel/zero-fsdp, /parallel/comm-overlap, /gpu/profiling]
stack: []
---

# 并行配置怎么选：从显存约束到 MFU 调优

> 给定模型、序列长度、卡数和 global batch，怎么定 TP / PP / CP / DP；配完以后 MFU 低，先动哪里

## 一句话结论

三条约束一起掐：**显存**（每卡的模型状态加激活要放下）、**拓扑**（TP、CP、EP 的通信在每层的关键路径上，只能放在 8 卡的 NVLink 域内；PP、DP 跨节点）、**global batch**（收敛决定 $B$ 的上限，$B = d \cdot b \cdot k$，PP 又要每步有足够多的 micro-batch）。决策顺序：TP 填满 NVLink → 显存不够加 PP → 序列太长加 CP → 剩下的卡给 DP，配 ZeRO-1。例：70B、seq 8k、1024 张 H100、global batch 4M token，得到 TP 8 × PP 2（interleaved $v = 5$）× DP 64，每卡约 43 GB，bubble 2.5%。配完 MFU 低，先 profile 一个 step 看时间去哪了，按 micro-batch → recompute → overlap → kernel 融合的顺序调，最后才改并行配置。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值（第 5 节） |
|---|---|---|
| $t$ | TP 度（tensor parallel） | 8 |
| $p$ | PP 度（pipeline stage 数） | 2 |
| $v$ | interleaved PP 下每张卡上的模型段数（virtual stage） | 5 |
| $c$ | CP 度（context parallel，按序列切） | 1 |
| $d$ | DP 度（data parallel） | 64 |
| $e$ | EP 度（MoE 的专家并行） | 不用 |
| $b$ | micro-batch size：一次 forward 的序列条数 | 1 |
| $k$ | gradient accumulation 步数，也就是每步每条 pipeline 跑的 micro-batch 个数 $m$ | 8 |
| $B$ | global batch：每次参数更新用的序列条数 | 512 |
| $s$ | 序列长度 | 8192 |
| $n_{\text{gpu}}$ | 总卡数，$= t \cdot p \cdot c \cdot d$ | 1024 |
| $N, L, h$ | 参数量、层数、hidden size | 70 B、80、8192 |

EP 不在 $n_{\text{gpu}}$ 的乘积里：MoE 层的专家是在 DP（和 CP）那些卡里再切一次，见第 6 节的 DeepSeek-V3。单位：GB = $10^9$ 字节；H100 每卡 80 GB HBM，每节点 8 卡用 NVLink 连。

### 1. 约束一：显存

推导在 [训练显存账](/posttrain/training-memory)，这里只用结论：

1. **模型状态**：每卡分到 $N/(tp)$ 个参数。DP 上开 ZeRO-1，每参数 $4 + 12/d$ 字节（bf16 参数 2 + bf16 梯度 2，fp32 主权重和 Adam 两个动量共 12 字节在 $d$ 张卡上切开）。
2. **激活**：FlashAttention、开 sequence parallel、不 recompute 时每层 $34sbh/t$ 字节；CP 把序列切成 $c$ 段，再除以 $c$。
3. **PP 不省第一个 stage 的激活**：1F1B 下第一个 stage 要同时存 $p$ 个 micro-batch 的激活，每个 $L/p$ 层，合起来还是 $L$ 层（[Korthikanti et al. 2022](https://arxiv.org/abs/2205.05198) 第 4.2.3 节）。interleaved 再乘 $1 + \frac{p-1}{pv}$。

$$
M_{\text{gpu}} \approx \underbrace{\frac{N}{tp}\left(4 + \frac{12}{d}\right)}_{\text{模型状态}} + \underbrace{\frac{34\,sbhL}{tc}\left(1 + \frac{p-1}{pv}\right)}_{\text{第一个 stage 的激活}}
$$

所以：**PP 和 ZeRO 只切模型状态，TP 和 CP 才切激活。**

例：70B、$t = 8$、$p = 1$、$d = 128$、$s = 8192$、$b = 1$：

1. 模型状态：$7\times10^{10}/8 = 8.75\times10^9$ 个参数 $\times (4 + 12/128) = 35.8$ GB。
2. 激活：每层 $34 \times 8192 \times 8192 / 8 = 0.285$ GB，80 层 22.8 GB。
3. 合计 58.6 GB。

公式没算 CUDA context、NCCL buffer、logits、allocator 碎片（见训练显存账的「临时 buffer」一节）。本页假设给它们留 20 GB，也就是公式值要 ≤ 60 GB，实际余量以 memory snapshot 为准。

### 2. 约束二：通信拓扑

| 维度 | 通信 | 多频繁 | 能不能藏 | 放哪 |
|---|---|---|---|---|
| TP | 激活的 all-reduce（开 SP 后是 reduce-scatter + all-gather） | 每层 forward 2 次、backward 2 次 | 在关键路径上，下一步要等它 | NVLink 域内 |
| CP | 交换 K、V | 每层 attention | 和 attention 计算重叠 | 尽量 NVLink 域内 |
| EP | dispatch / combine 的 all-to-all | 每个 MoE 层 2 次 | 要靠双 micro-batch 交错 | NVLink 或高带宽 IB |
| PP | stage 边界的 P2P | 每个 micro-batch 1 次 | 和别的 micro-batch 重叠 | 可以跨节点 |
| DP | 梯度 reduce-scatter、参数 all-gather | 每步 1 次 | 和 backward 重叠 | 跨节点，最外层 |

带宽差多少：H100 的 NVLink 单向约 450 GB/s，跨节点每卡一张 400 Gb/s 网卡约 50 GB/s，差 9 倍（算法见 [并行总览](/parallel/parallelism-overview)、[集合通信](/parallel/collective-comm)）。

**device mesh 的维度顺序**：把卡号排成多维数组，**最内层（相邻卡号）给通信最频繁的维度**。相邻的 8 个卡号在同一台机器上，所以 TP 放最内层：

$$
\text{rank} = \big((i_{\text{dp}} \cdot p + i_{\text{pp}}) \cdot c + i_{\text{cp}}\big) \cdot t + i_{\text{tp}}
$$

Llama 3 用的就是 [TP, CP, PP, DP] 从内到外的顺序：最内层要最高带宽、最低延迟，限制在同一台机器；DP（他们用 FSDP）放最外层，因为参数预取和梯度 reduce 可以异步做，能容忍多跳网络的延迟（[arXiv 2407.21783](https://arxiv.org/abs/2407.21783) 第 3.3.2 节）。

例：$t = 8, p = 2, c = 1, d = 64$。rank 21：$21 = 2 \times 8 + 5$，$i_{\text{tp}} = 5$；$2 = 1 \times 2 + 0$，$i_{\text{dp}} = 1, i_{\text{pp}} = 0$。它的 TP 组是 rank 16–23，正好是第 3 台机器的 8 张卡；PP 组是 rank 21 和 29，在相邻两台机器上；DP 组是 rank 5, 21, 37, ...，跨 64 台机器。PyTorch 里写成 `init_device_mesh("cuda", (64, 2, 8), mesh_dim_names=("dp", "pp", "tp"))`，最后一维是最内层。

### 3. 约束三：global batch

$B$ 由收敛决定，不由系统决定：batch 超过 critical batch size 后，再加大 batch，达到同样 loss 需要的 token 数明显变多（[McCandlish et al. 2018](https://arxiv.org/abs/1812.06162)）；训练早期用小 batch 也更稳，所以常常从小 batch 开始逐步加大。Llama 3 405B 从 4M token 加到 8M、16M（第 3.4.1 节）；DeepSeek-V3 从 3072 条加到 15360 条（序列 4K，[arXiv 2412.19437](https://arxiv.org/abs/2412.19437) 第 4.2 节）。

系统这边要满足：

$$
B = d \cdot b \cdot k, \qquad \text{bubble} = \frac{p - 1}{v\,m}, \quad m = k
$$

bubble 公式的推导和各种调度见 [Pipeline 并行调度](/train/pipeline-schedules)。三个量互相掐：

1. **$d$ 变大**（加卡）而 $B$ 不变：$k = m$ 变小，bubble 变大。Llama 3 从 8K 卡加到 16K 卡，每个 DP 组的 batch 从 32 条降到 16 条，MFU 从 43% 降到 41%（Table 4，作者自测）。
2. **$b$ 变大**（GEMM 更大、更高效）：$m$ 同样变小，激活还按 $b$ 线性涨。
3. **$p$ 变大**：bubble 的分子变大，而且 $m$ 至少要和 $p$ 一样多，1F1B 才能进入稳态（Megatron 的 interleaved 调度在 $m < p$ 时直接报错）。interleaved 还要把每个 stage 的层均分成 $v$ 段，所以 $L$ 要能被 $pv$ 整除。

例：$B = 512$、$b = 1$、$n_{\text{gpu}} = 1024$、$t = 8$：

| $p$ | $d$ | $m$ | bubble（$v = 1$） | bubble（$v = 5$） |
|---|---|---|---|---|
| 2 | 64 | 8 | 12.5% | 2.5% |
| 4 | 32 | 16 | 18.8% | 3.75% |
| 8 | 16 | 32 | 21.9% | 4.4% |

### 4. 决策流程

1. **TP 填满 NVLink 域：$t = 8$。** TP 是唯一既切权重又切激活、通信又在关键路径上的维度，只有 NVLink 撑得住；GQA 下还要 $t \le$ KV head 数，否则 KV head 要复制。单卡（配 ZeRO）就放得下的小模型不用 TP。
2. **代第 1 节的公式算显存。超了就加 PP**，取放得下的最小 $p$，再用 interleaved（取 $v$ 让 $L$ 能被 $pv$ 整除）把 bubble 压下去。PP 只切模型状态，所以只在模型状态是大头时有用。
3. **激活还是放不下（通常是序列很长）就加 CP**，激活按 $c$ 线性降。Llama 3 405B 在 128k 阶段开了 $c = 16$。CP 本身见 [长上下文训练](/train/long-context)。
4. **剩下的卡给 DP：$d = n_{\text{gpu}} / (tpc)$，开 ZeRO-1。** 有 PP 时不用 ZeRO-3：ZeRO-3 每个 micro-batch 都要重新 all-gather 一遍参数，通信变成 $k$ 倍；ZeRO-1 每步只同步一次，见 [ZeRO / FSDP](/parallel/zero-fsdp)。
5. **检查 batch**：$B$ 能被 $d \cdot b$ 整除，$m = B / (db) \ge p$，bubble 在几个百分点以内。$b$ 从 1 开始，有显存余量、而且 $m$ 还够多时才加大。

### 5. 算一笔：70B dense，seq 8k，1024 张 H100，global batch 约 4M token

模型按 Llama 3 70B（$L = 80$、$h = 8192$、64 个 query head、8 个 KV head，论文 Table 3）。$B = 4\text{M} / 8192 = 512$ 条序列。

1. **TP**：$t = 8$，等于 KV head 数，不用复制。
2. **先试 $p = 1$**：第 1 节已经算过，58.6 GB，压着 60 GB 的线，batch 或序列一变就 OOM。**$p = 2$**：模型状态 $7\times10^{10}/16 \times (4 + 12/64) = 18.3$ GB；激活 22.8 GB $\times (1 + \frac{1}{2 \times 5}) = 25.1$ GB；合计 **43.4 GB**，留得出 buffer。
3. **CP**：激活只有 25 GB，8k 不需要，$c = 1$。
4. **DP**：$d = 1024 / (8 \times 2) = 64$，开 ZeRO-1。
5. **batch**：$b = 1$，$k = m = 512 / 64 = 8$。每个 stage 40 层，$v = 5$ 时每段 8 层；$m = 8 \ge p$。bubble $= \frac{1}{5 \times 8} = 2.5\%$，不 interleave 是 12.5%。

再验一下通信和时间：

1. **一步的计算**：每卡 $6N \cdot Bs / n_{\text{gpu}} = 6 \times 7\times10^{10} \times 4.19\times10^6 / 1024 = 1.72\times10^{15}$ FLOPs。MFU 40% 时约 **4.35 s**（推导见 [训练算力账](/train/compute-mfu)）。
2. **DP 通信**：每卡的参数分片是 $4.375\times10^9$ 个，梯度 reduce-scatter 和参数 all-gather 在 ring 上各发 $\frac{63}{64} \times 4.375\times10^9 \times 2$ 字节 $\approx 8.6$ GB，共约 17.2 GB。跨节点 50 GB/s 约 0.34 s，是计算的 8%，还能和 backward 重叠。
3. **PP 通信**：每个 micro-batch 传一次 $[s, h]$ 的激活，开 SP 后每卡只传 $1/t$：$8192 \times 8192 \times 2 / 8 = 16.8$ MB，50 GB/s 下 0.3 ms，可以忽略。

对照一个公开配置：NeMo 的 `llama3_70b` 预训练 recipe 默认 TP 4、PP 4、VP 5、CP 2、开 SP，seq 8192，global batch 512，micro-batch 1（[NeMo v2.4.0 源码](https://github.com/NVIDIA/NeMo/blob/v2.4.0/nemo/collections/llm/recipes/llama3_70b.py)）。$t \cdot c = 8$，同样把一台机器的 8 张卡都用来切激活，只是一半给了 CP。放到 1024 卡上 $d = 32$、$m = 16$，bubble $3/80 = 3.75\%$，代第 1 节的公式约 45 GB，和上面的解差不多。

### 6. 两个公开配置对照

| | Llama 3 405B | DeepSeek-V3 |
|---|---|---|
| 结构 | dense，126 层，$h = 16384$ | MoE，总参数 671B，每 token 激活 37B，61 层 |
| 硬件 | 8K–16K 张 H100，机内 NVLink，机间 400 Gbps RoCE | 2048 张 H800，机内 NVLink 160 GB/s，机间 IB 50 GB/s |
| TP | 8 | 不用 |
| CP | 1（8k）；16（128k） | 不用 |
| PP | 16 | 16（DualPipe） |
| DP | 64 或 128（8k）；8（128k），FSDP | ZeRO-1 |
| EP | 无 | 64，跨 8 个节点 |
| 出处 | 论文 Table 4、第 3.3.2 节 | 技术报告第 3.2 节 |

为什么差这么多：

1. **Llama 3 405B 每一层都很大**。$h = 16384$，一层的权重和激活都必须切，只能靠 TP 8；405B 除以 8 还剩约 50B 参数，再用 PP 16 切到每卡约 32 亿参数。128k 阶段激活是 8k 的 16 倍，于是开 CP 16，同时 DP 从 128 降到 8，保持 16M token 的 global batch。
2. **DeepSeek-V3 的参数主要在专家里**。每个 token 只激活 37B，专家用 EP 64 分到 64 张卡上；attention 和 dense 部分不大，每张卡放一份，用 ZeRO-1 切优化器状态就够了。
3. **DeepSeek 明确说避开了「costly」的 TP**：H800 的 NVLink 只有 160 GB/s（报告第 3.2.2 节），TP 每层的 all-reduce 太贵。不用 TP 的代价是每卡激活更大，他们靠重算 RMSNorm 和 MLA 的 up-projection、把 EMA 放到 CPU、让 MTP 模块和主模型共用 embedding 和 output head 来省显存（第 3.2.3 节）。
4. **EP 的 all-to-all 跨节点，通信和计算差不多 1:1**（报告原话）。DualPipe 让一对 forward 和 backward 的计算和通信交错，把 all-to-all 和 PP 通信藏起来，这也是 PP 16 不只是为了显存的原因。

### 7. MFU 调优 checklist

按先后顺序。前面的便宜、不改收敛；最后才动并行配置，因为它连带显存、batch 和通信全要重算。

1. **先 profile 一个 step**（稳态，跳过前几步）。用 torch profiler 拿时间线（见 [Profiling](/gpu/profiling)），把一步拆成：GEMM 和 attention kernel、其他计算 kernel、单独跑的 NCCL（计算 stream 空着）、完全空闲（bubble、等数据、CPU）。同时用 $\text{token/s} \times 6N / (n_{\text{gpu}} \cdot \text{peak})$ 算出 MFU。
2. **调 micro-batch $b$**。看：峰值显存（`torch.cuda.max_memory_allocated()`）离上限还有多少；GEMM kernel 实测 TFLOP/s 是否明显低于大 GEMM。有余量、GEMM 偏小就加大 $b$，同时检查 $m = B/(db)$ 掉下来后 bubble 涨了多少。
3. **关掉不必要的 recompute**。看：配置里是不是开了 full recompute，显存余量是否够存回这些激活。full 换成 selective 或者关掉，每 token 从 $8N$ 回到 $6N$，step 时间约降 25%。
4. **打开 overlap**。看：时间线上单独跑的 NCCL 时间。DP 的梯度和参数同步对应 Megatron 的 `--overlap-grad-reduce`、`--overlap-param-gather`，TP 的通信对应 `--tp-comm-overlap`。原理和能藏住的条件见 [计算通信 overlap](/parallel/comm-overlap)。
5. **kernel 融合 / torch.compile**。看：非 GEMM kernel 的时间占比，以及大量很短的 elementwise kernel。融合 RMSNorm、RoPE、SwiGLU 的激活这类访存密集的算子，见 [torch.compile](/framework/torch-compile)。
6. **最后才改并行配置**。看：剩下的大头是什么。TP 通信单独跑的时间多，就在显存允许时降 $t$；bubble 大，就加 $v$ 或降 $p$；DP 通信藏不住，就让每卡每步的计算更多。改完回到第 1 步重新 profile。

## 面试追问

::: details Q：PP 切了 $p$ 段，为什么第一个 stage 的激活显存没降？
1F1B 要让流水线跑满，第一个 stage 必须先做 $p$ 个 micro-batch 的 forward，才开始第一个 backward，这时它存着 $p$ 份激活，每份 $L/p$ 层，合起来是 $L$ 层。所以 PP 降的是模型状态，激活要靠 TP、CP、recompute 或减小 $b$。
:::

::: details Q：卡数翻倍、global batch 不变，配置怎么调？
直接翻倍 DP 的话，每个 DP 组的 micro-batch 个数 $m$ 减半，bubble $\frac{p-1}{vm}$ 翻倍，每卡计算变少、通信更难藏。Llama 3 从 8K 卡到 16K 卡就是这样，MFU 43% → 41%。可以加大 $v$ 抵消 bubble；或者等训练后期 batch 按计划加大时再扩卡。
:::

::: details Q：有 PP 时 DP 为什么用 ZeRO-1 而不是 ZeRO-3？
ZeRO-3 的参数只存 $1/d$，每次 forward 和 backward 前都要 all-gather 回来。有 gradient accumulation 时一步要跑 $k$ 个 micro-batch，参数就要 gather $2k$ 次。ZeRO-1 每卡存完整参数（对这个 TP×PP 分片而言），一步只在最后做一次梯度 reduce-scatter 和参数 all-gather。PP 已经把参数切小了，存完整的一份不贵。
:::

::: details Q：TP 能不能开到 16？
不建议。H100 一台机器 8 张卡，TP 16 就要跨节点，每层 4 次集合通信走 50 GB/s 的网卡，比 NVLink 慢约 9 倍，而且在关键路径上。GQA 模型还有 $t \le$ KV head 数的限制。需要切更多就加 PP 或 CP。
:::

::: details Q：长序列时为什么加 CP 而不是加 TP？
TP 已经用满了 NVLink 域，再加就跨节点。CP 切的是序列，激活按 $c$ 降，attention 的计算量随 $s^2$ 增长，交换 K、V 的通信随 $s$ 增长，序列越长越容易藏住。GQA 下 K、V 比 Q 小得多，Llama 3 的 CP 直接 all-gather K、V，也能把开销压到可以忽略（第 3.3.2 节）。
:::

::: details Q：MoE 的 EP 放在哪个维度？
专家层在 DP（和 CP）那些卡里再按专家切：attention 部分照常做 DP，到 MoE 层时这些卡组成 EP 组，用 all-to-all 把 token 发到专家所在的卡。DeepSeek-V3 每层的路由专家均匀放在 8 个节点的 64 张卡上。原理见 [MoE 与 EP](/parallel/moe-ep)。
:::

## 手撕

枚举 $(t, p, d)$，按显存和 batch 两条约束过滤，打印候选：

```python
def mem_gb(N, L, h, s, b, t, p, d, v=1, c=1):
    states = N / (t * p) * (4 + 12 / d)                 # ZeRO-1：bf16 参数+梯度，fp32 优化器状态切 d 份
    act = 34 * s * b * h * L / (t * c)                  # 第一个 stage 存 L 层的激活（1F1B）
    act *= 1 + (p - 1) / (p * v) if v > 1 else 1        # interleaved 多存一些
    return (states + act) / 1e9

def candidates(N, L, h, s, n_gpu, B, b=1, budget=60, max_tp=8):
    out = []
    for t in (1, 2, 4, 8):
        if t > max_tp:
            continue
        for p in (1, 2, 4, 8, 16):
            if n_gpu % (t * p) or L % p:
                continue
            d = n_gpu // (t * p)
            if B % (d * b):
                continue
            m = B // (d * b)                            # 每步每条 pipeline 的 micro-batch 数
            if p > 1 and m < p:
                continue
            v = next((x for x in (5, 4, 2) if p > 1 and L % (p * x) == 0), 1)
            mem = mem_gb(N, L, h, s, b, t, p, d, v)
            if mem > budget:
                continue
            out.append((t, p, d, v, m, (p - 1) / (v * m), mem))
    return sorted(out, key=lambda r: (r[5], r[0] * r[1]))

for t, p, d, v, m, bub, mem in candidates(70e9, 80, 8192, 8192, 1024, 512):
    print(f"t={t} p={p} d={d:3d} v={v} m={m:2d} bubble={bub:5.1%} mem={mem:4.1f} GB")
# t=8 p=1 d=128 v=1 m= 4 bubble= 0.0% mem=58.6 GB
# t=8 p=2 d= 64 v=5 m= 8 bubble= 2.5% mem=43.4 GB
# t=8 p=4 d= 32 v=5 m=16 bubble= 3.8% mem=35.8 GB
# t=8 p=8 d= 16 v=5 m=32 bubble= 4.4% mem=32.0 GB
# t=8 p=16 d=  8 v=5 m=64 bubble= 4.7% mem=30.1 GB
# t=4 p=16 d= 16 v=5 m=32 bubble= 9.4% mem=59.4 GB
```

输出按 bubble 从小到大排。$p = 1$ 的解 bubble 为 0，但 58.6 GB 压着预算线；$t = 8, p = 2$ 留了约 17 GB 余量，bubble 2.5%，就是第 5 节的选择。$t \le 4$ 时激活只切了 4 份，PP 又不省激活，基本都超预算，只有 $p = 16$ 靠把模型状态压到很小才挤进来。常见变体：加入 $c$ 的循环处理 32k 以上序列；把 budget 换成实测的可用显存；按第 5 节的方法再给每个候选估一遍 DP 通信时间。

## 参考

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)：Table 4 并行配置与 MFU，第 3.3.2 节 4D 并行、维度顺序、PP 改进与 CP
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：第 3.2 节训练框架（PP 16、EP 64、ZeRO-1、DualPipe、不用 TP）
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)：每层激活公式，PP 下第一个 stage 的激活
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)：PTD-P、interleaved 1F1B
- [NeMo llama3_70b recipe（v2.4.0）](https://github.com/NVIDIA/NeMo/blob/v2.4.0/nemo/collections/llm/recipes/llama3_70b.py)
- [Megatron-LM arguments.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py)：`--overlap-grad-reduce`、`--overlap-param-gather` 等开关
