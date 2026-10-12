---
title: 长上下文训练：Context Parallel、RoPE 外推与分阶段扩长
status: draft
tags: [long-context, context-parallel, ring-attention, ulysses, rope, yarn]
difficulty: 4
order: 8
related: [/posttrain/training-memory, /parallel/parallelism-overview, /parallel/comm-overlap, /inference/flash-attention, /posttrain/sft, /inference/dit-serving]
stack: []
---

# 长上下文训练：Context Parallel、RoPE 外推与分阶段扩长

> 序列从 8K 拉到 128K，算力、显存、位置编码各出什么问题，分别怎么解

## 一句话结论

序列变长后有三个问题，各有一个标准解法：

1. **算力**：attention 每 token 的 FLOPs 和 $s$ 成正比。8B 模型在 128K 时 attention 是线性层的 2 倍多，所以预训练主要用短序列，**最后阶段才分几步扩到 128K**（Llama 3 405B 用 6 个阶段、约 800B token）。
2. **显存**：激活和 $s$ 成正比，128K 时一张卡放不下一条序列，要用 **Context Parallel（CP）** 把一条序列切到 $c$ 张卡上。attention 要看到全部 K、V，有三种做法：Ring Attention 环形传 KV；Ulysses 用 all-to-all 换成按 head 切；Llama 3 直接 all-gather KV。causal mask 下要用 **zigzag 切分**才能负载均衡。
3. **位置编码**：RoPE 的低频维度在训练长度内没转满一圈，超出长度就会遇到没见过的角度。办法是把频率调低：调大 base（ABF）、所有维度一起压缩（PI）、只压缩低频维度（YaRN）。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值（Llama 3 8B） |
|---|---|---|
| $s$ | 序列长度（token） | 8192 → 131072 |
| $N$ | 参数量 | $8\times10^9$ |
| $L$ | 层数 | 32 |
| $h$ | hidden size，$h = H d_h$ | 4096 |
| $H, H_{kv}$ | query head 数、KV head 数（GQA） | 32、8 |
| $d_h$ | 每个 head 的维度 | 128 |
| $c$ | CP 度：一条序列切到几张卡上 | 4、16 |
| $\Theta$ | RoPE 的 base | 10,000 / 500,000 |
| $\theta_j$ | RoPE 第 $j$ 对维度的角频率，$j = 0, \dots, d_h/2 - 1$ | $\Theta^{-2j/d_h}$ |
| $s_0, \kappa$ | 原训练长度、扩长倍数 $\kappa = s_{\text{新}} / s_0$ | 4096、16 |

Llama 3 的结构参数见 [Llama 3 报告 Table 3](https://arxiv.org/abs/2407.21783)：8B 是 32 层、$h=4096$；70B 是 80 层、$h=8192$；405B 是 126 层、$h=16384$；KV head 都是 8。

### 1. 序列变长时什么在涨

**FLOPs**。线性层每 token $6N$（forward 2、backward 4，见 [训练显存账](/posttrain/training-memory)），和 $s$ 无关。attention 的两个矩阵乘 $QK^\top$ 和 $PV$ 没有参数，不在 $6N$ 里：

1. 位置 $i$ 的 query 和 $i$ 个 key 做点积，所有 head 合计 $2hi$ FLOPs；乘 $V$ 再 $2hi$。causal 下 $i$ 平均是 $s/2$，每层每 token forward $4h \cdot s/2 = 2hs$。
2. backward 是 forward 的 2 倍（对两个输入各求一次梯度），训练合计 $6hs$。
3. 乘层数：每 token $6Lhs$。GQA 只减少 K、V 的大小，不减少这部分 FLOPs（$Q$ 的 head 数没变）。
4. 这是 GPU 实际执行的量。报 MFU 时 PaLM 的口径不减半，记 $12Lhs$，见 [训练算力账](/train/compute-mfu)。两者只差 causal 这个 ½。

$$
\frac{\text{attention}}{\text{线性层}} = \frac{6Lhs}{6N} = \frac{Lhs}{N}
$$

| 模型 | $Lh/N$ | $s = 8$K | $s = 32$K | $s = 128$K |
|---|---|---|---|---|
| 8B | $1.64\times10^{-5}$ | 0.13 | 0.54 | 2.15 |
| 70B | $9.36\times10^{-6}$ | 0.08 | 0.31 | 1.23 |
| 405B | $5.10\times10^{-6}$ | 0.04 | 0.17 | 0.67 |

自己算一遍 8B、128K：$32 \times 4096 \times 131072 / 8\times10^9 = 2.15$，即每 token 的训练算力是 8K 时的 $(1 + 2.15)/(1 + 0.13) = 2.8$ 倍。模型越大，$h/N$ 越小，长序列的相对代价越低。

**激活**。用了 FlashAttention 后每层约 $34sbh$ 字节（推导见 [训练显存账](/posttrain/training-memory)），和 $s$ 成正比。8B、128K、$b=1$：每层 $34 \times 131072 \times 4096 = 18.3$ GB，32 层 584 GB；full checkpointing 后仍有 $2sbhL = 34$ GB 加上重算时一层的 18 GB。单卡放不下，只能沿序列维切。

### 2. Context Parallel / Ring Attention

CP 把一条序列切成 $c$ 段，每张卡持有 $s/c$ 个 token。线性层、norm 都是逐 token 的，各算各的，不需要通信。只有 attention 要看到全部 K、V。

[Ring Attention](https://arxiv.org/abs/2310.01889) 的做法：

1. 每张卡算出自己那段的 $Q, K, V$。
2. 共 $c$ 步。每一步用本地 $Q$ 和手上的一块 $K, V$ 算 attention，得到部分输出和每行的 logsumexp；同时把这块 $K, V$ 发给下一张卡，从上一张卡收下一块。
3. 各步的部分结果用 online softmax 合并（[Online Softmax](/handson/online-softmax)，和 [FlashAttention](/inference/flash-attention) 里块与块之间的合并是同一件事）。
4. backward 时 $K, V$ 再绕一圈，$\partial K$、$\partial V$ 跟着传回所属的卡。

**causal mask 下负载不均**。平均切成 $c$ 段时，第 0 段只看自己，最后一段要看全部，最后一张卡的活最多。解法是 zigzag（load-balanced）切分：序列切成 $2c$ 块，卡 $i$ 拿第 $i$ 块和第 $2c-1-i$ 块，一头一尾配对。Llama 3 就是这样切的（[Llama 3 §3.3.2](https://arxiv.org/abs/2407.21783)）。

$c = 4$，8 块（`#` 整块要算，`\` 对角块算一半，`.` 被 mask 掉；单位是一个 $(s/8)^2$ 的块）：

```
K 块:   0 1 2 3 4 5 6 7   工作量
Q0 r0   \ . . . . . . .   0.5
Q1 r1   # \ . . . . . .   1.5
Q2 r2   # # \ . . . . .   2.5
Q3 r3   # # # \ . . . .   3.5
Q4 r3   # # # # \ . . .   4.5
Q5 r2   # # # # # \ . .   5.5
Q6 r1   # # # # # # \ .   6.5
Q7 r0   # # # # # # # \   7.5
```

1. zigzag：卡 $i$ 的工作量是 $(i + 0.5) + (2c - 1 - i + 0.5) = 2c$，四张卡都是 8。
2. 平均切：卡 0 拿 Q0、Q1，工作量 2；卡 3 拿 Q6、Q7，工作量 14。平均 8，最慢的卡是平均的 1.75 倍，整体就慢 1.75 倍。

**通信量和能否藏住**。每张卡每层 forward 要收 $c-1$ 块 KV，合计

$$
\frac{c-1}{c} \cdot 2 s H_{kv} d_h \cdot 2 \ \text{字节}
$$

（$K$、$V$ 两份，bf16 每元素 2 字节）。每一步的计算是 $s/c$ 个 query 对 $s/c$ 个 key，$4(s/c)^2h$ FLOPs；通信是一块 KV，$2(s/c)H_{kv}d_h \times 2$ 字节。计算时间不短于通信时间才藏得住（推导见 [计算通信 Overlap](/parallel/comm-overlap)，那里取 $b = 2$ 字节就是下式）：

$$
\frac{4(s/c)^2 h}{\text{peak} \cdot \text{MFU}} \ge \frac{4(s/c) H_{kv} d_h}{\text{BW}}
\iff
\frac{s}{c} \ \ge\ \frac{H_{kv}}{H} \cdot \frac{\text{peak} \cdot \text{MFU}}{\text{BW}}
$$

代入 H100 bf16 dense 峰值 989 TFLOP/s、MFU 50%、Llama 3 8B 的 $H_{kv}/H = 1/4$：NVLink 单向 450 GB/s 时每卡至少 274 个 token，IB 每卡 50 GB/s 时至少 2470 个。causal 下 zigzag 每一步平均只有一半的块要算，门槛再乘 2，即 IB 下约 4940。128K、$c = 16$ 时每卡 8192 个 token，跨节点也藏得住。

### 3. Ulysses 与 Llama 3 的 all-gather 方案

[DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509) 换一个思路：attention 的各个 head 互相独立，那就让每张卡看完整序列、只算一部分 head。

1. 进 attention 前一次 all-to-all：每张卡从「$s/c$ 个 token、全部 head」换成「全部 $s$ 个 token、$H/c$ 个 head」（$Q$、$K$、$V$ 都换）。
2. 本地就是一个普通的 attention，任何 mask、任何 FlashAttention kernel 都能直接用。
3. 算完再一次 all-to-all 换回按序列切。

论文的通信量（MHA）：每层每条链路 $4sh/c$ 个元素，序列和卡数同比例增加时不变（论文用 $N$ 记序列长度）。

约束：

1. $c$ 不能超过 $H$，且要整除 $H$。
2. GQA 下 $K$、$V$ 只有 $H_{kv}$ 个 head。$c > H_{kv}$ 时，同一个 KV head 要发到多张卡上（复制）。Llama 3 8B 的 $H_{kv} = 8$，Ulysses 度超过 8 就要复制 KV。

**Llama 3 用的是第三种：all-gather KV**（[Llama 3 §3.3.2](https://arxiv.org/abs/2407.21783)）。zigzag 切成 $2c$ 块；attention 前先 all-gather 全部 $K, V$，再用本地的 $Q$ 算。all-gather 的延迟暴露在关键路径上，作者给了两个理由：

1. 支持各种 mask（比如文档掩码）更简单、更灵活。
2. 用了 GQA，$K, V$ 比 $Q$ 小得多，attention 是 $O(s^2)$、all-gather 是 $O(s)$，暴露的时间可以忽略。

算一笔，405B、128K、$c = 16$：每层完整的 $K + V$ 是 $2 \times 131072 \times (8 \times 128) \times 2 = 537$ MB，每个 CP rank 收其中 15/16，即 503 MB，IB 50 GB/s 下 10 ms。同一 rank 每层 forward 的计算：attention $2s^2h / c = 3.5\times10^{13}$ FLOP，线性层 $2 \times (405\text{B}/126) \times s / c = 5.3\times10^{13}$ FLOP，合计按 494 TFLOP/s 约 178 ms。通信约占 6%。TP 会把计算和每卡要收的 KV 一起除以 TP 度，比例不变。Llama 3 405B 在 128K 阶段用 TP 8、CP 16、PP 16、DP 8，MFU 38%，8K 阶段同样 16K 张卡是 41%（作者自测，报告 Table 4）。

三种做法对比（视频生成里 Ulysses 和 Ring 的组合 USP 见 [DiT 推理与 Serving](/inference/dit-serving)）：

| | Ring（P2P 传 KV） | all-gather KV（Llama 3） | Ulysses（all-to-all） |
|---|---|---|---|
| 每卡每层通信（forward） | $\frac{c-1}{c} \cdot 2sH_{kv}d_h$ 元素，几乎不随 $c$ 变 | 同左 | $\frac{2s(h + H_{kv}d_h)}{c}$ 元素，随 $c$ 增大而减小 |
| 能否和计算重叠 | 能，算当前块时收下一块 | 不能，靠 GQA 让它小 | 不能，在关键路径上 |
| 并行度上限 | 无 head 限制 | 无 head 限制 | $\le H$；超过 $H_{kv}$ 要复制 KV |
| causal / 文档 mask | 要 zigzag；文档 mask 要逐块判断 | 本地有完整 KV，mask 好做 | 本地是完整序列，mask 直接用 |
| 适合 | 跨节点、序列极长 | GQA 模型、复杂 mask | 节点内（all-to-all 对拓扑敏感） |

### 4. 文档掩码

预训练把多篇文档拼成一条 128K 的序列（packing）。不加掩码时，后一篇文档的 token 会 attend 到前面无关的文档。Llama 3 加了文档掩码：「标准预训练里影响有限，但在超长序列的继续预训练里很重要」（[Llama 3 §3.2](https://arxiv.org/abs/2407.21783)，作者自测）。

实现上不用稠密 mask，而是用 FlashAttention 的 varlen 接口传 `cu_seqlens`（每篇文档的累计起点），同时把 position_ids 按文档重置，做法见 [SFT 的 packing 一节](/posttrain/sft)。

文档掩码还改变了 FLOPs：attention 的平方项从 $s^2$ 变成 $\sum_i l_i^2$（$l_i$ 是各文档长度）。一条 128K 序列如果由 32 篇 4K 文档拼成，attention 的计算只有不加掩码时的 $1/32$。第 1 节的比值是「整条都是一篇长文档」时的上界。和 CP 合用时，各卡的工作量取决于文档边界落在哪，zigzag 不再严格均衡，这也是 Llama 3 选 all-gather 的原因之一。

### 5. RoPE 扩长：改哪些维度的频率

RoPE 把 $q, k$ 的维度两两配对，第 $j$ 对在位置 $m$ 处旋转角度 $m\theta_j$，$\theta_j = \Theta^{-2j/d_h}$。转一圈需要的 token 数（波长）是 $\lambda_j = 2\pi/\theta_j$。$\Theta = 10{,}000$、$d_h = 128$ 时，$\lambda_j$ 从 $j=0$ 的 6 个 token 到 $j=63$ 的 54,410 个。

问题出在低频维度：训练长度 $s_0 = 4096$ 时，有 18 对维度的波长超过 4096，训练中从没转满一圈。推理时位置超过 $s_0$，这些维度会出现训练中没见过的角度，attention 分数就乱了。高频维度在训练中已经转过很多圈，各种角度都见过，没有这个问题。

三种做法都是把频率调低，区别在调哪些维度：

**(1) ABF（adjusted base frequency）**：把 base 调大（[Xiong et al. 2023](https://arxiv.org/abs/2309.16039) 把 10,000 调到 500,000）。

$$
\theta_j' = \Theta'^{-2j/d_h}, \qquad \frac{\theta_j'}{\theta_j} = \left(\frac{\Theta}{\Theta'}\right)^{2j/d_h}
$$

$j = 0$ 不变，$j$ 越大降得越多，最后一对降到约 1/47，波长从 5.4 万变成 256 万。Llama 3 的模型结构直接把 $\Theta$ 定为 500,000（报告 §3.2、Table 3），并引 Xiong et al. 说这个值在 32,768 长度内有效。

**(2) PI（Position Interpolation）**：把位置压回原范围（[Chen et al. 2023](https://arxiv.org/abs/2306.15595)）。

$$
m \to \frac{m}{\kappa} \quad\Longleftrightarrow\quad \theta_j \to \frac{\theta_j}{\kappa}\ \ (\text{所有 } j)
$$

所有维度一起放慢 $\kappa$ 倍，不会再出现没见过的角度。代价是高频维度也被压缩了，相邻 token 的角度差缩小 $\kappa$ 倍，局部分辨率下降，需要微调。作者把 LLaMA 7B–65B 扩到 32768，微调在 1000 步以内（作者自测）。

**(3) YaRN**（[Peng et al. 2023](https://arxiv.org/abs/2309.00071)）：只压缩低频维度。用 $r_j = s_0 / \lambda_j$（第 $j$ 对在原长度内转了几圈）分三段，$\gamma$ 是线性过渡：

$$
\theta_j' = (1-\gamma_j)\frac{\theta_j}{\kappa} + \gamma_j\,\theta_j, \qquad
\gamma_j = \begin{cases} 0 & r_j < \alpha \\ 1 & r_j > \beta \\ \frac{r_j - \alpha}{\beta - \alpha} & \text{其他} \end{cases}
$$

Llama 系列取 $\alpha = 1$、$\beta = 32$。转不满一圈的维度完全按 PI 压缩，转了 32 圈以上的维度不动。另外给 attention logits 加温度 $t$，$\sqrt{1/t} = 0.1\ln\kappa + 1$。例：$\Theta = 10{,}000$、$s_0 = 4096$、$\kappa = 16$，$j = 0$–20 共 21 对不动，$j = 46$–63 共 18 对除以 16，中间 25 对过渡，$\sqrt{1/t} = 1.28$。作者自测比之前的方法少用 10 倍 token、2.5 倍训练步数。

### 6. 分阶段扩长

Llama 3 405B 的做法（[Llama 3 §3.4.2–3.4.3](https://arxiv.org/abs/2407.21783)）：

1. 预训练共 15.6T token。开头 252M token 用 4K 序列，之后主体用 8K（§3.4.1）。
2. 最后阶段分 **6 步**从 8K 扩到 128K，共约 **800B token**。每一步训到模型适应为止，标准是两条：短上下文评测完全恢复；该长度下「大海捞针」全对。
3. 最后 40M token 做退火，学习率线性降到 0，保持 128K。

报告给的理由是 self-attention 的计算随序列长度平方增长。用第 1 节的比值算 405B（8K 时 0.042，128K 时 0.668）：

1. 全程 128K：每 token 的代价是 8K 时的 $1.668 / 1.042 = 1.60$ 倍。
2. 分阶段（按 800B 包含在 15.6T 内，且这 800B 全按 128K 算，是上界）：$14.8\text{T} \times 1.042 + 0.8\text{T} \times 1.668 = 16.75\text{T}$，比全程 8K 的 $15.6\text{T} \times 1.042 = 16.25\text{T}$ 只多 3%，比全程 128K 的 26.0T 少 36%。
3. 显存和并行度：8K 阶段不需要 CP，128K 阶段要 CP 16，MFU 从 41% 降到 38%（作者自测）。

## 面试追问

::: details Q：GQA 能降低长序列的 attention FLOPs 吗？
不能。FLOPs 由 $QK^\top$ 和 $PV$ 决定，query 的 head 数 $H$ 没变，每 token 还是 $6Lhs$。GQA 减少的是 K、V 的大小：推理时是 KV cache，训练时是 CP 要传的 KV（通信量正比于 $H_{kv}$）。这也是 Llama 3 敢用不重叠的 all-gather 的原因。
:::

::: details Q：Ring Attention 和 Ulysses 怎么选？
看并行度和网络。Ulysses 每层两次 all-to-all，本地是普通 attention，mask 和 kernel 都不用改，适合节点内；并行度受 head 数限制，GQA 下超过 $H_{kv}$ 要复制 KV。Ring 用 P2P，能和计算重叠，可以跨节点、并行度不受 head 限制，但 causal 要 zigzag、文档 mask 要逐块处理。序列极长时两者组合：节点内 Ulysses，节点间 Ring。
:::

::: details Q：causal mask 下为什么不能平均切序列？
后面的 token 要 attend 的 key 更多。平均切成 $c$ 段时卡 $i$ 的工作量正比于 $i + 0.5$，$c = 4$ 时最慢的卡是平均的 1.75 倍，其他卡空等。切成 $2c$ 块、卡 $i$ 拿第 $i$ 块和第 $2c-1-i$ 块，每卡都是 $2c$ 个单位，完全均衡。
:::

::: details Q：PI 和 ABF 都是降低频率，区别是什么？
PI 所有维度一起除以 $\kappa$，高频维度也被压缩，相邻 token 更难区分，需要微调。ABF 调大 base，$j = 0$ 不变，越低频的维度降得越多，对局部位置信息影响小。YaRN 更进一步，按每个维度在原长度内转了几圈来决定压不压：转得多的不动，转不满一圈的按 PI 压。
:::

::: details Q：为什么不从一开始就用 128K 训练？
1. 算力：405B 全程 128K，每 token 贵 60%；分阶段只多约 3%。
2. 数据：多数文档本来就短，加了文档掩码后长序列里大部分 attention 也用不上。
3. 并行：128K 要 CP 16，MFU 更低。
所以先用短序列学知识，最后用约 5% 的 token 扩长度，扩完检查短任务没退化、大海捞针全对。
:::

::: details Q：长上下文训练还有哪些显存手段？
激活和 $s$ 成正比，按 [训练显存账](/posttrain/training-memory) 的顺序：FlashAttention 去掉 $s^2$ 项；selective / full checkpointing；TP 配 sequence parallel 把激活再除以 TP 度；最后是 CP，把每卡的 $s$ 变成 $s/c$。CP 是唯一让每卡激活和总长度脱钩的手段。
:::

## 手撕

**zigzag 切分与负载均衡**：对一个小序列建 causal mask，数每张卡要算的 $(q, k)$ 对数。

```python
import numpy as np

def zigzag_chunks(c):
    """序列切 2c 块，卡 i 拿第 i 块和第 2c-1-i 块。"""
    return [[i, 2 * c - 1 - i] for i in range(c)]

def naive_chunks(c):
    """平均切 c 段，卡 i 拿第 i 段（用 2c 小块表示，即第 2i、2i+1 块）。"""
    return [[2 * i, 2 * i + 1] for i in range(c)]

def causal_work(s, c, chunks):
    """每张卡要算的 (q, k) 对数：q 属于本卡，k <= q。"""
    n = s // (2 * c)
    q = np.arange(s)[:, None]
    k = np.arange(s)[None, :]
    allowed = k <= q
    return [int(sum(allowed[j * n:(j + 1) * n].sum() for j in ch)) for ch in chunks]

s, c = 64, 4
print(zigzag_chunks(c))                        # [[0, 7], [1, 6], [2, 5], [3, 4]]
print(causal_work(s, c, naive_chunks(c)))      # [136, 392, 648, 904]
print(causal_work(s, c, zigzag_chunks(c)))     # [520, 520, 520, 520]
```

证明：块大小 $n = s/2c$，第 $j$ 块要算 $jn^2 + n(n+1)/2$ 对。卡 $i$ 拿第 $i$ 和第 $2c-1-i$ 块，合计 $(2c-1)n^2 + n(n+1)$，和 $i$ 无关。

**RoPE 频率、PI 与 YaRN**：打印 $j = 0, 16, 32, 48, 63$ 的波长。

```python
def rope_theta(d_h=128, base=10000.0):
    j = np.arange(d_h // 2)
    return base ** (-2 * j / d_h)                      # 第 j 对维度的角频率

def pi_theta(theta, kappa):
    return theta / kappa                               # 所有维度一起放慢 kappa 倍

def yarn_theta(theta, kappa, s0, alpha=1.0, beta=32.0):
    r = s0 * theta / (2 * np.pi)                       # 原训练长度内转了几圈
    gamma = np.clip((r - alpha) / (beta - alpha), 0, 1)
    return (1 - gamma) * theta / kappa + gamma * theta

th = rope_theta()
wl = lambda t: np.round(2 * np.pi / t[[0, 16, 32, 48, 63]]).astype(int)
print(wl(th))                          # [    6    63   628  6283 54410]
print(wl(rope_theta(base=5e5)))        # ABF: [6 167 4443 118143 2559196]
print(wl(pi_theta(th, 16)))            # PI:  [101 1005 10053 100531 870562]
print(wl(yarn_theta(th, 16, s0=4096))) # YaRN:[6 63 2739 100531 870562]
```

YaRN 一行能看出它的意思：高频维度（前两个）和原来一样，低频维度（后两个）和 PI 一样，中间过渡。

## 参考

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)：§3.2 文档掩码与 RoPE base，§3.3.2 CP，§3.4.2 长上下文预训练，Table 4
- [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889)（Liu et al., 2023）
- [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)（Jacobs et al., 2023）
- [Effective Long-Context Scaling of Foundation Models](https://arxiv.org/abs/2309.16039)（Xiong et al., 2023）：RoPE ABF
- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)（Chen et al., 2023）
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)（Peng et al., 2023）
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)：每层激活 $34sbh$
