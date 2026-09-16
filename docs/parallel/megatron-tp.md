---
title: Megatron Tensor Parallel
status: draft
tags: [tp, megatron]
difficulty: 3
order: 2
related: [/parallel/parallelism-overview, /parallel/collective-comm, /inference/memory-accounting]
---

# Megatron Tensor Parallel

> column / row parallel 组合为什么能只要两次 all-reduce

## 一句话结论

**先 column 后 row**。第一个矩阵按列切（输出按列分块，无需通信），第二个矩阵按行切（输入正好是上一步的列分块，输出是部分和），只在最后做一次 all-reduce。MLP 和 attention 各是一个这样的 pair，所以每层两次 all-reduce。

## 推导

### 两种切法

对 $Y = XA$，$X \in \mathbb{R}^{S \times d}$，$A \in \mathbb{R}^{d \times h}$：

**Column parallel**：$A = [A_1, A_2]$ 按列切。

$$
Y = X[A_1, A_2] = [XA_1, XA_2]
$$

每张卡拿到完整的 $X$，算出 $Y$ 的一部分列。**不需要通信**，但输出是分块的。

**Row parallel**：$A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}$ 按行切，此时 $X$ 必须按列切成 $[X_1, X_2]$。

$$
Y = [X_1, X_2] \begin{bmatrix} A_1 \\ A_2 \end{bmatrix} = X_1 A_1 + X_2 A_2
$$

每张卡算出一个**部分和**，形状是完整的 $Y$，但数值不完整，**需要 all-reduce** 才能得到正确结果。

### 为什么组合起来只要一次通信

关键：column parallel 的**输出恰好是** row parallel 需要的**输入形式**（都是按 hidden 维度分块）。

MLP 是 $Y = \text{GeLU}(XA)B$：

```
X (完整) ──column parallel A──> [XA_1, XA_2]  (每卡一块，无通信)
                                     ↓ GeLU 是 elementwise，按块算没问题
        ──row parallel B──> 部分和  ──all-reduce──> Y (完整)
```

如果反过来（先 row 后 column），第一步就要 all-reduce 才能拿到完整的中间结果，第二步结束又要一次，变成两次。所以顺序不能颠倒。

**GeLU 必须是 elementwise 的**，这是这个技巧成立的前提。如果中间夹一个需要全局信息的操作（比如 LayerNorm 沿 hidden 维归一化），就必须先 all-reduce。这也是为什么 LayerNorm 被留在 all-reduce 之后，或者用 sequence parallel 单独处理。

### Attention 同理

$W_q, W_k, W_v$ 按**列**切，等价于**按 head 切**：每张卡拿到 $H / N$ 个完整的 head。attention 本身是 head 内的计算，head 之间独立，所以每张卡独立算自己那几个 head，**完全不需要通信**。

$W_o$ 按**行**切，输入正是各卡的 head 输出拼起来的列分块，输出是部分和，一次 all-reduce。

```
X ──[Wq,Wk,Wv 按 head 切]──> 每卡 H/N 个 head 的 attention  (无通信)
  ──[Wo 按行切]──> 部分和 ──all-reduce──> 完整输出
```

**约束**：$H$ 必须能被 TP 整除。GQA 下还要求 $H_{kv}$ 能被 TP 整除，否则 KV head 要复制（见 [显存账](/inference/memory-accounting)）。

### 通信量

每次 all-reduce 的消息大小是一份完整激活 $B \cdot S \cdot d \cdot b$，ring 算法下每卡实际收发 $\frac{2(N-1)}{N}$ 倍。

**注意通信量和模型大小无关**，只和 $B \cdot S \cdot d$ 有关。参数量涨 10 倍（靠加层）通信次数涨 10 倍但单次不变；序列长度涨 10 倍则单次涨 10 倍。

前向每层 2 次，反向每层也是 2 次（all-reduce 的反向还是 all-reduce，因为 $f$ 和 $g$ 是一对共轭算子：前向 identity 的地方反向 all-reduce，前向 all-reduce 的地方反向 identity）。

### 加上 SP

TP 下 LayerNorm 和 dropout 在每卡上重复计算，激活也是完整复制的。Sequence parallel 把这段按序列维切开，把 all-reduce 拆成 `reduce-scatter`（进入 SP 区）和 `all-gather`（离开 SP 区）。**通信总量完全不变**（一次 all-reduce = 一次 reduce-scatter + 一次 all-gather），但激活显存降到 $1/N$。基本是白捡，Megatron 默认开。

## 交互

把 TP 从 1 调到 8，看每卡权重下降和 all-reduce 通信量的变化。

<ParallelismViz :tp="8" :pp="1" />

## 面试追问

::: details Q：为什么不能先 row parallel 再 column parallel？
row parallel 的输出是部分和，必须 all-reduce 才能用；而 column parallel 需要完整的输入。所以 row→column 要在中间通信一次，末尾还要再来一次（column 的输出是分块的，下一层需要完整输入），总共两次。column→row 的输出天然是完整的部分和，一次搞定。
:::

::: details Q：all-reduce 能不能和计算 overlap？
前向比较难，因为 MLP 的输出要等 all-reduce 完成才能进下一层，在关键路径上。可行的做法是把 GEMM 切成小块，算完一块就开始通信这一块，用 `reduce-scatter` 的流水化（Megatron 的 `--tp-comm-overlap`，需要 NVLink 和特殊的 kernel）。反向传播的梯度 all-reduce 容易 overlap 得多，因为后面还有别的层要算。
:::

::: details Q：TP=8 和 TP=4 + PP=2，同样 8 张卡，选哪个？
单节点内选 TP=8：不引入 bubble，延迟更低，实现也简单。跨节点必须避免 TP：TP=4+PP=2 让 TP 组落在节点内走 NVLink，PP 的 P2P 跨节点走 IB（通信量小两个数量级）。所以这个选择本质由拓扑决定，不由卡数决定。
:::

::: details Q：为什么 embedding 和 LM head 也要切？
vocab 很大（128k）时 embedding 矩阵 $V \times d$ 本身就有 10 亿参数，LM head 输出的 logits $[B, S, V]$ 更是巨大（batch 8、2k token、128k vocab、fp32 = 8 GiB）。按 vocab 维切开后，每卡算自己那部分 logits，交叉熵可以用一次 all-reduce 求全局 max 和 sum（正是 [数值稳定 softmax](/handson/stable-softmax) 的两个统计量）来完成，避免物化完整 logits。
:::

## 参考

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Reducing Activation Recomputation in Large Transformer Models (SP)](https://arxiv.org/abs/2205.05198)
