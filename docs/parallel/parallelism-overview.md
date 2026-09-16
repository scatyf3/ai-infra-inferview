---
title: TP / PP / DP / EP / SP / CP 总览
status: draft
tags: [tp, pp, dp, ep, sp]
difficulty: 3
order: 1
related: [/parallel/megatron-tp, /parallel/collective-comm, /inference/memory-accounting]
---

# TP / PP / DP / EP / SP / CP 总览

> 各自切什么、通信量、通信插在哪一层

## 一句话结论

六种并行切的是六个不同的维度：**TP 切权重矩阵的行列、PP 切层、DP 切 batch、EP 切专家、SP/CP 切序列**。选哪个由「什么放不下」和「什么链路快」共同决定：NVLink 域内用 TP，跨节点用 PP 或 DP，MoE 用 EP，长上下文用 CP。

## 推导

| 并行 | 切什么 | 每卡权重 | 通信原语 | 通信量（每层每次） | 插在哪 |
|---|---|---|---|---|---|
| TP | 权重的行/列 | $1/N$ | all-reduce | $B S d \cdot b$ | attention 输出后、MLP 输出后，各一次 |
| PP | 层 | $1/N$ | P2P | $B_{micro} S d \cdot b$ | stage 边界（不是每层） |
| DP | batch | ×1（复制） | 推理无 / 训练 all-reduce 梯度 | 训练：整个权重 | 每 step 一次 |
| EP | expert | expert 部分 $1/N$ | all-to-all | $B S \cdot \text{topk} \cdot d \cdot b$ ×2 | 每个 MoE 层 dispatch + combine |
| SP | 序列（norm/dropout 处） | 同 TP | all-gather + reduce-scatter | 总量同 TP 的 all-reduce | 和 TP 配套，替换 all-reduce |
| CP | 序列（attention 处） | 同 TP | ring P2P 传 KV | $B S_{chunk} H_{kv} d_h \cdot b$ | attention 内部，环形轮转 |

### TP：通信在关键路径上

Megatron 的组合让每层只需要两次 all-reduce（见 [Megatron Tensor Parallel](/parallel/megatron-tp)）。关键性质是**通信量与模型大小无关**，只和 $B \cdot S \cdot d$ 有关。70B、batch 8、2k token、bf16：$8 \times 2048 \times 8192 \times 2 = 256$ MiB 一次，ring all-reduce 下每卡实际发 $2(N-1)/N \times 256$ MiB。80 层 × 2 次 = 160 次，总量非常可观，所以 **TP 必须在 NVLink 域内**（900 GB/s），跨 PCIe 或跨节点做 TP 会被通信打死。

推论：TP 的上限通常是单节点的 8 张卡。

### PP：通信量小但有 bubble

只在 stage 边界传激活，通信量比 TP 小两个数量级，可以跨节点走 IB。代价是流水线气泡：

$$
\text{bubble} = \frac{p - 1}{m}
$$

$p$ 是 stage 数，$m$ 是 micro-batch 数。$m \gg p$ 才划算，所以 PP 需要大 batch 来摊薄。1F1B 调度能把峰值激活显存从 $O(m)$ 降到 $O(p)$，interleaved（virtual pipeline）能把 bubble 再降到 $\frac{p-1}{m v}$，代价是通信次数 ×$v$。

推理场景 PP 用得少，因为 decode 的 batch 天然小，bubble 摊不掉；但显存实在放不下时（比如 8 卡放不下 405B）还是要用。

### DP：推理时最省事

每个副本持有完整模型，请求按负载分流，副本之间零通信。推理扩容的默认手段。注意 **DP attention**：MoE 模型上 attention 部分用 DP（因为 attention 的权重小）、expert 部分用 EP，能避免 attention 的 KV 被 TP 复制，DeepSeek 的部署方案就是这么做的。

### EP：all-to-all 是新瓶颈

每个 MoE 层两次 all-to-all：dispatch 把 token 送到 expert 所在的卡，combine 把结果收回。通信量正比于 $\text{topk}$，且是 all-to-all 不是 all-reduce，对网络拓扑更敏感。最大的麻烦是**负载不均**：某些 expert 被路由到的 token 远多于平均，那张卡成为木桶短板，其他卡空等。缓解手段是 auxiliary loss（训练时）、capacity factor + drop（牺牲质量）、EPLB 式的 expert 复制与重排（推理时）。

### SP 和 CP：都切序列但位置不同

**SP（sequence parallel）** 是 TP 的补充：TP 下 LayerNorm 和 dropout 这些 elementwise 操作在每张卡上是重复算的，激活也是完整复制的。SP 把这段也按序列切开，把 TP 的 all-reduce 拆成 `reduce-scatter + all-gather`，**总通信量不变**但激活显存降到 $1/N$。基本是白捡的，Megatron 默认开。

**CP（context parallel）/ Ring Attention** 是给超长上下文的：把序列切成 $N$ 段，每张卡持有一段的 Q/K/V，然后环形轮转 K/V，每轮和本地 Q 算一次局部 attention，用 online softmax 合并。通信可以和计算 overlap，因为下一轮的 K/V 可以在算当前轮时提前收。

### 组合起来

典型 3D：$N = \text{TP} \times \text{PP} \times \text{DP}$。分配原则：
1. TP 填满 NVLink 域（≤ 8），且不超过 $H_{kv}$。
2. 还放不下就加 PP，跨节点走 IB。
3. 剩下的都给 DP 扩吞吐。
4. MoE 的话 expert 维度单独用 EP，和 attention 的并行方式解耦。

## 交互

调 TP / PP / DP，看每卡权重、KV、层数和各项通信量怎么变。把 TP 调到 16 会触发 $H_{kv} = 8$ 的警告。切到 DeepSeek-V3 会出现 EP 旋钮和 all-to-all 那一行。

<ParallelismViz :tp="4" :pp="2" />

## 面试追问

::: details Q：为什么 TP 一定要在 NVLink 域内，PP 可以跨节点？
TP 的通信在每层的关键路径上，且通信量与 batch × seq × hidden 成正比，80 层要做 160 次 all-reduce。NVLink 900 GB/s 下每次 256 MiB 的 all-reduce 约 0.5 ms，160 次就是 80 ms；换成 100 Gbps 的 IB（12.5 GB/s）要 36 倍的时间，完全不可接受。PP 只在 stage 边界传一次激活，且能和下一个 micro-batch 的计算 overlap，所以跨节点没问题。
:::

::: details Q：推理时 TP=8 和 DP=8 怎么选？
看单卡放不放得下。放得下就 DP，吞吐线性扩展且零通信开销，还能独立扩缩容。放不下才用 TP。中间地带（勉强放得下但 KV 空间很小）要算：TP=8 让每卡 KV 空间变成 8 倍，能跑更大 batch，可能反而吞吐更高；但 TP 的 all-reduce 会拉高 TPOT。对延迟敏感选 TP（单请求更快），对吞吐敏感选 DP。
:::

::: details Q：SP 既然不增加通信量，为什么不是默认全开？
Megatron 里确实基本默认开。限制是它要求 TP > 1（SP 是依附在 TP 上的），且实现上要求序列长度能被 TP 整除，以及和某些 fused kernel、和 CP 的组合需要额外处理。推理框架里 SP 用得少，因为推理的激活本来就小，省激活显存的收益不明显。
:::

::: details Q：EP 的 all-to-all 和 TP 的 all-reduce 哪个更难优化？
all-to-all。all-reduce 的通信模式规则，ring 算法能达到带宽上界，且每个节点发送量相同。all-to-all 的消息大小取决于路由结果，运行时才知道，无法提前规划；负载不均时某些链路成为热点；而且 token 数不确定意味着难以用固定形状的 CUDA graph。DeepEP 这类库就是专门优化这个的，用 NVSHMEM 做低延迟的节点内外混合传输。
:::

## 参考

- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [Reducing Activation Recomputation (Sequence Parallel)](https://arxiv.org/abs/2205.05198)
- [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889)
