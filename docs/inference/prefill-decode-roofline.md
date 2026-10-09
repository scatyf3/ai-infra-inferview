---
title: Prefill vs Decode 与 Roofline
status: draft
tags: [roofline, arithmetic-intensity]
difficulty: 3
order: 1
related: [/inference/memory-accounting, /inference/batching-scheduling, /gpu/tensor-core-gemm, /gpu/profiling]
stack: [hw-mem, k-gemm]
---

# Prefill vs Decode 与 Roofline

感觉这部分agent写的太傻了，手动写个序，

算法侧的理解
1. prefill是训练原生的时候 模型读数据的模式，大部分llm教程写的是prefill like 的阶段
	1. prefill可以算一些有意思的loss，比如在teacher forcing序列上算ce/kl/ppl
	2. 有一部分优化就是想把decode往prefill方向转换
		1. spec decoding verify draft
		2. diffussion对一堆`[mask]`
2. decode时有kv cache之后ar生成内容的方式，偏向gemv

硬件侧的理解
1. llm一次forward必须把全部权重load到sram上 一次，这是所谓的memory bound
2. 但是搬一次权重算多少token是未定的，算一个token是memory bound，算很多token是compute bound，这里有个分界线
3. 可以在这个轴里做一些adaptive的设计，
	1. 比如single request时候的spec decoding，在这一侧可以进一步扩展并行性(Tree)，然后通过巧妙的设计把tree的side设置在2提到的焦点
	2. 或者根据batch控制sd的尺寸，小batch开sd，大batch关sd


**prefill 和 decode 是两个完全不同的负载，跑在同一块卡上。** prefill 的 arithmetic intensity 约等于序列长度（上千），落在 roofline 的右侧，compute-bound；decode 的 AI 约等于 batch size（几十），落在左侧，memory-bound。所以 prefill 的优化是「把 tensor core 喂饱」，decode 的优化是「少读字节」，两者的手段几乎没有交集。

## 推导

### 一个算子被什么卡住

一个算子要做两件事：在 SM 上算 FLOPs，从 HBM 搬字节。硬件这边对应两个规格：**peak** 是峰值算力（FLOP/s），即全部 tensor core 跑满时每秒能做的浮点运算数；**BW** 是 HBM 带宽（B/s），即每秒能在显存和芯片之间搬的字节数。H100 SXM 的 bf16 dense peak 是 989 TFLOP/s，BW 是 3.35 TB/s。注意 peak 随精度变（fp8 翻倍），而且规格表上标 "with sparsity" 的数字是 dense 的 2 倍，平时要用 dense 的值。

两件事可以重叠，所以耗时取决于更慢的那件：

$$
t \approx \max\left(\frac{\text{FLOPs}}{\text{peak}},\ \frac{\text{Bytes}}{\text{BW}}\right)
$$

哪一项更大？把两项写成不等式，两边同乘 $\text{BW} / \text{Bytes}$（正数，不等号方向不变）：

$$
\frac{\text{FLOPs}}{\text{peak}} > \frac{\text{Bytes}}{\text{BW}}
\iff
\underbrace{\frac{\text{FLOPs}}{\text{Bytes}}}_{\text{AI}} > \underbrace{\frac{\text{peak}}{\text{BW}}}_{\text{ridge}}
$$

移项之后，左边只和算子有关，右边只和硬件有关。左边叫**算术强度** AI，意思是每读 1 字节做几次运算；右边叫 **ridge point**。AI 大于 ridge 时算得更慢，算子是 compute-bound；小于 ridge 时搬字节更慢，是 memory-bound。

例：70B 模型（参数量 P = 70 × 10⁹）batch 1 的 decode，生成 1 个 token。每个参数做一次乘加，FLOPs ≈ 2P = 140 GFLOP；bf16 下每个参数 2 字节，权重整读一遍 Bytes ≈ 2P = 140 GB。在 H100 上：

- 算：140e9 ÷ 989e12 ≈ 0.14 ms
- 搬：140e9 ÷ 3.35e12 ≈ 42 ms

搬比算慢 295 倍，正好是 ridge ÷ AI = 295 ÷ 1。

把耗时换成吞吐，就是 roofline 模型给出的上界（Williams et al., 2009）：

$$
\text{attainable FLOP/s} = \min(\text{peak}, \ \text{BW} \times \text{AI})
$$

所以分析 prefill 和 decode 的步骤一样：从 shape 数出 FLOPs 和字节数，相除得到 AI，再和 ridge 比较。比较的结果决定该省算力还是省字节。

ridge 不用背，拿规格表现算就行。H100 SXM 的 bf16 dense 峰值是 989 TFLOP/s，HBM3 带宽 3.35 TB/s，所以 ridge $= 989 \div 3.35 \approx 295\ \text{FLOP/B}$。AI 低于 295 时算力闲着，高于 295 时带宽闲着。

下图是 H100 的 roofline，标出了后面要推的三个点。两个轴都取对数。斜线段是带宽上界 $\text{BW} \times \text{AI}$，水平段是算力峰值，两段的交点就是 ridge。

- **decode 的点在斜线上**：要往上走，只能加 batch（点向右移），或者换带宽更大的卡（斜线整体上移）。
- **prefill 的点在平台上**：已经顶到算力上限，再提速只能靠提高 MFU 或换更低精度。

<RooflineChart :peak="989e12" :bw="3.35e12" peak-label="H100 bf16 989 TFLOP/s" :points="[{ label: 'decode B=1', ai: 1, color: '#ef4444' }, { label: 'decode B=64', ai: 64, color: '#f59e0b' }, { label: 'prefill S=2048', ai: 2048, color: '#22c55e' }]" />

实测时怎么判断一个 kernel 落在哪一侧：ncu 的 SM Throughput 和 DRAM Throughput 分别对应 FLOPs ÷ peak 和 Bytes ÷ BW，见 [Profiling：实测判断 compute-bound 还是 memory-bound](/gpu/profiling#_2-实测判断-compute-bound-还是-memory-bound)。

### 符号和计数规则

下面推 prefill 和 decode 的 FLOPs 和字节数，用到这些符号（右列以 Llama-3-70B 为例）：

| 符号        | 含义                                                                        | Llama-3-70B                                                  |
| --------- | ------------------------------------------------------------------------- | ------------------------------------------------------------ |
| $P$       | 参数量（所有权重矩阵的元素个数之和）                                                        | $70 \times 10^9$                                             |
| $L$       | 层数                                                                        | 80                                                           |
| $d$       | hidden 维度                                                                 | 8192                                                         |
| $B$       | batch：同时处理的请求数                                                            | 看负载                                                          |
| $S$       | prefill 时每个请求的 prompt 长度（token 数）                                         | 看负载                                                          |
| $S_{ctx}$ | decode 时每个请求已有的上下文长度，也就是 KV cache 里存了多少个 token                            | 看负载                                                          |
| $b_w$     | 每个权重占几字节                                                                  | bf16 = 2，fp8 = 1，int4 = 0.5                                  |
| KV/token  | 一个 token 在所有层的 K、V 共占多少字节：$2 \cdot L \cdot H_{kv} \cdot d_h \cdot b_{kv}$ | $2 \times 80 \times 8 \times 128 \times 2 = 320\ \text{KiB}$ |

最后一行里，$H_{kv}$ 是 KV head 数，$d_h$ 是 head 维度，$b_{kv}$ 是 KV cache 每个元素的字节数，开头的 2 代表 K 和 V 两份。

计算量先按**乘加**（MAC，multiply-accumulate：一次乘法加一次累加）数，最后统一 ×2 换成 FLOP。要换是因为硬件规格的 peak 按 FLOP 标，一条 FMA 指令记 2 FLOP。分子按 MAC、peak 按 FLOP 去比，会差 2 倍。

矩阵乘 $(m \times k) \cdot (k \times n)$：输出有 $mn$ 个元素，每个元素做 $k$ 次乘加，共 $mkn$ 次 MAC，即 $2mkn$ FLOP。

对全部线性层：
线性层 $y = xW$，一个 token 经过它时，$W$ 里的每个元素恰好参与 **1 次乘加**。所有线性层的权重加起来是 $P$ 个，所以一个 token 走完整个模型约 $P$ 次 MAC，即 $2P$ FLOP。

对attention模块：
$QK^\top$ 和 $PV$ 是激活乘激活，不碰权重，所以不在 $P$ 里，要单独数。设 $H$ 个 Q head，每个 head 维度 $d_h$，$H d_h = d$；一个序列有 $S$ 个 token，看一层：

1. $QK^\top$：每个 head 是 $(S \times d_h) \cdot (d_h \times S)$，$S^2 d_h$ 次 MAC；乘 $H$ 个 head，得 $S^2 d$。
2. $PV$：每个 head 是 $(S \times S) \cdot (S \times d_h)$，同样 $S^2 d_h$；乘 $H$，得 $S^2 d$。
3. 合计每层 $2S^2 d$ 次 MAC，乘 $L$ 层、$B$ 个序列是 $2LdS^2B$ 次 MAC，即 $4 L d S^2 B$ FLOP。softmax 每个 score 只有几次运算，约是上面的 $1/d_h$，忽略不计。

几点说明：

- **GQA 不改变这个数**。K、V 是多个 Q head 共享的，但每个 Q head 仍要和 K、V 各做一次矩阵乘，所以按 $H$ 个 Q head 算。GQA 省的是 KV cache 的字节，不是 FLOPs。
- **causal mask 让实际计算减半**。每个 token 只看它前面的 token，score 矩阵只有下三角有用。FlashAttention 这类 kernel 会跳过全被 mask 的 tile，实际约 $2LdS^2B$ FLOP。不同文献的数法不统一：PaLM 算 MFU 时按不带 mask 的满额 $4LdS^2B$（Chowdhery et al., 2022，arXiv:2204.02311，Appendix B，原文写作每 token 训练 $12LHQT$，取前向的 1/3、乘 $S$ 个 token 就是 $4LdS^2$）；FlashAttention-2 测速时也按 $4 \cdot S^2 \cdot d_h \cdot H$ 计，并注明 causal 时除以 2（Dao, 2023，arXiv:2307.08691）。下文沿用满额的 $4LdS^2B$ FLOP 做上界。
- **decode 是同一个式子的特例**。新 token 只有 1 个 query，要和 $S_{ctx}$ 个已缓存的 K、V 做乘法：$QK^\top$ 是 $(1 \times d_h) \cdot (d_h \times S_{ctx})$，每层两项合计 $2 d S_{ctx}$ 次 MAC，整体 $4 L d S_{ctx} B$ FLOP。反过来，把 prefill 看成第 $t$ 个 token 看前 $t$ 个 token，对 $t = 1 \ldots S$ 求和得 $4Ld \cdot S^2/2 = 2LdS^2$ FLOP，正好是 causal 的那一半，两种数法对得上。

数字节只看 HBM 读写：权重读一遍是 $P b_w$，KV cache 读一遍是 KV/token × 序列里的 token 数。激活读写通常比这两项小得多，先忽略。

### prefill

一次处理 $S$ 个 token。FLOPs 主体是所有 GEMM，每个参数对每个 token 做一次乘加：

$$
\text{FLOPs}_{\text{prefill}} \approx 2 P B S + \underbrace{4 L d S^2 B}_{\text{attention}}
$$

访存主体是把权重读一遍：$\approx P b_w$。于是

$$
\text{AI}_{\text{prefill}} \approx \frac{2 P B S}{P b_w} = \frac{2 B S}{b_w} \sim S
$$

$S = 2048$ 时 AI 是几千，远在 ridge 右边 → **compute-bound**。结论：prefill 的时间由算力决定，TTFT $\approx$ FLOPs / (peak × MFU)。优化方向是提高 MFU：大 tile 的 GEMM、fp8 tensor core、chunked prefill 让每个 chunk 都足够大填满 SM。

注意 attention 项是 $S^2$ 的：$S$ 很长时它会反超线性项。70B、8k 时 attention 占约 15%；32k 时占约 40%。这就是长上下文 prefill 特别贵、以及 FlashAttention 对 prefill 至关重要的原因。

### decode

一次只处理 1 个 token。FLOPs：

$$
\text{FLOPs}_{\text{decode}} \approx 2 P B + 4 L d S_{ctx} B
$$

访存：权重整读一遍 **加上整个 KV cache 读一遍**：

$$
\text{Bytes}_{\text{decode}} \approx P b_w + \text{KV/token} \cdot B \cdot S_{ctx}
$$

于是

$$
\text{AI}_{\text{decode}} \approx \frac{2 P B}{P b_w + \text{KV}} \xrightarrow{\text{KV} \ll W} \frac{2B}{b_w} = B \ \ (\text{bf16})
$$

**decode 的 AI 就是 batch size**。batch 1 时 AI = 1，比 ridge 低 295 倍，意味着 tensor core 只有 0.3% 在干活，整块 H100 在等 HBM。这不是实现问题，是这个计算本身的性质：每个权重元素读进来只用一次，是 GEMV 不是 GEMM。

### 总结

prefill和decode的形态决定了两者的优化不同

|      | prefill                                        | decode                                                                  |
| ---- | ---------------------------------------------- | ----------------------------------------------------------------------- |
| 瓶颈   | 算力                                             | 带宽                                                                      |
| AI   | ≈ S（上千）                                        | ≈ B（几十）                                                                 |
| 指标   | TTFT                                           | TPOT / ITL                                                              |
| 有效手段 | fp8、大 tile GEMM、chunked prefill、FlashAttention | weight-only 量化、GQA/MLA、continuous batching 堆 batch、speculative decoding |
| 无效手段 | 堆 batch（已经饱和）                                  | 单纯换更强算力的卡                                                               |


### 两者混在一起的麻烦

prefill 和 decode 抢同一块卡。一个长 prompt 的 prefill 会把所有正在 decode 的请求卡住几百毫秒，表现为 ITL 尖刺。解法是 **chunked prefill**（把 prefill 切成小块，和 decode 拼在同一个 batch 里）或 **PD 分离**（prefill 和 decode 跑在不同的卡上，KV 通过网络传）。

想代入具体模型、GPU、batch 算 AI 和 TTFT / TPOT，用 [显存账页面的计算器](./memory-accounting)。

## 面试追问

::: details Q：batch 加到多大，decode 就 compute-bound 了？
理论上 batch ≈ ridge point ≈ 295（bf16）。但实际到不了：batch 涨的同时 KV cache 也在涨，分母的 KV 项开始主导，AI 会饱和在一个低于 batch 的值。KV 越大（长 context、MHA）饱和得越早。这正是 GQA/MLA 除了省显存以外的第二个价值：让 AI 能随 batch 涨得更久。
:::

::: details Q：MFU 和 MBU 分别是什么，各自该看哪个？
MFU = 实际 FLOP/s ÷ 峰值 FLOP/s，衡量算力利用率，prefill 看它。MBU = 实际访存带宽 ÷ 峰值带宽，decode 看它。一个良好实现的 decode 应该有 60–80% 的 MBU；如果 MBU 很低而 MFU 也很低，说明瓶颈既不是带宽也不是算力，而是 kernel launch 开销或 CPU 调度，该上 CUDA graph。实测时 ncu 的 SM Throughput、DRAM Throughput 大致对应这两个比例，见 [Profiling](/gpu/profiling#_2-实测判断-compute-bound-还是-memory-bound)。
:::

::: details Q：为什么 roofline 上 decode 的点画在斜坡上而不是峰值线上？
因为斜坡代表带宽上界。AI < ridge 时，即使算力无限，也只能达到 BW × AI 的有效算力。decode 的点落在斜坡上意味着它最好的情况就是把带宽打满，换更强算力的卡（比如同带宽但算力翻倍）对它一点帮助没有。H200 相对 H100 就是靠带宽从 3.35 涨到 4.8 TB/s 提升 decode 的，算力完全一样。
:::

::: details Q：chunked prefill 的 chunk 该设多大？
太小则每个 chunk 的 GEMM 打不满 tensor core，prefill 效率下降；太大则又会阻塞 decode，ITL 变差。实践中取 512–2048 token，让 chunk 的 AI 仍远高于 ridge 即可。vLLM 的 `max_num_batched_tokens` 就是这个旋钮，它同时决定了一个 batch 里 prefill token 加 decode token 的总预算。
:::

## 参考

- [Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785)
- [LLM Inference Performance Engineering — Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [SARATHI: chunked prefill](https://arxiv.org/abs/2308.16369)
