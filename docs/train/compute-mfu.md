---
title: 训练算力账：6ND、MFU 与训练时长
status: draft
tags: [flops, mfu, scaling-law, chinchilla, ettr]
difficulty: 3
order: 1
related: [/posttrain/training-memory, /train/parallel-config, /train/fault-tolerance, /inference/prefill-decode-roofline, /parallel/comm-overlap]
stack: []
train: [tr-j-plan]
---

# 训练算力账：6ND、MFU 与训练时长

> 一次预训练要多少 FLOPs、多少卡、多少天，以及为什么 MFU 到不了 100%

## 一句话结论

训练总算力 $C \approx 6ND$：每个参数对每个 token，forward 做 2 次 FLOP，backward 做 4 次（对激活、对权重各求一次梯度，各是一次同样大的矩阵乘）。训练时长 $= \dfrac{6ND}{n_{\text{gpu}} \cdot \text{peak} \cdot \text{MFU} \cdot \text{ETTR}}$。**MFU** 是「按模型本身需要的 FLOPs 算出的吞吐 ÷ 峰值」，recompute 不算进分子；公开的大规模 dense 训练在 30–55% 之间（Llama 3 405B 为 38–43%，作者自测）。**ETTR** 是墙钟时间里真正在推进训练的比例，管的是故障、重启、checkpoint，Llama 3 报告 >90%。Chinchilla 给出算力最优约每参数 20 个 token，但为了推理便宜，现在的小模型普遍 over-train（Llama 3 8B 约 1875 个 token / 参数）。

## 推导

### 0. 记号

| 符号 | 含义 | 例子（Llama 3 405B） |
|---|---|---|
| $N$ | 参数量（MoE 取每个 token 的**激活**参数） | 405 B $= 4.05\times10^{11}$ |
| $D$ | 训练 token 总数 | 15.6 T $= 1.56\times10^{13}$ |
| $C$ | 训练总 FLOPs | $3.8\times10^{25}$ |
| $L$ | Transformer 层数 | 126 |
| $h$ | hidden size（所有 head 的维度之和） | 16384 |
| $s$ | 序列长度 | 8192 |
| peak | 单卡峰值 FLOP/s | H100 SXM bf16 dense 989 TFLOP/s |
| $n_{\text{gpu}}$ | 卡数 | 16384 |
| $T_{\text{gpu}}$ | 每卡吞吐，token/s/GPU | |
| MFU | model FLOPs utilization | 41% |
| ETTR | effective training time ratio | >90% |

单位：1 B = $10^9$，1 T（token 数）= $10^{12}$，1 TFLOP/s = $10^{12}$ FLOP/s。H100 规格表上 bf16 写的是 1979 TFLOPS，那是带 2:4 稀疏的数，dense 是一半，989。数字出处：Llama 3 的 $L, h$ 见论文 Table 3，$N, D, C$ 见摘要和第 3 节，卡数和 MFU 见 Table 4（[arXiv 2407.21783](https://arxiv.org/abs/2407.21783)）。

### 1. 每 token 训练 FLOPs = 6N

看一个线性层 $Y = XW$，$X$ 是 $[T, k]$（$T$ 个 token），$W$ 是 $[k, m]$：

| 计算 | 公式 | FLOPs | 为什么需要 |
|---|---|---|---|
| forward | $Y = XW$ | $2Tkm$ | 每个权重对每个 token 做 1 次乘 + 1 次加 |
| backward 对激活 | $\partial X = \partial Y \, W^\top$ | $2Tkm$ | 梯度要继续往前一层传 |
| backward 对权重 | $\partial W = X^\top \partial Y$ | $2Tkm$ | 更新这一层的参数 |
| **合计** | | $6Tkm$ | |

$km$ 是这一层的参数个数，所以每个参数、每个 token 是 6 FLOPs。把所有参与矩阵乘的权重加起来就是 $N$：

$$
\text{每 token 训练 FLOPs} \approx 6N, \qquad C \approx 6ND
$$

自己算一遍：

1. 7B 训 1T token：$6 \times 7\times10^9 \times 10^{12} = 4.2\times10^{22}$ FLOPs。
2. Llama 3 405B 训 15.6T token：$6 \times 4.05\times10^{11} \times 1.56\times10^{13} = 3.79\times10^{25}$，和论文写的 $3.8\times10^{25}$ 对得上。

**attention 项**。$6N$ 只数了「激活 × 权重」的矩阵乘。$QK^\top$ 和 $PV$ 是激活乘激活，不碰权重，要另算。一层里一个 token 的 query 要和 $s$ 个 key 做点积（$2sh$），再对 $s$ 个 value 加权求和（$2sh$），forward 共 $4sh$；backward 同样是两倍，合计 $12sh$。乘层数（[PaLM 附录 B](https://arxiv.org/abs/2204.02311) 写作 $6LH(2QT)$，$H$ 个 head、每个 head 维度 $Q$，$HQ = h$，$T$ 是序列长度）：

$$
\text{每 token 训练 FLOPs} = 6N + 12Lhs, \qquad \frac{\text{attention 项}}{6N} = \frac{2Lhs}{N}
$$

这里按不带 causal mask 的满额算，这是 PaLM 定义 MFU 时的口径，报 MFU 用它。causal mask 下每个 query 平均只看 $s/2$ 个 key，GPU **实际执行**的是一半，$6Lhs$；估长序列实际耗时用这个，[长上下文训练](/train/long-context) 就按 $6Lhs$ 算。代 Llama 3 的结构（Table 3）：

| 模型 | $L$ | $h$ | $s$ | $2Lhs/N$ |
|---|---|---|---|---|
| 405B | 126 | 16384 | 8192 | 8.4% |
| 405B | 126 | 16384 | 131072 | 134% |
| 70B | 80 | 8192 | 8192 | 15.3% |
| 8B | 32 | 4096 | 8192 | 26.8% |

结论：模型越小、序列越长，attention 项越大。标准结构每层参数约 $12h^2$（attention $4h^2$ + MLP $8h^2$），$N \approx 12Lh^2$，比值化简成 $s / (6h)$：**$s$ 到 $6h$ 时 attention 项和 $6N$ 一样大**。8k 训练时大模型可以忽略，128k 的长上下文阶段一定要算上。

### 2. MFU 与 HFU

[PaLM](https://arxiv.org/abs/2204.02311)（第 4.1 节、附录 B）的定义：

$$
\text{MFU} = \frac{\text{实测 token/s} \times (6N + 12Lhs)}{n_{\text{gpu}} \cdot \text{peak}}
$$

| 指标 | 分子 | recompute 算不算 |
|---|---|---|
| MFU（model FLOPs utilization） | 模型**需要**的 FLOPs：$6N + 12Lhs$ 每 token | 不算 |
| HFU（hardware FLOPs utilization） | 硬件**实际执行**的 FLOPs | 算 |

PaLM 提出 MFU 的理由：recompute 多做的 FLOPs 是拿算力换显存，目标是 token/s 而不是做更多 FLOPs。HFU 会因为 recompute 变好看，MFU 只看 token/s 和模型结构，不同系统之间能直接比。

例：开 full recompute，每 token 硬件实际做 $8N$（多一次 forward，见 [训练显存账](/posttrain/training-memory)），所以 $\text{HFU} = \frac{8}{6}\text{MFU}$。PaLM 540B 只给部分激活做了 rematerialization：MFU 46.2%，HFU 57.8%（作者自测）。

**token/s 和 MFU 互换**（忽略 attention 项）：

$$
\text{MFU} = \frac{T_{\text{gpu}} \cdot 6N}{\text{peak}}, \qquad T_{\text{gpu}} = \frac{\text{MFU} \cdot \text{peak}}{6N}
$$

1. 70B 在 H100 上实测每卡 1000 token/s：$1000 \times 6 \times 7\times10^{10} / 9.89\times10^{14} = 42.5\%$。
2. 反过来，7B、MFU 40%：$0.4 \times 9.89\times10^{14} / (6 \times 7\times10^9) \approx 9400$ token/s/GPU。

### 3. 算一笔：Llama 3 405B 要训多少天

$$
t = \frac{6ND}{n_{\text{gpu}} \cdot \text{peak} \cdot \text{MFU}}
$$

代入 Table 4 的 16K 卡配置（DP = 128，MFU 41%）：

1. $C = 3.79\times10^{25}$ FLOPs。
2. 集群有效算力：$16384 \times 989\times10^{12} \times 0.41 = 6.64\times10^{18}$ FLOP/s。
3. $t = 3.79\times10^{25} / 6.64\times10^{18} = 5.71\times10^6$ s $\approx$ **66 天**；再除以 ETTR 0.9 是 **73 天**。
4. 换成 GPU 小时：$C / (\text{peak} \cdot \text{MFU}) / 3600 = 2.60\times10^7$，即 **26.0 M**；除以 0.9 是 28.9 M。

对照 [model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)：405B 的 Training Time 是 **30.84 M** GPU 小时。反推 $\text{MFU} \times \text{ETTR} = 3.79\times10^{25} / (3.084\times10^7 \times 3600 \times 989\times10^{12}) = 34.5\%$。比 $41\% \times 0.9 = 37\%$ 低一点，差在：

1. 不是全程都在 41%：Table 4 里 128k 长上下文那段只有 38%，而且这一段约 800B token，attention 项（上表 134%）完全没进 $6ND$。
2. 前期 batch 小：论文第 3.4.1 节，先用 4M token 的 batch、序列 4096，再到 8M、16M，batch 小时每卡效率更低。
3. model card 只说「训练每个模型需要的总 GPU 时间」，没写包含哪些阶段，这部分拆不开。

面试里能估到同一个数量级、讲清楚差在哪就够了。

### 4. Chinchilla：$D/N \approx 20$

[Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556) 训了 400 多个模型（70M 到 16B 参数，5B 到 500B token），问：给定 $C$，$N$ 和 $D$ 怎么分最好。用三种方法拟合 $N_{\text{opt}} \propto C^a$、$D_{\text{opt}} \propto C^b$，得到 $a \approx b \approx 0.5$（Table 2：0.50/0.50、0.49/0.51、0.46/0.54，作者自测）。$a = b$ 意味着 $N$ 和 $D$ 同比例放大，$D/N$ 是常数；Table 3 的外推里 1B 配 20.2B token、10B 配 205.1B token，就是 **约 20 token / 参数**。

给定 $C$ 求最优：把 $D = 20N$ 代入 $C = 6ND$：

$$
C = 120N^2 \ \Rightarrow\ N_{\text{opt}} = \sqrt{C / 120}, \quad D_{\text{opt}} = 20N_{\text{opt}}
$$

1. Gopher 的预算 $C = 5.76\times10^{23}$：$N_{\text{opt}} = \sqrt{4.8\times10^{21}} = 69$ B，$D_{\text{opt}} = 1.39$ T。这正是 Chinchilla（70B、1.4T token），同样算力下全面超过 280B 的 Gopher（作者自测）。
2. Llama 3 的预算 $3.8\times10^{25}$：按 20 倍算是 563B、11.3T。Llama 3 在自己数据上重新拟合的结果是 402B、16.55T（论文第 3.2.1 节），约 41 token / 参数。20 是 Chinchilla 那套数据和结构下的值，换数据要重新拟合。

**为什么现在都 over-train**：Chinchilla 只最小化**训练**算力。模型上线后每生成一个 token 还要约 $2N$ FLOPs，总成本是

$$
6ND + 2N D_{\text{infer}}
$$

$D_{\text{infer}}$ 是一生要服务的 token 数，往往远大于 $D$。小一点的模型多训一些 token，训练多花一点，每次推理都省。例：Llama 3 8B 训了 15T token（model card 写 15T+），$15\times10^{12} / 8\times10^9 = 1875$ token / 参数，是 Chinchilla 最优（160B token）的约 94 倍。Llama 3 论文第 1 节的原话：「we also train our smaller models for much longer than is compute-optimal」，这样的模型在同样的推理预算下比算力最优的模型更好。

### 5. MFU 为什么到不了 100%

公开的数字（全部作者自测）：

| 训练 | 硬件 | MFU | 出处 |
|---|---|---|---|
| GPT-3 175B | V100 | 21.3% | PaLM Table 3 |
| Gopher 280B | 4096 TPU v3 | 32.5% | PaLM Table 3 |
| MT-NLG 530B | 2240 A100 | 30.2% | PaLM Table 3 |
| PaLM 540B | 6144 TPU v4 | 46.2% | PaLM Table 3 |
| MegaScale 175B | 12288 Ampere GPU | 55.2% | [arXiv 2402.15627](https://arxiv.org/abs/2402.15627) |
| Llama 3 405B | 8K–16K H100 | 38–43% | Llama 3 Table 4 |

一个 step 的时间里，下面这些都在分母里、却不产生分子：

1. **非 GEMM 算子**：RMSNorm、softmax、SwiGLU 的激活、RoPE、残差加、optimizer step。它们是 memory-bound，要花时间，FLOPs 却不在 $6N$ 里。看 profiler 里非 GEMM kernel 的时间占比。
2. **小 GEMM**：TP 把每个矩阵切成 $1/t$，micro-batch 又小，GEMM 太小喂不满 SM（见 [Tensor Core GEMM](/gpu/tensor-core-gemm)）。Llama 3 从 8K 卡（DP = 64）到 16K 卡（DP = 128）时 global batch 不变，每个 DP 组的 batch 从 32 减到 16，MFU 从 43% 掉到 41%，论文就是这么归因的。
3. **没藏住的通信**：TP 的 all-reduce 在关键路径上，DP 的梯度同步、PP 的 P2P、MoE 的 all-to-all 没和计算重叠的部分。看时间线上 NCCL kernel 单独跑、计算 stream 空着的时段，见 [计算通信 overlap](/parallel/comm-overlap)。
4. **PP bubble**：interleaved 1F1B 下 bubble 占比 $\frac{p-1}{vm}$（$p$ 个 stage，每卡 $v$ 段，每步 $m$ 个 micro-batch，Llama 3 第 3.3.2 节）。$p = 8$、$m = 32$、$v = 1$ 时 22%，$v = 4$ 时 5.5%。各种调度见 [Pipeline 并行调度](/train/pipeline-schedules)。
5. **recompute**：多算的那次 forward 不进 MFU 分子。full recompute 下每 token 做 $8N$、只记 $6N$，就算硬件 100% 忙，MFU 上限也只有 75%。
6. **dataloader 和 host 端**：取数据、Python 开销、同步点让 GPU 空等，时间线上表现为 kernel 之间的空隙。

还有一类损失不在 MFU 里：MFU 是在稳定运行时测的，**故障、重启、写 checkpoint** 的时间要另算，这就是 ETTR。

#### ETTR：墙钟时间有多少在推进训练

[Kokolis et al. 2024](https://arxiv.org/abs/2410.21680) 的定义：ETTR = 有效运行时间 ÷ 可用墙钟时间。不算有效的三类时间：

1. 从上一个 checkpoint 到故障之间算过的 step，要重算。
2. 重启开销：重新调度、加载 checkpoint、初始化通信。
3. 写 checkpoint 时训练停下来的时间。

所以真正的训练时长是 $\dfrac{6ND}{n_{\text{gpu}} \cdot \text{peak} \cdot \text{MFU} \cdot \text{ETTR}}$。

规模越大故障越频繁。Llama 3 在 54 天的统计窗口里有 466 次中断，其中 419 次是意外，约 78% 和硬件有关，平均约 3 小时一次；最终 ETTR 仍然 >90%（第 3.3.4 节，作者自测）。手段是缩短启动和 checkpoint 的时间、加快故障诊断。

ETTR 怎么从故障率、checkpoint 写入时间、重启时间算出来，最优 checkpoint 间隔怎么定，见 [大规模训练的容错](/train/fault-tolerance)。本页只用结论：估训练时长时，在 MFU 之外再乘一个 ETTR，万卡规模取 0.9 左右。

## 面试追问

::: details Q：为什么 backward 的 FLOPs 是 forward 的两倍？
线性层 $Y = XW$ 的 forward 是一次矩阵乘。backward 要算两个梯度：$\partial X = \partial Y\,W^\top$ 往前一层传，$\partial W = X^\top \partial Y$ 用来更新参数，形状和 forward 一样大，各是一次同样 FLOPs 的矩阵乘。所以 forward $2N$、backward $4N$，合计 $6N$。第一层的 $\partial X$ 其实不用算，对深层网络可以忽略。
:::

::: details Q：开 full recompute 后 MFU 和 HFU 怎么变？
每个 step 多跑一次 forward，时间约变成 $8/6$ 倍。MFU 的分子（token/s × 6N）按比例降，MFU 约降到原来的 75%；HFU 把重算的 FLOPs 也算进去，基本不变。所以显存够的时候应关掉 recompute，MFU 直接涨。
:::

::: details Q：MoE 的 6ND 里 N 用总参数还是激活参数？
激活参数，因为每个 token 只经过被选中的专家。DeepSeek-V3 总参数 671B、激活 37B，报告每 1T token 需要 180K H800 GPU 小时（[arXiv 2412.19437](https://arxiv.org/abs/2412.19437)）。按 $6 \times 3.7\times10^{10} \times 10^{12} / (1.8\times10^5 \times 3600)$ 折算，每卡约 343 TFLOP/s（本页推算，没算 attention 和 MTP 模块）。MoE 的 MFU 通常比同激活参数的 dense 低：专家 GEMM 小、all-to-all 难藏、负载不均，见 [MoE 与 EP](/parallel/moe-ep)。
:::

::: details Q：为什么 Llama 3 从 8K 卡加到 16K 卡，MFU 从 43% 掉到 41%？
global batch 固定在 16M token，DP 从 64 翻到 128，每个 DP 组分到的序列从 32 条减到 16 条（Table 4）。每个 pipeline 一步能切的 micro-batch 变少，bubble 占比 $\frac{p-1}{vm}$ 变大，每张卡的计算也更难盖住通信。论文把这 2 个点归因于「每个 DP 组的 batch 更小」。
:::

::: details Q：Chinchilla 的 20 是常数吗？
不是。它来自 Chinchilla 的数据和结构下拟合出的 $a \approx b \approx 0.5$ 和系数。Llama 3 用自己的数据重新拟合，$3.8\times10^{25}$ 的预算给出 402B、16.55T，约 41 token / 参数。而且它只回答「给定训练算力，loss 最低怎么分」，不管推理成本。
:::

::: details Q：训练报 MFU 用的 peak 取哪个？
取那块卡在**训练所用精度**下的 dense 峰值，H100 bf16 是 989 TFLOP/s。不要用规格表上带稀疏的数，也不要拿 bf16 训练去比 fp8 的峰值。
:::

## 手撕

```python
import math

def estimate_training_days(N, D, n_gpu, peak, mfu, ettr=1.0):
    """N 参数量（MoE 用激活参数），D token 数，peak 单卡峰值 FLOP/s。"""
    C = 6 * N * D                                  # forward 2N + backward 4N，每 token
    seconds = C / (n_gpu * peak * mfu * ettr)
    days = seconds / 86400
    gpu_hours = seconds * n_gpu / 3600
    return days, gpu_hours

def chinchilla_opt(C):
    N = math.sqrt(C / 120)                         # C = 6ND，D = 20N
    return N, 20 * N

H100 = 989e12
d, gh = estimate_training_days(405e9, 15.6e12, 16384, H100, 0.41)
print(f"{d:.1f} days, {gh / 1e6:.1f}M GPU-h")      # 66.0 days, 26.0M GPU-h
d, gh = estimate_training_days(405e9, 15.6e12, 16384, H100, 0.41, ettr=0.9)
print(f"{d:.1f} days, {gh / 1e6:.1f}M GPU-h")      # 73.4 days, 28.9M GPU-h
d, _ = estimate_training_days(7e9, 2e12, 512, H100, 0.40)
print(f"{d:.1f} days")                              # 4.8 days
N, D = chinchilla_opt(5.76e23)                      # Gopher 的预算
print(f"{N / 1e9:.0f}B params, {D / 1e12:.2f}T tokens")  # 69B params, 1.39T tokens
```

常见变体：给定预算和天数反求要多少卡；给 token/s 求 MFU；问 128k 上下文阶段要不要把 attention 项加进去（用 $2Lhs/N$ 判断）。

## 参考

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)：Table 3 结构，Table 4 并行配置与 MFU，第 3.3.4 节可靠性与 ETTR
- [Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)：各尺寸的训练 GPU 小时
- [PaLM](https://arxiv.org/abs/2204.02311)：第 4.1 节和附录 B，MFU / HFU 定义与 $6N + 12LHQT$
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)（Kaplan et al. 2020）：$6N$ 的来源
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)（Chinchilla）
- [MegaScale](https://arxiv.org/abs/2402.15627)：12288 卡上 55.2% MFU
- [Revisiting Reliability in Large-Scale Machine Learning Research Clusters](https://arxiv.org/abs/2410.21680)：ETTR 的定义
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
