---
title: LoRA / QLoRA 原理与显存账
status: draft
tags: [lora, qlora]
difficulty: 3
order: 2
related: [/posttrain/training-memory, /posttrain/sft, /inference/quantization, /inference/memory-accounting, /inference/batching-scheduling, /parallel/zero-fsdp]
stack: [f-runner]
---

# LoRA / QLoRA 原理与显存账

> 低秩更新怎么省显存、省在哪、没省在哪；QLoRA 的 NF4；多 LoRA serving

## 一句话结论

LoRA（[Hu et al. 2021](https://arxiv.org/abs/2106.09685)）冻结原权重 $W_0$，只训练一对低秩矩阵 $B, A$（秩 $r$ 通常 8–64），forward 变成 $W_0x + \frac{\alpha}{r}BAx$。可训练参数从 $dk$ 降到 $r(d+k)$，梯度和优化器状态跟着缩小上百倍；**激活不省**，backward 仍要穿过每一层。7B 全参微调模型状态 112 GB，LoRA 后约 14.6 GB（大头是 bf16 底座 14 GB）；QLoRA（[Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)）把冻结底座存成 4 bit NF4，70B 底座从 138 GB 压到约 36 GB，一张 48 GB 卡就能微调。推理时单个 adapter 可以合并回 $W_0$ 零开销；多个 adapter 同时服务时不合并，用 Punica / S-LoRA 的分段 batched kernel 共享一份底座。

## 推导

### 记号与约定

一个线性层 $h = W_0 x$，$x \in \mathbb{R}^{k}$ 是输入（$k$ = 输入维），$h \in \mathbb{R}^{d}$ 是输出（$d$ = 输出维），$W_0 \in \mathbb{R}^{d \times k}$，和 `nn.Linear(k, d).weight` 的形状一致（PyTorch 存 `[out, in]`）。

| 符号 | 含义 |
|---|---|
| $r$ | LoRA 秩，$r \ll \min(d, k)$ |
| $A \in \mathbb{R}^{r \times k}$ | 降维矩阵（"shrink"），先把 $x$ 投到 $r$ 维 |
| $B \in \mathbb{R}^{d \times r}$ | 升维矩阵（"expand"），再投回 $d$ 维 |
| $\alpha$ | 缩放超参，实际乘的是 $s = \alpha / r$ |
| $\Psi$ | 模型总参数量 |

单位：GB = $10^9$ 字节，和 [训练显存账](/posttrain/training-memory) 一致。

### 1. LoRA 是什么：把权重更新限制成低秩

全参微调学的是 $W = W_0 + \Delta W$，$\Delta W$ 有 $dk$ 个自由度。LoRA 的假设是**微调需要的更新本身是低秩的**，所以直接把 $\Delta W$ 参数化成两个小矩阵之积：

$$
h = W_0 x + \Delta W x = W_0 x + s \cdot B A x, \qquad s = \frac{\alpha}{r}
$$

这个假设的依据：[Aghajanyan et al. 2020](https://arxiv.org/abs/2012.13255) 发现预训练模型微调时只需在一个很低维的子空间里优化：RoBERTa 在 MRPC 上只训练 200 个参数（随机投影回全参数空间）就能达到全参微调 90% 的效果（作者自测）；LoRA 论文 Sec. 7 也观察到学到的 $\Delta W$ 有效秩很低（作者自测）。

计算顺序很重要：先算 $Ax$（$r$ 维），再乘 $B$，**从不显式构造 $d \times k$ 的 $BA$**。每 token 多出的 FLOPs 是 $2r(k + d)$，相对原来的 $2dk$：$d = k = 4096$、$r = 16$ 时是 $2 \cdot 16 \cdot 8192 / (2 \cdot 4096^2) = 0.78\%$。

### 2. 初始化：A 随机、B 全零

LoRA 论文用高斯随机初始化 $A$、$B = 0$（HF PEFT 默认对 $A$ 用 Kaiming uniform，同一个意思）。这样第 0 步 $BA = 0$，模型输出和原模型**完全相同**，微调从预训练点出发。

为什么不能两个都是 0？记 $g = \partial L / \partial h \in \mathbb{R}^d$，对单个样本：

$$
\frac{\partial L}{\partial B} = s \cdot g\,(Ax)^\top, \qquad \frac{\partial L}{\partial A} = s \cdot B^\top g\, x^\top
$$

- $B = 0$、$A$ 随机：$\partial L/\partial A = 0$，但 $Ax \ne 0$，所以 $\partial L/\partial B \ne 0$。第一步 $B$ 先动起来，之后 $A$ 的梯度也不再是 0。
- $A = B = 0$：两个梯度都是 0，永远停在原点。
- 反过来 $A = 0$、$B$ 随机也能起步，只是惯例是前者。

**缩放 $\alpha / r$**：LoRA 论文的说法是用 Adam 时调 $\alpha$ 约等于调学习率，所以固定 $\alpha$（比如等于第一次试的 $r$），换 $r$ 时不用重调 lr。[rsLoRA](https://arxiv.org/abs/2312.03732) 指出在 $\alpha/r$ 下 $r$ 变大时更新会被压得过小、大秩学不动，改用 $\alpha / \sqrt{r}$（作者自测）。PEFT 里是 `use_rslora=True`。

### 3. 可训练参数：算一遍 7B 和 70B

一个 $d \times k$ 矩阵加 LoRA，多 $r(d + k)$ 个参数。

**Llama-2-7B**：$h = 4096$，MLP 中间维 $i = 11008$，32 层，MHA（K/V 也是 $4096 \times 4096$），词表 32000。线性层参数 6.48B，加 embedding 和 lm_head 共约 6.74B。

| 加在哪 | 每层 LoRA 参数（$r=16$） | ×32 层 | 占比 |
|---|---|---|---|
| 单个 $4096 \times 4096$ 矩阵 | $16 \times 8192 = 131{,}072$ | — | 该矩阵的 0.78% |
| $W_q, W_v$（LoRA 论文的默认） | $2 \times 131{,}072 = 262{,}144$ | 8.4M | 0.12% |
| $W_q, W_k, W_v, W_o$ | $524{,}288$ | 16.8M | 0.25% |
| 全部 7 个线性层（+ gate/up/down，各 $16 \times (4096 + 11008)$） | $524{,}288 + 3 \times 241{,}664 = 1{,}249{,}280$ | 40.0M | 0.59% |

**Llama-2-70B**：$h = 8192$，$i = 28672$，80 层，GQA 8 个 KV head（$W_k, W_v$ 是 $1024 \times 8192$），线性层约 68.5B，总约 69B。全部 7 个线性层、$r = 16$：

- attention：$W_q, W_o$ 各 $16 \times 16384$，$W_k, W_v$ 各 $16 \times 9216$，合计 819,200
- MLP：$3 \times 16 \times (8192 + 28672) = 1{,}769{,}472$
- 每层 2,588,672，×80 = **207M，占 0.30%**

[QLoRA 论文](https://arxiv.org/abs/2305.14314) 的一个结论是：要追平 16 bit 全参微调，**LoRA 必须加在所有线性层上**，$r$ 本身反而不太敏感（作者自测）。所以下面的显存账按「全部线性层」算。

### 4. 显存账：省了什么、没省什么

训练显存 = 模型状态 + 激活（推导见 [训练显存账](/posttrain/training-memory)）。全参混合精度 Adam 每参数 16 字节（bf16 参数 2 + bf16 梯度 2 + fp32 主权重 4 + Adam $m, v$ 各 4）。LoRA 改的是**模型状态**：

- 冻结的底座：只存 bf16 权重，2 字节/参数。没有梯度、没有主权重、没有优化器状态。
- adapter：照常 16 字节/参数的全套。

$$
M_{\text{state}}^{\text{LoRA}} = 2\Psi_{\text{base}} + 16\Psi_{\text{lora}}
$$

| | 7B（$\Psi \approx 7\times10^9$，取整同训练显存账页） | 70B（$\Psi \approx 69\times10^9$） |
|---|---|---|
| 全参，16 B/参数 | 112 GB | 1104 GB（≥14 张 80 GB 卡，还没算激活） |
| LoRA，底座 bf16 + adapter（全线性层 $r=16$） | 14 + 40M×16 = 14 + 0.64 = **14.6 GB** | 138 + 207M×16 = 138 + 3.3 = **141 GB** |
| QLoRA，底座 NF4 + adapter | 3.9 + 0.64 ≈ **4.5 GB** | 36.4 + 3.3 ≈ **40 GB** |

QLoRA 底座怎么算见第 6 节。

**没省的：激活。** backward 要把梯度从 loss 一路传回最底层的 adapter，每一层都得做 $\partial L/\partial x = W_0^\top g$；而且 $\partial L / \partial A$ 需要每个 LoRA 层的输入 $x$，非线性（SwiGLU、softmax）的反传也需要它们的输入。所以激活基本和全参一样：7B、seq 4096、micro-batch 1、FlashAttention 下约 18 GB，比 LoRA 的模型状态还大。要省激活还是得靠 gradient checkpointing（降到约 1 GB）和减小 micro-batch，见 [训练显存账](/posttrain/training-memory)。

小例外：没挂 LoRA 的冻结线性层，它的输入本来只为算 $\partial L/\partial W$ 而存，冻结后可以不存。但这类张量在 34 系数里只占一部分，量级不变。

**计算省了约 1/3。** 训练每 token 约 $6\Psi$ FLOPs：forward $2\Psi$，backward 对输入求梯度 $2\Psi$、对权重求梯度 $2\Psi$。LoRA 冻结底座，**对权重的那 $2\Psi$ 不用算**，剩约 $4\Psi$，理论上快 1.5 倍；开 full checkpointing 后是 $6\Psi$ 对 $8\Psi$。LoRA 论文在 GPT-3 175B 上报告训练吞吐提升 25%、显存从 1.2 TB 降到 350 GB（作者自测）。

**多卡也一样省通信。** 数据并行只 all-reduce adapter 的梯度：7B 全线性层是 40M 个参数，bf16 下 80 MB，对比全参的 14 GB。

### 5. 合并：部署单个 adapter 零开销

训练完把低秩更新加回底座：

$$
W' = W_0 + s \cdot BA
$$

$W'$ 和 $W_0$ 形状相同，推理就是普通模型，**没有额外延迟**（LoRA 论文强调这是相对 Adapter 方法的优势，后者插入串行的小层）。可以反过来 $W_0 = W' - sBA$ 卸载，但 bf16 下一加一减会有舍入误差，切换频繁时应该保留原始 $W_0$ 而不是来回减。

QLoRA 的坑：底座是 NF4，$BA$ 是 bf16，直接合并要先把 $W_0$ 反量化成 bf16、加上 $sBA$，得到的是一个 **bf16 模型**；如果再量化回 4 bit，新的量化误差和训练时看到的不一样，效果可能掉。常见做法是合并成 bf16 后再用 GPTQ / AWQ 等重新量化并评估，见 [量化](/inference/quantization)。

### 6. QLoRA：把冻结底座压到 4 bit

QLoRA 的思路：底座既然冻结，就不需要高精度存储，只要 forward / backward 时能还原出近似的 bf16 权重。三件事：

**(a) NF4（4-bit NormalFloat）。** 预训练权重近似服从均值 0 的正态分布。把一个块（64 个权重）除以块内 $\max|w|$ 归一化到 $[-1, 1]$ 后，NF4 的 16 个格点取在**标准正态分布的分位数**上，让每个格点覆盖的概率质量大致相等（论文称对正态分布信息论最优）。为了能精确表示 0，负半轴取 8 个分位数、正半轴取 9 个，合并两个 0，再归一化。按 bitsandbytes 的构造自己算一遍得到：

```text
-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0910, 0.0,
 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
```

可以看到格点在 0 附近密、两端稀，和权重的分布形状吻合。对比 INT4 是等间距格点，见 [量化](/inference/quantization) 的「格式」一节。

```python
from statistics import NormalDist
N = NormalDist()
p_hi = 0.9677083                       # 不能取 1（分位数为 ∞），取一个接近 1 的端点
lin = lambda a, b, n: [a + (b - a) * t / (n - 1) for t in range(n)]
pos = [N.inv_cdf(p) for p in lin(p_hi, 0.5, 9)[:-1]]      # 8 个正格点
neg = [-N.inv_cdf(p) for p in lin(p_hi, 0.5, 8)[:-1]]     # 7 个负格点
code = sorted(pos + [0.0] + neg)
code = [c / max(code) for c in code]                      # 归一化到 [-1, 1]，共 16 个
```

**(b) Double quantization。** 每 64 个权重一个 fp32 的 absmax 常数，摊下来 $32 / 64 = 0.5$ bit/参数，对 4 bit 来说是 12.5% 的额外开销。QLoRA 把这些常数**再量化一次**：常数本身按 256 个一块量化成 8 bit float，第二级常数仍用 fp32。开销变成

$$
\frac{8}{64} + \frac{32}{64 \times 256} \approx 0.127 \ \text{bit/参数}
$$

每参数省 0.373 bit，65B 模型约省 3 GB（作者自测数字与此一致）。

**(c) Paged optimizer。** 长序列的 batch 会让显存瞬间冲高导致 OOM。QLoRA 用 NVIDIA unified memory 把优化器状态放在可分页内存里：显存不够时由驱动自动换出到 CPU 内存，用到时再换回。bitsandbytes 里是 `PagedAdamW` / `PagedAdamW8bit`。它解决的是**峰值**，不降低平均占用。

**底座字节数怎么算**（只有 `nn.Linear` 被换成 4 bit，embedding 和 lm_head 保持 bf16）：

| | 线性层 × (4 + 0.127) / 8 | embedding + lm_head（bf16） | 合计 |
|---|---|---|---|
| 7B | $6.48\text{B} \times 0.516 = 3.34$ GB | $0.26\text{B} \times 2 = 0.52$ GB | **3.9 GB** |
| 70B | $68.45\text{B} \times 0.516 = 35.3$ GB | $0.52\text{B} \times 2 = 1.05$ GB | **36.4 GB** |

70B 底座 36.4 GB + adapter 状态 3.3 GB ≈ 40 GB，48 GB 卡还剩约 8 GB 给激活，所以必须配 gradient checkpointing。论文的说法是 65B 模型可以在单张 48 GB 卡上微调并保持 16 bit 微调的效果（作者自测）。

**计算怎么走。** 存储类型是 NF4，**计算类型是 bf16**：

$$
h = \text{dequant}(W_0^{\text{NF4}}, c_1, c_2)\,x + s \cdot B A x
$$

每次 forward 和 backward 都要把用到的块反量化成 bf16 再做 GEMM，不写回 HBM。梯度只流向 $A, B$，对 $W_0$ 只算 $\partial L/\partial x$。反量化是额外工作，所以 QLoRA 每步比 bf16 LoRA 慢（幅度看 kernel 实现，这里不给数字）；它是拿时间换显存。

### 7. LoRA 的效果边界

[LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) 在代码、数学的继续预训练和指令微调上对比：LoRA 明显不如全参（尤其数据量大的继续预训练），但对底座原有能力的遗忘更少；全参学到的 $\Delta W$ 秩比常用 LoRA 配置高 10–100 倍（均为作者自测）。经验上：风格、格式、领域适配用 LoRA 足够；要灌大量新知识用全参。

### 8. 多 LoRA serving

场景：一个底座，上百个客户各自一个 adapter。如果每个 adapter 合并成独立模型，7B 就是每个 14 GB；不合并时一个 adapter（全线性层 $r=16$）只有 $40\text{M} \times 2 = 80$ MB。

一个 batch 里不同请求用不同 adapter，第 $t$ 个 token 属于 adapter $a(t)$：

$$
y_t = W_0 x_t + s_{a(t)} B_{a(t)} A_{a(t)} x_t
$$

第一项所有请求共享，照常一个大 GEMM；难点在第二项。朴素写法：

```python
# x: [T, k] 本步 batch 拼平后的所有 token；adapter_id: [T]，-1 表示不用 LoRA
y = x @ W0.T                                   # 共享底座：一个大 GEMM
for a in adapter_id.unique():                  # 每个 adapter 一组小 GEMM
    if a < 0: continue
    idx = (adapter_id == a).nonzero().squeeze(1)
    y[idx] += s[a] * (x[idx] @ A[a].T) @ B[a].T   # [n_a, k]→[n_a, r]→[n_a, d]
```

问题：batch 里有 32 个不同 adapter 就要 64 次小 kernel launch，每个都很瘦（$r = 16$），GPU 利用率很低。

**Punica 的 SGMV**（[Chen et al. 2023](https://arxiv.org/abs/2310.18547)）：先把 token 按 adapter 排序，让同一 adapter 的 token 连续，形成若干段（segment），**一个 kernel 处理所有段**，每个 thread block 根据段号去取对应 adapter 的 $A$ 或 $B$：

```python
# 排序后：seg_start = [0, 5, 7, 12, ...]，第 g 段的 token 都用 adapter seg_adapter[g]
def sgmv(x, W_stack, seg_start, seg_adapter):   # W_stack: [n_adapters, out, in]
    out = empty(x.shape[0], W_stack.shape[1])
    for g in parallel(range(len(seg_adapter))):   # 一个 kernel 内并行，不是 Python 循环
        lo, hi = seg_start[g], seg_start[g + 1]
        out[lo:hi] = x[lo:hi] @ W_stack[seg_adapter[g]].T
    return out

v = sgmv(x, A_stack, seg_start, seg_adapter)       # shrink：[T, k] → [T, r]
y += scale_per_token[:, None] * sgmv(v, B_stack, seg_start, seg_adapter)   # expand
```

论文报告相对当时的 LLM serving 系统吞吐提升 12 倍，每 token 只增加约 2 ms 延迟（作者自测）。

**S-LoRA**（[Sheng et al. 2023](https://arxiv.org/abs/2311.03285)）在此基础上解决「adapter 太多放不下」：所有 adapter 存在 CPU 内存，只把当前 batch 用到的拷到 GPU；**Unified Paging** 让不同秩的 adapter 权重和 KV cache 共用同一个分页内存池（和 [PagedAttention](/inference/kv-cache-paged-attention) 同一思路），减少碎片。论文报告单卡可服务上千个 adapter，吞吐比 HF PEFT 和朴素支持 LoRA 的 vLLM 高最多 4 倍（作者自测）。

**自己估开销：**
- 换入：80 MB 的 adapter 走 PCIe（按 25 GB/s 估）约 3.2 ms，和一步 decode 同量级，所以要提前预取、把同 adapter 的请求攒在一起调度。
- decode 带宽：decode 是 memory-bound，每步要把底座读一遍（7B bf16 约 13.5 GB），每个**出现在 batch 里的不同** adapter 也要读一遍。batch 里有 32 个不同 adapter：$32 \times 80\text{ MB} = 2.56$ GB，额外读约 19%，单步 decode 时间按带宽也涨约 19%。所以 vLLM 用 `--max-loras` 限制一个 batch 里同时出现的 adapter 数，`--max-lora-rank` 决定预分配的槽大小。roofline 见 [Prefill vs Decode](/inference/prefill-decode-roofline)。

## 面试追问

::: details Q：LoRA 为什么省不了激活显存？
反向传播要把梯度从 loss 一路传回最底层的 adapter，每层都要算 $\partial L/\partial x$；$\partial L/\partial A = s B^\top g x^\top$ 需要该层输入 $x$，非线性层反传需要它们的输入，这些激活和全参微调时一样要存。省下来的只是参数梯度、主权重和优化器状态。7B、seq 4k 时激活约 18 GB（FlashAttention），比 LoRA 的模型状态 14.6 GB 还大，要靠 gradient checkpointing，见 [训练显存账](/posttrain/training-memory)。
:::

::: details Q：B 初始化成 0，那 B 的梯度会不会也一直是 0？
不会。$\partial L/\partial B = s\, g (Ax)^\top$，只要 $A$ 随机、$x \ne 0$，$Ax \ne 0$，$B$ 的梯度就非零。是 $A$ 的梯度 $sB^\top g x^\top$ 在第 0 步为零，等 $B$ 更新一步后才有。两个都初始化为 0 才会永远学不动。
:::

::: details Q：LoRA 训练为什么比全参快？快多少？
每 token 训练 FLOPs 约 $6\Psi$，其中对权重求梯度占 $2\Psi$。底座冻结后这部分省掉，剩约 $4\Psi$，理论 1.5 倍。还省了优化器 step（只更新 0.6% 的参数）和数据并行的梯度 all-reduce（7B 从 14 GB 降到 80 MB）。LoRA 论文在 GPT-3 上实测训练吞吐 +25%（作者自测）。QLoRA 多了反量化，会比 bf16 LoRA 慢。
:::

::: details Q：NF4 和 INT4 有什么区别？为什么 NF4 只适合权重？
INT4 格点等间距；NF4 格点取在标准正态分位数上，0 附近密、两端稀，按块 absmax 归一化后正好匹配近似正态的权重分布，量化误差更小。激活有大幅 outlier、分布不是正态，用 NF4 没有这个优势；而且 NF4 没有原生 Tensor Core 支持，必须反量化成 bf16 再算，它省的是存储和带宽，不省计算。
:::

::: details Q：QLoRA 训出来的 adapter 部署时怎么办？
两种：(1) 推理也用 NF4 底座 + 不合并的 adapter，和训练时看到的权重完全一致；(2) 把 NF4 底座反量化成 bf16、合并 $sBA$，得到 bf16 模型，再用 GPTQ / AWQ 等重新量化并重新评测。不要把合并后的权重直接再压回 NF4 后不评测就上线，误差和训练时不一样。
:::

::: details Q：多 LoRA 服务时，batch 里 adapter 越多越慢吗？
是。底座 GEMM 共享，但每个出现在 batch 里的不同 adapter 都要从 HBM 读一遍，SGMV 的段也更碎。7B、全线性层 $r=16$ 时一个 adapter 80 MB，32 个不同 adapter 让 decode 每步多读约 19% 的字节。所以调度器要限制每个 batch 内的 adapter 数（vLLM 的 `max_loras`），并尽量把同 adapter 的请求放在一起。
:::

::: details Q：rank 选多大？$\alpha$ 怎么设？
QLoRA 的实验里，覆盖所有线性层比调大 $r$ 更重要（作者自测），常见 $r = 8$–$64$。$\alpha$ 常设为 $r$ 或 $2r$，换 $r$ 时保持 $\alpha$ 不变就近似不用重调 lr。$r$ 很大时用 rsLoRA 的 $\alpha/\sqrt{r}$。学习率通常比全参大一个数量级（LoRA 常见 1e-4 量级，全参 1e-5 量级）。
:::

## 手撕

```python
import math, torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)          # 冻结底座
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        k, d = base.in_features, base.out_features
        self.A = nn.Parameter(torch.empty(r, k))         # shrink: [r, k]
        self.B = nn.Parameter(torch.zeros(d, r))         # expand: [d, r]，全零
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)) # 和 nn.Linear 默认初始化一致
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):                                # x: [..., k]
        y = self.base(x)
        if not self.merged:
            y = y + self.scale * (self.drop(x) @ self.A.T) @ self.B.T   # 先降到 r 维
        return y

    @torch.no_grad()
    def merge(self):                                     # 部署：W' = W0 + s·BA
        self.base.weight += self.scale * (self.B @ self.A)
        self.merged = True

def lora_params(shapes, r, L):
    """shapes: 每层加 LoRA 的 (d, k) 列表；返回总的可训练参数"""
    return L * sum(r * (d + k) for d, k in shapes)

h, i = 4096, 11008
qv  = [(h, h), (h, h)]
all7 = [(h, h)] * 4 + [(i, h), (i, h), (h, i)]
print(lora_params(qv, 16, 32))     # 8,388,608   ≈ 0.12% of 6.74B
print(lora_params(all7, 16, 32))   # 39,976,960  ≈ 0.59%
```

常见变体：算 70B 全线性层 $r=16$ 的参数量（207M）；QLoRA 下 70B 底座多少 GB（36.4）；问 `merge` 之后还能不能继续训练（能，但要先 unmerge 或重新挂一个零初始化的 adapter）。

## 参考

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)：NF4、double quantization、paged optimizer
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)
- [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA)](https://arxiv.org/abs/2312.03732)
- [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)
- [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547)：SGMV kernel
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285)：Unified Paging
- [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861)：bitsandbytes 的分块量化与 8-bit Adam
