---
title: 训练稳定性：loss spike 从哪来、怎么防、怎么救
status: draft
tags: [stability, loss-spike, qk-norm, z-loss, mup, initialization]
difficulty: 3
order: 6
related: [/train/optimizer-lr, /posttrain/training-memory, /posttrain/rl-monitoring, /inference/flash-attention, /parallel/megatron-tp]
stack: []
---

# 训练稳定性：loss spike 从哪来、怎么防、怎么救

> 不稳定通常是某个量悄悄涨到了数值或优化器扛不住的地方；先监控、再从结构上限住它，最后才是回滚

## 一句话结论

大模型训练的不稳定多数能追到「某个量无界增长」：**attention logits** 涨到 softmax 变成 one-hot（用 QK-norm 限住），**输出 logits** 整体漂移（用 z-loss $10^{-4}\log^2 Z$ 拉回），或者 **Adam 的分母**和梯度尺度对不上（$\beta_2$、$\epsilon$）。初始化按 std 0.02、残差输出层再除以 $\sqrt{2L}$；μP 让超参能从小模型迁移。真出了 spike，标准做法是 PaLM 的：回到 spike 前约 100 步的 checkpoint，跳过 200–500 个 batch 再训（作者自测）。这些不稳定大多能在小模型上用大学习率复现（[Wortsman et al. 2023](https://arxiv.org/abs/2309.14322)），所以先在小规模上验证对策。

## 推导

### 0. 记号、三种不稳定、监控什么

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $q_i, k_j$ | 一个 head 里第 $i$ 个 query、第 $j$ 个 key 向量 | |
| $d_h$ | head 维度 | 128 |
| $z_{ij} = q_i^\top k_j / \sqrt{d_h}$ | attention logit | |
| $y \in \mathbb{R}^V$ | 一个位置的输出 logits，$V$ 为词表大小 | |
| $Z = \sum_j e^{y_j}$ | softmax 的归一化项，$\log Z$ 即 logsumexp | |
| $c$ | soft-cap 上限 | 30 / 50 |
| $L$ | Transformer 层数（每层 2 个残差分支） | 32 |
| $\sigma$ | 初始化标准差 | 0.02 |
| $n$ | 宽度（$d_{\text{model}}$） | 4096 |
| $w, k$ | spike 检测的窗口步数、标准差倍数 | 50, 5 |

不稳定有三种样子：
1. **spike 后恢复**：loss 突然跳高，几步到几百步后回到原来的轨迹。代价是浪费一些算力，模型通常没事。
2. **spike 后不恢复 / 发散**：loss 停在高位或一路涨，grad norm 爆炸，只能回滚。
3. **缓慢 NaN**：loss 看着正常，但某个内部量（logit、激活、grad norm）一直在涨，某天溢出成 inf / NaN。

所以只看 loss 不够，要看能提前预警的量：

| 量 | 怎么算 | 看什么 |
|---|---|---|
| loss | 每步训练 loss | spike、发散 |
| grad norm | 裁剪前的全局 L2 norm（见 [一个训练 step](/train/optimizer-lr)） | 突增；OLMo 2 观察到 grad norm 的 spike 常常先于 loss spike，grad norm 缓慢上涨和 spike 变频繁相关（[OLMo 2](https://arxiv.org/abs/2501.00656) §3，作者自测） |
| update / param ratio | 每层 $\lVert\Delta\theta\rVert / \lVert\theta\rVert$ | CS231n 的经验值是 $10^{-3}$ 左右；某层突然大很多说明它在乱跳 |
| 最大 attention logit | 每层 $\max_{ij} \lvert z_{ij}\rvert$ | 持续上涨 → 第 1 节 |
| $\log Z$ 的均值 | 输出 logits 的 logsumexp | 远离 0 → 第 2 节 |
| 最大激活 | 残差流的 $\max\lvert x\rvert$ | 持续上涨 → bf16 精度问题（第 7 节） |

RL 阶段的监控见 [RL 训练监控](/posttrain/rl-monitoring)。

### 1. 根因一：attention logits 增长 → softmax 塌缩 → QK-norm

[Dehghani et al. 2023](https://arxiv.org/abs/2302.05442)（ViT-22B）把 ViT 放大到约 8B 参数时，训练几千步后 loss 发散。原因是 attention logits 变得极大，不做归一化时 8B 模型的 logits 很快超过 50000（作者自测），softmax 输出近乎 one-hot、熵接近 0。

为什么会涨：$z_{ij}$ 是 $q$ 和 $k$ 的内积，而 $q = W_Q x$、$k = W_K x$，所以 logits 对参数的 norm 是**二次**的。$W_Q$、$W_K$ 各涨 3 倍，logits 涨 9 倍。Wortsman et al. 的实验也指向这一点：logit 增长来自 $q$、$k$ 的 norm 变大，而不是它们更对齐（作者自测）。

数字例子：一个 query 对两个 key 的 logits 是 2 和 1，softmax 是 $[0.73, 0.27]$。$q$、$k$ 的 norm 各涨 3 倍后 logits 是 18 和 9，softmax 是 $[0.9999, 0.0001]$。softmax 对 logit 的导数是 $p_i(1-p_i)$，从 0.2 掉到 $10^{-4}$，这个 head 几乎没有梯度、也不再混合信息。

**QK-norm**：算内积之前对 $q$、$k$ 各做一次 LayerNorm / RMSNorm（每个 head 单独做）：

$$
z_{ij} = \frac{\text{Norm}(q_i)^\top \text{Norm}(k_j)}{\sqrt{d_h}}
$$

RMSNorm 后向量的 RMS 是 1（乘上可学的 scale $\gamma$ 前），即 $\lVert q\rVert = \sqrt{d_h}$。由 Cauchy–Schwarz，$\gamma = 1$ 时 $\lvert z_{ij}\rvert \le \sqrt{d_h}\sqrt{d_h}/\sqrt{d_h} = \sqrt{d_h}$，$d_h = 128$ 时不超过 11.3。logit 的上界只剩 $\gamma$ 能调，增长速度慢得多。Wortsman et al. 在小模型上复现了这个不稳定，加 QK-norm 后 1.2B 模型在 lr 0.3 下也能训（作者自测）。Qwen3、OLMo 2 都用了 QK-norm（[Qwen3](https://arxiv.org/abs/2505.09388)、[OLMo 2](https://arxiv.org/abs/2501.00656) 技术报告）。

### 2. 根因二：输出 logits 漂移 → z-loss

softmax 对 logits 整体平移不变：所有 $y_j$ 加同一个常数，概率和 cross-entropy 都不变。所以 CE 本身不约束 $\log Z$ 的绝对位置，它可以一路漂走。Wortsman et al. 复现到的现象是 logits 整体变得很负，发生在训练后期，没有 weight decay 的模型无论大小都会出现（作者自测）。漂到很大的绝对值后，bf16 的分辨率不够（第 7 节）。

**z-loss**（[PaLM](https://arxiv.org/abs/2204.02311) §5）：加一项辅助 loss 把 $\log Z$ 拉向 0：

$$
\mathcal{L} = \text{CE} + 10^{-4}\cdot\log^2 Z, \qquad \frac{\partial\, (10^{-4}\log^2 Z)}{\partial y_j} = 2\times10^{-4}\cdot\log Z\cdot p_j
$$

梯度对每个 logit 的推动和它的概率 $p_j$ 成正比，方向由 $\log Z$ 的符号决定：$\log Z < 0$ 时把所有 logits 往上推，相当于纠正整体平移，不改相对大小。

例子：$V = 4$，logits $= [-100, -101, -102, -103]$，目标是第 0 个。
1. $\log Z = -100 + \log(1 + e^{-1} + e^{-2} + e^{-3}) = -100 + 0.440 = -99.56$。
2. CE $= \log Z - y_0 = 0.440$。
3. z-loss $= 10^{-4}\times 99.56^2 = 0.991$，比 CE 还大。
4. 整体加 100 后 CE 仍是 0.440，$\log Z = 0.440$，z-loss 只有 $1.9\times10^{-5}$。

系数 $10^{-4}$ 来自 PaLM，Wortsman et al. 沿用；OLMo 2 用的是 $10^{-5}$（其表 3）。CE 本来就要算 logsumexp，z-loss 只多一次平方，代价可以忽略。

### 3. Logit soft-capping（Gemma 2）

另一种思路是直接把 logits 压进 $(-c, c)$（[Gemma 2](https://arxiv.org/abs/2408.00118) §2）：

$$
\text{softcap}(x) = c \cdot \tanh(x / c)
$$

$\lvert x\rvert \ll c$ 时近似恒等，接近 $c$ 时被压平。Gemma 2 对 attention logits 取 $c = 50$，对最终输出 logits 取 $c = 30$。例：$c = 30$ 时 $x = 10 \to 9.65$，$x = 60 \to 28.92$，$x = 100 \to 29.92$。

和 FlashAttention 的兼容问题：soft-cap 要在 $QK^\top$ 之后、softmax 之前逐元素做 tanh，backward 也要乘 tanh 的导数，这必须写进融合 kernel 里，标准的 SDPA / FlashAttention 接口只支持 scale 和 mask。Gemma 2 发布时 HF 明确说 soft-capping 与 FlashAttention / SDPA 不兼容，微调要用 eager attention（[HF blog](https://huggingface.co/blog/gemma2)）；之后 FlashAttention 2.6 加了 softcap 参数（[README](https://github.com/Dao-AILab/flash-attention)）。FlashAttention 本身见 [FlashAttention](/inference/flash-attention)。

### 4. Adam 自身的不稳定：$\beta_2$ 与 $\epsilon$

Adam 的更新是 $\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$（公式见 [一个训练 step](/train/optimizer-lr) 第 3 节）。分母和当前梯度尺度对不上时，会往两个方向出事：

**$v$ 相对当前梯度太小 → 更新过大。** 一个参数之前梯度一直很小、这一步突然变大，$v$ 还停在旧的尺度。Adam 论文给的单步上界是 $\eta(1-\beta_1)/\sqrt{1-\beta_2}$：$\beta_2 = 0.999$ 时 $3.16\eta$，$\beta_2 = 0.95$ 时 $\eta$。这是 LLM 用 0.95 的原因，推导见 [一个训练 step](/train/optimizer-lr)。

**$\epsilon$ 相对梯度太大 → 更新过小。** Wortsman et al. §3.4 的发现（作者自测）：
1. 梯度的 RMS 随模型变大、随 lr 变大而**下降**；在他们测的最大规模和 lr 下，梯度 RMS 已经接近默认的 $\epsilon = 10^{-8}$。
2. 此时 $\sqrt{\hat v} \approx \epsilon$，更新被压小，参数学不动，梯度 RMS 进一步塌缩。
3. 4.8B 模型、lr 0.3：$\epsilon$ 从 $10^{-8}$ 降到 $10^{-15}$，loss 变好；升到 $10^{-6}$ 则发散。

算一下：梯度 RMS 恒为 $g$ 时 $\sqrt{\hat v} \approx g$，更新是正常步长的 $g/(g+\epsilon)$。$g = 10^{-8}$、$\epsilon = 10^{-8}$ 时只剩 0.5；$g = 10^{-9}$ 时剩 0.09。OLMo 2 把 $\epsilon$ 从 $10^{-5}$ 降到 $10^{-8}$ 后，早期的 grad norm 很快稳定并保持更低（作者自测）。

### 5. 初始化与参数化

**std 0.02 与残差缩放。** GPT 系列和 Megatron 的默认：所有权重 $\mathcal{N}(0, 0.02^2)$；每个残差分支**最后一个**线性层（attention 的输出投影、MLP 的 down 投影）再乘 $1/\sqrt{N}$，$N = 2L$ 是残差分支数（[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) §2.3；Megatron `output_layer_init_method` 的 std 是 `init_method_std / sqrt(2 * num_layers)`）。

原因：残差流是 $x_{2L} = x_0 + \sum_{l=1}^{2L} f_l(x_{l-1})$。初始化时各分支近似独立，方差相加，$2L$ 个方差为 $s^2$ 的分支加起来是 $2L\cdot s^2$，深度越大残差流越大。分支输出层的 std 除以 $\sqrt{2L}$，每个分支的方差变成 $s^2/(2L)$，总和回到 $s^2$。$L = 32$：$0.02/\sqrt{64} = 0.0025$。

**μP：为什么超参能从小模型迁移。**（[Tensor Programs V](https://arxiv.org/abs/2203.03466)）

核心直觉：在标准参数化（SP）下，最优 lr 随宽度变。用 Adam 时更新 $\Delta W$ 的每个元素约为 $\eta$（Adam 把尺度归一化了），而 $\Delta W$ 来自 $g x^\top$，和输入 $x$ 是对齐的，所以 $(\Delta W x)_i$ 是 $n$ 个同号项相加，大小约 $\eta \cdot n$。宽度翻倍，每步激活的变化翻倍，最优 $\eta$ 就得减半。μP 把这类随宽度变化的因子显式写进参数化，让每层激活每步的变化量和宽度无关，于是最优超参在不同宽度下不变。

宽度从基准 $n_0$ 放大到 $n$（$m = n/n_0$）时，用 Adam 要改三处（论文 Table 8 的形式，也是 [mup](https://github.com/microsoft/mup) 库的做法）：
1. **隐藏层矩阵的 lr 除以 $m$**：`MuAdam` 里是 `globalLR / width_mult`。初始化方差照常 $\propto 1/\text{fan\_in}$。
2. **输出层 logits 乘 $1/m$**：`MuReadout` 替换输出线性层。
3. **attention 的缩放从 $1/\sqrt{d_h}$ 改成 $1/d_h$**。

embedding、bias、norm 的 scale 这类只有一个维度随宽度变的参数，lr 和初始化不随宽度缩放。例：$n_0 = 256$ 上调出 lr $= 10^{-2}$，迁到 $n = 4096$（$m = 16$）时隐藏层 lr 用 $6.25\times10^{-4}$，其他参数仍用 $10^{-2}$。论文从 40M 参数的代理模型迁移超参，超过了公开的 GPT-3 6.7B 结果，调参成本是预训练的 7%（作者自测）。μP 解决的是宽度方向，深度和训练步数的迁移不在这篇的保证里。

### 6. 出事了怎么救

**PaLM 的做法**（[PaLM](https://arxiv.org/abs/2204.02311) §5.1，作者自测）：540B 模型训练中出现约 20 次 spike，开着梯度裁剪也有，小模型没有。处理方法：
1. 回到 spike 开始前约 **100 步**的 checkpoint。
2. 跳过约 **200–500 个 batch**（覆盖 spike 之前和期间见过的数据），接着训。
3. 这样处理后，同一位置没有再出现 spike。

**为什么换数据顺序常能过去。** PaLM 做了对照：把 spike 附近的那些 batch 拿出来，从**更早的另一个 checkpoint** 开始训，不会 spike。所以 spike 不是某批「坏数据」单独造成的，而是**特定数据和特定参数状态的组合**。跳过这些 batch 后，参数走到这些数据时状态已经不同，组合不再出现。

其他手段，按代价从小到大：
1. **降 lr**：回滚后把 lr 降一些（或重新短 warmup），让参数穿过不稳定区域。
2. **查数据**：OLMo 2 发现 spike 处的 batch 里有大量长的重复 n-gram，过滤后 spike 减少（作者自测）。和「组合问题」不矛盾：某些数据更容易触发。
3. **改结构**：加 QK-norm、z-loss，只能从头训或在较早的 checkpoint 上改，代价最大，所以应在开训前就加。

工程前提：checkpoint 要足够密（回滚 100 步要有 100 步前的 checkpoint，间隔怎么定见 [大规模训练的容错](/train/fault-tolerance)）；data loader 要确定、可按全局样本号跳过和恢复，否则「跳过 200–500 个 batch」无法精确执行（`consumed_samples` 的做法见 [预训练数据管线](/train/data-pipeline)）。

### 7. bf16 相关的坑

bf16 尾数 7 位，数值越大绝对误差越大：$[2^e, 2^{e+1})$ 区间里相邻两数间隔 $2^{e-7}$。精度账见 [训练显存账](/posttrain/training-memory)。
1. **logits 和 loss 用 fp32 算**：logit $= -200$ 落在 $[128, 256)$，间隔 $2^{7-7} = 1.0$，舍入误差可达 0.5，对应概率差 $e^{0.5} = 1.65$ 倍。这也是 z-loss 把 $\log Z$ 拉回 0 附近的意义。
2. **softmax、RMSNorm 的归约用 fp32**：几千个数的平方和在 bf16 里累加，后加的小数会被吞掉。FlashAttention 内部的 softmax 统计量就是 fp32。
3. **梯度累加和 reduce 用 fp32**：Megatron 的 `main_grad` 是 fp32；主权重和 Adam 的 $m$、$v$ 也是 fp32。

## 面试追问

::: details Q：attention logits 为什么会无限涨，QK-norm 为什么能治？
logit 是 $q^\top k/\sqrt{d_h}$，$q$、$k$ 都是参数乘输入，logit 对参数 norm 是二次的，$W_Q$、$W_K$ 慢慢变大，logit 涨得更快，softmax 塌成 one-hot，梯度 $p(1-p)$ 趋于 0。QK-norm 在内积前把 $q$、$k$ 归一化，RMS 为 1 时 $\lvert z\rvert \le \sqrt{d_h}$，只剩可学的 scale 能放大它。
:::

::: details Q：z-loss 在优化什么？为什么不影响模型预测？
优化 $\log Z$ 接近 0。softmax 对整体平移不变，所以把所有 logits 一起平移不改变任何概率，z-loss 只消除了这个自由度，不改相对大小。好处是 logits 保持在 0 附近，bf16 下分辨率足够；PaLM 的系数是 $10^{-4}$。
:::

::: details Q：soft-capping 和 z-loss 有什么区别？
soft-capping 是前向里的硬结构：$c\tanh(x/c)$ 把每个 logit 压在 $(-c, c)$，改变了函数本身，接近上限时梯度变小。z-loss 是 loss 里的软约束，只约束整体平移。soft-capping 用在 attention logits 上需要 kernel 支持（FlashAttention 2.6 起有 softcap 参数）。
:::

::: details Q：AdamW 的 eps 为什么会影响稳定性？默认 1e-8 一定对吗？
更新是 $\hat m/(\sqrt{\hat v}+\epsilon)$。梯度 RMS 接近 $\epsilon$ 时更新被压到正常的一半甚至更小，Wortsman et al. 发现大模型、大 lr 下梯度 RMS 确实会降到 $10^{-8}$ 附近，4.8B 模型把 $\epsilon$ 降到 $10^{-15}$ 更好（作者自测）。OLMo 2 也把 $\epsilon$ 从 $10^{-5}$ 降到 $10^{-8}$（作者自测）。所以 $\epsilon$ 要对照实际梯度尺度看。
:::

::: details Q：为什么残差输出层要除以 √(2L)？
残差流是 $2L$ 个分支输出相加，初始化时近似独立，方差线性累加，深层的残差流越来越大。每个分支输出层 std 除以 $\sqrt{2L}$，总方差和深度无关。$L = 32$ 时 $0.02 \to 0.0025$。
:::

::: details Q：μP 能让我在 100M 模型上调 lr 直接用到 7B 吗？
宽度方向可以：μP 下隐藏层 lr 按 $1/m$ 缩、输出 logits 乘 $1/m$、attention 用 $1/d_h$，最优的全局 lr 在不同宽度下基本不变（论文从 40M 迁到 6.7B，作者自测）。但 batch size、训练步数、深度不在它的保证里，需要另外处理。
:::

::: details Q：loss spike 了你怎么办？
先看是否自己恢复；不恢复就按 PaLM：回到 spike 前约 100 步的 checkpoint，跳过 200–500 个 batch，必要时降 lr。PaLM 的对照实验表明 spike 是数据和参数状态的组合，同样的数据从更早的 checkpoint 训不会 spike，所以换顺序通常能过去。之后查 spike 前的 grad norm、attention logit、$\log Z$ 曲线定位根因，下次开训前把 QK-norm、z-loss 加上。
:::

::: details Q：为什么同样的数据从别的 checkpoint 训就不 spike？
PaLM 的对照实验给出的结论是：spike 是特定数据和特定参数状态的组合，单独哪一个都不够。直观理解：参数要先走到对这类数据敏感的状态，这批数据的梯度才会把它推出去。换了起点或换了顺序，参数遇到这批数据时状态不同，组合就不再出现。这也说明 spike 不能只靠清洗数据根治，OLMo 2 过滤重复 n-gram 只是减少了容易触发的数据。
:::

## 手撕

z-loss、QK-norm attention、spike 检测：

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def ce_with_z_loss(logits, target, z_coef=1e-4):
    """logits: (N, V)，target: (N,)。返回 (总 loss, ce, z_loss)"""
    logits = logits.float()                              # loss 一律在 fp32 里算
    log_z = torch.logsumexp(logits, dim=-1)              # log Z，每个位置一个
    ce = (log_z - logits.gather(-1, target[:, None]).squeeze(-1)).mean()
    z_loss = z_coef * (log_z ** 2).mean()
    return ce + z_loss, ce, z_loss

class QKNormAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.dh)                # 每个 head 的 q、k 各做一次 norm
        self.k_norm = nn.RMSNorm(self.dh)

    def forward(self, x):                                # x: (B, T, d_model)
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)            # 唯一的改动：在 QK^T 之前 norm
        q, k, v = (t.transpose(1, 2) for t in (q, k, v)) # (B, h, T, dh)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(o.transpose(1, 2).reshape(B, T, -1))

def find_spikes(losses, window=50, k=5.0):
    """loss 超过前 window 步的均值 + k 倍标准差，记为 spike；spike 步不进入统计窗口"""
    hist, spikes = [], []
    for step, x in enumerate(losses):
        if len(hist) >= window:
            mu = sum(hist) / len(hist)
            sd = math.sqrt(sum((h - mu) ** 2 for h in hist) / len(hist))
            if x > mu + k * sd:
                spikes.append(step)
                continue
        hist.append(x)
        hist = hist[-window:]
    return spikes

logits = torch.tensor([[-100.0, -101.0, -102.0, -103.0]])
print([round(t.item(), 4) for t in ce_with_z_loss(logits, torch.tensor([0]))])
# [1.4314, 0.4402, 0.9912]
print([round(t.item(), 4) for t in ce_with_z_loss(logits + 100, torch.tensor([0]))])
# [0.4402, 0.4402, 0.0]        ← 平移不改 CE，z-loss 消失

torch.manual_seed(0)
print(QKNormAttention(64, 4)(torch.randn(2, 10, 64)).shape)
# torch.Size([2, 10, 64])

torch.manual_seed(0)
losses = [3.0 - 0.001 * s + 0.01 * torch.randn(()).item() for s in range(300)]
losses[200] += 0.5; losses[201] += 0.3
print(find_spikes(losses))
# [200, 201]
```

要点：
1. `nn.RMSNorm` 需要 PyTorch 2.4+；QK-norm 是每个 head 在 $d_h$ 维上做，不是在 $d_{\text{model}}$ 上做。
2. spike 步不进统计窗口，否则一次 spike 把标准差拉大，紧接着的第二次就检测不到。窗口里要先去掉 loss 的下降趋势时，可以改成对 loss 的差分做同样的检测。

## 参考

- [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322)：小模型复现 logit 增长、输出 logit 发散，$\epsilon$ 的问题
- [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442)：attention logit 增长与 QK-norm
- [PaLM](https://arxiv.org/abs/2204.02311)：z-loss（§5）、loss spike 的处理（§5.1）
- [Gemma 2](https://arxiv.org/abs/2408.00118)：logit soft-capping；[HF blog: Welcome Gemma 2](https://huggingface.co/blog/gemma2)；[FlashAttention README](https://github.com/Dao-AILab/flash-attention)（2.6 softcap）
- [Adam](https://arxiv.org/abs/1412.6980)：§2.1 单步更新上界
- [OLMo 2](https://arxiv.org/abs/2501.00656) §3：spike 与 grad norm、重复 n-gram、QK-norm、z-loss、$\epsilon$
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)：残差层 $1/\sqrt{N}$ 初始化
- [Megatron-LM `transformer_config.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py)：`init_method_std`、`output_layer_init_method`
- [Tensor Programs V (μP / μTransfer)](https://arxiv.org/abs/2203.03466)、[microsoft/mup](https://github.com/microsoft/mup)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)：QK-Norm
- [CS231n: Neural Networks Part 3](https://cs231n.github.io/neural-networks-3/)：update / param ratio 约 $10^{-3}$
