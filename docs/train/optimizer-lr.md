---
title: 一个训练 step：AdamW、学习率调度与梯度裁剪
status: draft
tags: [optimizer, adamw, lr-schedule, gradient-clipping, muon]
difficulty: 3
order: 2
related: [/posttrain/training-memory, /parallel/zero-fsdp, /posttrain/sft, /parallel/comm-overlap, /train/stability]
stack: []
train: [tr-st-opt, tr-st-lr, tr-st-accum]
---

# 一个训练 step：AdamW、学习率调度与梯度裁剪

> 从取 batch 到 zero_grad，每一行代码在干什么、超参为什么这么取

## 一句话结论

一个 step 的顺序是固定的：forward → backward（梯度在最后一个 micro-batch 时跨 DP rank 同步）→ **按全局 norm 裁剪** → **AdamW 更新** → **调学习率** → 清梯度。gradient accumulation 下 loss 要除以**全局**计入 loss 的 token 数，不是每个 micro-batch 各自平均。AdamW 的 LLM 常用超参是 $\beta_1 = 0.9$、$\beta_2 = 0.95$、weight decay 0.1、裁剪阈值 1.0（GPT-3、Llama 2）；$\beta_2$ 取 0.95 能保证单步更新量不超过学习率。学习率用 warmup + cosine，或 WSD（stable 段的 checkpoint 可以随时续训）。Muon 只替换隐藏层 2D 矩阵的优化器，Moonlight 报告约 2 倍算力效率（作者自测）。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $\theta$ | 参数 | |
| $t$ | optimizer step 序号，从 1 开始 | |
| $g_t$ | 第 $t$ 步的梯度（同步、裁剪之后） | |
| $m_t, v_t$ | Adam 一阶动量（梯度的滑动平均）、二阶动量（梯度平方的滑动平均） | |
| $\beta_1, \beta_2$ | $m$、$v$ 的衰减系数 | 0.9, 0.95 |
| $\epsilon$ | 分母里防除零的小数 | $10^{-8}$ |
| $\lambda$ | weight decay 系数 | 0.1 |
| $\eta_t$ | 第 $t$ 步学习率；$\eta_{\max}$ 峰值，$\eta_{\min}$ 终值 | $3\times10^{-4}$ |
| $T_w, T$ | warmup 步数、总步数 | 2000, 100000 |
| $W$ | 数据并行（DP）rank 数 | 64 |
| $K$ | 每个 rank 每步累积的 micro-batch 数 | 8 |
| $b, s$ | micro-batch 的序列数、序列长度 | 2, 4096 |
| $B$ | global batch，按 token 计：$B = W K b s$ | 4.19M |
| $N$ | 这一步所有 rank 所有 micro-batch 里**计入 loss** 的 token 数 | $\le B$ |
| $c$ | 梯度裁剪阈值（max_norm） | 1.0 |

例子：$64 \times 8 \times 2 \times 4096 = 4{,}194{,}304 \approx 4$M token，正好是 Llama 3 405B 起步时的 batch（见第 5 节）。

### 1. 一个 step 按顺序做了什么

```python
for step in range(T):
    mbs = next_global_batch()                    # 本 rank 这一步的 K 个 micro-batch
    n_tok = sum(count_loss_tokens(mb) for mb in mbs)
    dist.all_reduce(n_tok)                       # 跨 DP rank 求和 → 全局 N（第 2 节）
    for k, mb in enumerate(mbs):
        last = (k == K - 1)
        with (nullcontext() if last else model.no_sync()):   # 前 K-1 个不同步梯度
            loss_sum = model(mb).loss_sum        # forward：本 micro-batch 的 token loss 之和
            (loss_sum * W / n_tok).backward()    # backward：梯度累加进 .grad
    # 最后一个 micro-batch 的 backward 里，DDP 已对梯度做了 all-reduce 求平均
    gnorm = clip_grad_norm_(model.parameters(), 1.0)   # 裁剪（第 6 节）
    optimizer.step()                             # AdamW（第 3 节）
    scheduler.step()                             # 学习率（第 4 节）
    optimizer.zero_grad(set_to_none=True)
```

每一步为什么在这个位置：
1. **gradient accumulation**：$K$ 次 forward/backward 只做一次更新，梯度在 `.grad` 里累加。它让「放不下的大 batch」变成 $K$ 个小 micro-batch，激活按 $b$ 算（见 [训练显存账](/posttrain/training-memory)）。
2. **`no_sync()`**：DDP 默认每次 backward 都 all-reduce 梯度。前 $K-1$ 个 micro-batch 不需要同步，同步一次就够，通信量降到 $1/K$。DDP 的 all-reduce 按 bucket 和 backward 重叠，见 [计算通信 overlap](/parallel/comm-overlap)；FSDP 把 all-reduce 换成 reduce-scatter，见 [ZeRO / FSDP](/parallel/zero-fsdp)。
3. **裁剪在同步之后、更新之前**：裁剪看的是全局梯度的 norm，必须等所有 rank 的梯度合并好；更新要用裁剪后的梯度。
4. **`scheduler.step()` 在 `optimizer.step()` 之后**：这一步用当前 lr 更新，再把 lr 推到下一步的值。PyTorch 1.1 起顺序反了会报警告，而且第一步的 lr 会被跳过。
5. **`zero_grad`**：不清的话下一步会接着累加。`set_to_none=True` 直接释放梯度张量，比填 0 省一次写。

### 2. Gradient accumulation 下 loss 怎么归一化

单卡多个 micro-batch 的问题（按 micro-batch 平均会放大短样本的权重）在 [SFT 的「Loss 怎么平均」](/posttrain/sft) 已经讲过。这里只讲**跨 DP rank** 多出来的一层：DDP 的 all-reduce 对梯度**求平均**（除以 $W$），所以每个 rank 应该反传

$$
\frac{W}{N}\sum_{\text{本 rank 的 token } i} \ell_i
$$

$W$ 个 rank 求和再除以 $W$，得到的正好是 $\frac{1}{N}\sum_{\text{全部 token}} \ell_i$。

例子：$W = 2$，$K = 2$，四个 micro-batch 的（计入 loss 的 token 数，loss 之和）：

| | micro-batch 1 | micro-batch 2 | 本 rank 合计 |
|---|---|---|---|
| rank 0 | 100 token，loss 和 200 | 300 token，loss 和 450 | 400 token，650 |
| rank 1 | 500 token，loss 和 1100 | 100 token，loss 和 250 | 600 token，1350 |

1. 正确：$N = 1000$，$(650 + 1350)/1000 = 2.0$。实现上 rank 0 反传 $650 \times 2/1000 = 1.3$，rank 1 反传 $2.7$，DDP 平均后是 $2.0$。
2. 每个 micro-batch 各自平均：rank 0 $(2.0 + 1.5)/2 = 1.75$，rank 1 $(2.2 + 2.5)/2 = 2.35$，DDP 平均 $2.05$。
3. 每个 rank 按本地 token 数平均：rank 0 $650/400 = 1.625$，rank 1 $1350/600 = 2.25$，DDP 平均 $1.94$。token 少的 rank 0 被放大了权重。

梯度和 loss 一起缩放，所以上面的「loss」换成「每个 token 梯度的权重」结论一样。$N$ 要在 forward 之前就知道，所以第 1 节的代码先统计 token 数、做一次很小的 all-reduce。打日志的 loss 也用同一个 $N$：$\sum$ loss 和跨 rank all-reduce 后除以 $N$。

### 3. Adam → AdamW

**Adam 的更新**（[Kingma & Ba 2014](https://arxiv.org/abs/1412.6980)），所有运算逐元素：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta_t = \theta_{t-1} - \eta_t \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

$m$ 给方向（动量），$\sqrt v$ 给每个参数各自的尺度：梯度一直大的参数步子被除小，一直小的被放大。正常情况下 $|\hat m/\sqrt{\hat v}| \lesssim 1$，所以**每步每个参数大约挪 $\eta$**，和梯度的绝对大小无关。

**偏差修正为什么需要。** $m_0 = v_0 = 0$，前几步的滑动平均被 0 拉低。第 1 步，$g_1 = 0.5$，$\beta_2 = 0.95$：

1. 不修正：$m_1 = 0.1 \times 0.5 = 0.05$，$v_1 = 0.05 \times 0.25 = 0.0125$，$m_1/\sqrt{v_1} = 0.05/0.112 = 0.447$。
2. 修正：$\hat m_1 = 0.05/0.1 = 0.5$，$\hat v_1 = 0.0125/0.05 = 0.25$，比值 $= 1$。
3. $\beta_2 = 0.999$ 不修正：$v_1 = 0.00025$，比值 $0.05/0.0158 = 3.16$，第一步走了正常步长的 3 倍多。

$m$ 和 $v$ 被低估的倍数不同（$1-\beta_1^t$ 对 $1-\beta_2^t$），比值就偏了。$t$ 大了以后 $\beta^t \to 0$，修正项趋于 1。

**L2 正则 vs decoupled weight decay**（[Loshchilov & Hutter 2017](https://arxiv.org/abs/1711.05101)）：

1. L2 正则：把 $\frac{\lambda}{2}\lVert\theta\rVert^2$ 加进 loss，等于 $g_t \leftarrow g_t + \lambda\theta$，然后进 $m$、$v$。衰减项也被除以 $\sqrt{\hat v}$。
2. AdamW：衰减不进梯度，直接作用在参数上：

$$
\theta_t = \theta_{t-1} - \eta_t\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta_{t-1}\right)
$$

区别在 L2 下衰减被自适应尺度扭曲了。例：两个参数都是 $\theta = 1$、$\lambda = 0.1$，一个历史梯度大（$\sqrt{\hat v} = 10$），一个小（$\sqrt{\hat v} = 0.01$）。L2 下衰减项近似为 $\eta\lambda\theta/\sqrt{\hat v}$（衰减项远小于梯度时），分别是 $0.01\eta$ 和 $10\eta$，差 1000 倍；梯度大的参数几乎不被正则。AdamW 下两者都是 $0.1\eta$。SGD 下两者等价（差一个 $\eta$ 的重参数化），Adam 下不等价，这是论文的核心论点。

量级感：$\eta = 3\times10^{-4}$、$\lambda = 0.1$，每步参数乘 $(1 - 3\times10^{-5})$；没有梯度的话，$10^5$ 步后剩 $e^{-3} \approx 5\%$。

**LLM 常用超参**：

| | $\beta_1$ | $\beta_2$ | $\epsilon$ | $\lambda$ | 裁剪 | 来源 |
|---|---|---|---|---|---|---|
| GPT-3 | 0.9 | 0.95 | $10^{-8}$ | 0.1 | 1.0 | [Brown et al. 2020](https://arxiv.org/abs/2005.14165) 附录 B |
| Llama 2 | 0.9 | 0.95 | $10^{-5}$ | 0.1 | 1.0 | [Touvron et al. 2023](https://arxiv.org/abs/2307.09288) §2.2 |

**为什么 $\beta_2 = 0.95$ 而不是默认的 0.999。** Adam 论文 §2.1 给了单步更新量的上界：当 $1-\beta_1 > \sqrt{1-\beta_2}$ 时 $|\Delta\theta| \le \eta\,\frac{1-\beta_1}{\sqrt{1-\beta_2}}$，否则 $|\Delta\theta| \le \eta$。前一种情况在「一个参数前面梯度一直是 0、这一步突然有梯度」时取到：

1. $\beta_2 = 0.999$：$\sqrt{0.001} = 0.032 < 0.1$，上界 $\eta \times 0.1/0.032 = 3.16\eta$。
2. $\beta_2 = 0.95$：$\sqrt{0.05} = 0.224 > 0.1$，上界就是 $\eta$。

LLM 里这种情况很常见：低频 token 的 embedding 行大多数步没有梯度；loss spike 前后梯度尺度突然变大，而 $\beta_2 = 0.999$ 的 $v$ 记忆约 $1/(1-\beta_2) = 1000$ 步，跟不上，$\sqrt v$ 偏小、更新偏大。$\beta_2 = 0.95$ 的记忆约 20 步，$v$ 能及时跟上梯度尺度的变化。$\epsilon$ 的问题（太大会让更新变小）放在 [训练稳定性](/train/stability) 讲。

### 4. 学习率：warmup、cosine、WSD

**warmup 为什么需要。** 前几步 $v$ 只平均了很少几个样本，$1/\sqrt{\hat v}$ 这个「自适应步长」方差很大（RAdam，[Liu et al. 2019](https://arxiv.org/abs/1908.03265)）。偏差修正只修了均值、修不了方差，第 3 节的例子里 $t = 1$ 时 $\hat v_1 = g_1^2$，完全由一个样本决定。warmup 让 lr 从很小开始，等 $v$ 稳下来再用大步长。[Wortsman et al. 2023](https://arxiv.org/abs/2309.14322) 测到 warmup 越长，loss 对 lr 越不敏感，模型越大越明显（作者自测）。

**warmup + cosine：**

$$
\eta_t = \begin{cases} \eta_{\max}\,\dfrac{t}{T_w}, & t \le T_w \\[2mm] \eta_{\min} + \dfrac{1}{2}(\eta_{\max}-\eta_{\min})\left(1 + \cos\dfrac{\pi (t - T_w)}{T - T_w}\right), & t > T_w \end{cases}
$$

Llama 3 405B 的数（[Llama 3 Herd](https://arxiv.org/abs/2407.21783) §3.4.1）：$\eta_{\max} = 8\times10^{-5}$，线性 warmup 8000 步，cosine 在 1,200,000 步内降到 $\eta_{\min} = 8\times10^{-7}$。自己算：
1. $t = 4000$：$8\times10^{-5} \times 4000/8000 = 4\times10^{-5}$。
2. cosine 走到一半（$t \approx 604{,}000$）：$\cos(\pi/2) = 0$，$\eta = 8\times10^{-7} + 0.5 \times 7.92\times10^{-5} = 4.04\times10^{-5}$。
3. 终点：$\cos\pi = -1$，$\eta = \eta_{\min}$。GPT-3、Llama 2 都是降到峰值的 10%。

cosine 的问题：形状由 $T$ 决定，**开训前就要定总步数**。中途想多训，后面的 lr 已经降下去了；从中途 checkpoint 续训要重新拉高 lr。

**WSD（warmup-stable-decay）**（MiniCPM，[Hu et al. 2024](https://arxiv.org/abs/2404.06395)）：warmup 到 $\eta_{\max}$，然后**一直保持**，最后一段用递减函数 $f$ 降下去（MiniCPM 用指数衰减）：

$$
\eta_t = \begin{cases} \eta_{\max}\, t/T_w, & t \le T_w \\ \eta_{\max}, & T_w < t \le T_s \\ \eta_{\max}\, f(t - T_s), & T_s < t \le T \end{cases}
$$

MiniCPM 的观察（作者自测）：decay 段里 loss 快速下降，降到和同步数的 cosine 持平或更低；decay 占总 token 的 10% 就够，2.5% 不够。

例子（线性 decay 便于计算）：$T = 100$k，$T_w = 2$k，$T_s = 90$k，$\eta_{\max} = 3\times10^{-4}$。
1. 2k–90k 步 lr 恒为 $3\times10^{-4}$；95k 步时 $1.5\times10^{-4}$；100k 步降到 0。
2. 训完想加到 200k 步：从 **90k 的 stable checkpoint** 接着用 $3\times10^{-4}$ 训到 180k，再 decay 2 万步。90k–100k 那段 decay 只是一个「分支」，可以扔掉。
3. 想看「训 50k 步的模型」有多好：从 50k 的 checkpoint 分出去 decay 5k 步就行，不用从头按 $T = 55$k 重训一个 cosine。MiniCPM 正是用这一点以线性代价测 data scaling law。

### 5. Batch size 与学习率：critical batch size

[McCandlish et al. 2018](https://arxiv.org/abs/1812.06162) 的模型：梯度 $G$ 由 batch 估计，batch 越大噪声越小、每步能走得越远，但收益递减。分界点是 **gradient noise scale**

$$
B_{\text{noise}} \approx B_{\text{simple}} = \frac{\operatorname{tr}(\Sigma)}{\lVert G\rVert^2}
$$

$\Sigma$ 是单样本梯度的协方差，$\operatorname{tr}(\Sigma)$ 是各分量方差之和。直觉：batch 为 $B$ 时梯度估计的噪声方差是 $\operatorname{tr}(\Sigma)/B$，和信号 $\lVert G\rVert^2$ 一样大时 $B = B_{\text{noise}}$。

设达到某个 loss 最少需要 $S_{\min}$ 步（batch 无穷大时）、最少 $E_{\min}$ 个样本（batch 无穷小时），固定 batch $B$ 时：

$$
S = S_{\min}\left(1 + \frac{B_{\text{noise}}}{B}\right), \quad E = E_{\min}\left(1 + \frac{B}{B_{\text{noise}}}\right), \quad \left(\frac{S}{S_{\min}}-1\right)\left(\frac{E}{E_{\min}}-1\right) = 1
$$

**critical batch size** $B_{\text{crit}} = E_{\min}/S_{\min} \approx B_{\text{noise}}$。$B = B_{\text{crit}}$ 时步数和数据量都是最优值的 2 倍，是时间和算力的折中点。例（假设 $B_{\text{crit}} = 1$M token，$S_{\min} = 10$k，$E_{\min} = 10^{10}$ token）：

| $B$ | 步数 $S$ | token 数 $E$ |
|---|---|---|
| 250k | 50k | $1.25\times10^{10}$ |
| 1M（$=B_{\text{crit}}$） | 20k | $2\times10^{10}$ |
| 4M | 12.5k | $5\times10^{10}$ |

batch 从 1M 加到 4M，步数只少了 37.5%，算力多花 2.5 倍。

**batch ramp-up。** 论文同时发现 $B_{\text{noise}}$ 随训练变大：$\lVert G\rVert$ 随 loss 下降而变小、方差变化不大。所以训练初期用小 batch 不浪费数据，后期再加大。实际做法：
1. GPT-3：batch 从 32k token 线性增加到满值（175B 是 3.2M），用前 4–12B token，看模型大小（[Brown et al. 2020](https://arxiv.org/abs/2005.14165) 附录 B）。
2. Llama 3 405B：起步 4M token、序列 4096；训完 252M token 后翻倍到 8M、序列 8192；训完 2.87T token 后再翻倍到 16M。论文给的理由是早期小 batch 更稳定（作者自测：「few loss spikes」，没有人工干预）。

batch 定了以后怎么拆成 DP 度、micro-batch 和 PP 的 micro-batch 数，见 [并行配置怎么选](/train/parallel-config)。

### 6. 梯度裁剪

把所有参数的梯度看成一个大向量，算全局 L2 norm，超过阈值 $c$ 就整体等比例缩小：

$$
\lVert g\rVert = \sqrt{\sum_{p} \lVert g_p\rVert_2^2}, \qquad g \leftarrow g \cdot \min\left(1,\ \frac{c}{\lVert g\rVert + 10^{-6}}\right)
$$

$10^{-6}$ 是 PyTorch `clip_grad_norm_` 里防除零的常数。等比例缩放**不改方向**，只限制步长。例：三个参数张量梯度 norm 分别是 3、4、12，全局 $\sqrt{9+16+144} = 13$，$c = 1$ 时每个梯度乘 $1/13 \approx 0.077$。

**阈值为什么常取 1.0。** 这是经验值：GPT-3、Llama 2、PaLM 都用 1.0。目的是只截异常步：平稳期大多数步的 norm 在阈值以下，裁剪不生效；spike 时 norm 突然大几倍，被截住。对 Adam 来说，一个异常大的梯度会写进 $m$、$v$，影响之后几十步（$v$ 的记忆约 $1/(1-\beta_2) = 20$ 步），裁剪把这个影响限制住。所以要盯 `grad_norm` 曲线：如果大多数步都在裁剪，说明阈值太小或训练本身有问题。

**分布式下要先求平方和再开根。** 每个 rank 只有一部分梯度：
1. FSDP / ZeRO-2/3：梯度沿 DP 切片。每个 rank 算本地平方和 $s_r$，`all_reduce(SUM)` 后再开根。两个 rank 的 $s_r$ 是 9 和 16 时，全局 norm 是 $\sqrt{25} = 5$，不是 $3+4 = 7$，也不是平均。
2. TP：被切开的矩阵在 TP 组内同样求和；**没被切开的参数**（比如 LayerNorm 权重在每个 TP rank 上都有一份）只能算一次，否则被重复计入 $t$ 次。Megatron 用 `param_is_not_tensor_parallel_duplicate` 过滤。
3. PP：不同 stage 的参数不同，平方和也要跨 stage 求和。
4. 纯 DDP：all-reduce 之后每个 rank 梯度相同，本地算就是全局的，不需要再求和。

所有 rank 必须用同一个缩放系数，否则副本之间参数会漂开。FSDP2 下梯度是 DTensor，`get_total_norm` 算出的 norm 处在「各 rank 部分和」的状态，torchtitan 对它调 `full_tensor()` 完成这次归约，PP 再跨 stage 对平方和做一次 all-reduce（[torchtitan `clip_grad_norm_`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py)）。

### 7. Muon（简要）

[Muon](https://kellerjordan.github.io/posts/muon/)（Keller Jordan 2024）只用于**隐藏层的 2D 权重矩阵**。对形状 $[A, B]$ 的矩阵：

$$
M_t = \mu M_{t-1} + G_t, \qquad O_t = \text{NS}_5(M_t) \approx U V^\top \ \ (M_t = U\Sigma V^\top)
$$

1. $\text{NS}_5$ 是 5 步 Newton–Schulz 迭代：$X_0 = M/\lVert M\rVert_F$，$X \leftarrow aX + b(XX^\top)X + c(XX^\top)^2X$，$(a,b,c) = (3.4445, -4.7750, 2.0315)$。它只用矩阵乘近似把所有奇异值推到 1 附近，即正交化，不用做 SVD。例：$64\times256$ 的随机矩阵归一化后奇异值在 0.067–0.186，迭代 5 步后在 0.68–1.13（自己算的）；这组系数为收敛快而不精确收敛到 1。
2. 直觉：动量矩阵的奇异值往往很不均匀，少数方向主导更新；正交化后每个方向的步长一样，小奇异值方向也能学到东西。
3. 为什么只用于隐藏层矩阵：标量、向量参数（norm 的 scale、bias）不是矩阵，没法正交化；embedding 和输出 head 虽然是 2D，作者实验发现用 AdamW 更好（embedding 有理论依据，head 是经验结论）。所以实际是 **Muon + AdamW 混用**。

**Moonlight**（[Liu et al. 2025](https://arxiv.org/abs/2502.16982)）把 Muon 推到大规模时做了两处改动：
1. 加 weight decay：不加时权重和层输出的 RMS 一直涨，超出 bf16 的高精度范围。
2. 统一更新的 RMS：正交矩阵 $O$ 的 RMS 是 $1/\sqrt{\max(A,B)}$（$4096\times11008$ 的 MLP 矩阵只有 0.0095），所以更新写成 $W \leftarrow W - \eta\,(0.2\sqrt{\max(A,B)}\,O_t + \lambda W)$，让 RMS 落在 0.2，和 AdamW 典型的 0.2–0.4 对齐，直接复用 AdamW 的 lr 和 weight decay。

结果：scaling law 实验里 Muon 达到同样 loss 只需约 52% 的训练 FLOPs，即约 2 倍算力效率（作者自测）。优化器状态只有动量 $M$ 一份，比 Adam 的 $m$、$v$ 每参数省 4 字节。

**分布式的额外通信。** AdamW 逐元素更新，ZeRO-1 每个 rank 只更新自己那片就行；Muon 要**整块矩阵**才能做 Newton–Schulz。Moonlight 的 Distributed Muon：reduce-scatter 梯度、在本地切片上更新动量之后，**在 DP 组内 gather 出完整矩阵**，算完 NS 只保留自己那片更新，再照常 all-gather 参数。多出来的 gather 用 bf16，作者估计总通信量是 ZeRO-1 AdamW 的 1–1.25 倍，优化器部分的延迟占 forward-backward 的 1%–3%（作者自测）。开了 TP 还要在 TP 组再 gather 一次。

## 面试追问

::: details Q：gradient accumulation 和直接用大 batch 在数学上等价吗？
loss 按全局 token 数归一化时，梯度等价（第 2 节）。不等价的地方：BatchNorm 这类跨样本统计的层（LLM 不用）；以及每个 micro-batch 各自平均造成的权重偏差。还有一个工程差别：前 $K-1$ 个 micro-batch 要用 `no_sync()`，否则每个 micro-batch 都 all-reduce 一次，通信白白多 $K$ 倍。
:::

::: details Q：为什么先裁剪再 optimizer.step，裁剪放在 all-reduce 之前行不行？
不行。裁剪用的是全局梯度的 norm，all-reduce 之前每个 rank 的梯度只是本地 micro-batch 的，各 rank 算出的 norm 和缩放系数不同，同步后的梯度就不是任何一种正确裁剪的结果。顺序是：同步 → 求全局 norm（切片时跨 rank 求平方和）→ 统一缩放 → step。
:::

::: details Q：Adam 的偏差修正去掉会怎样？
前几步更新量失真。$\beta_1 = 0.9$、$\beta_2 = 0.999$ 时第一步 $m_1/\sqrt{v_1} = 0.1/\sqrt{0.001} = 3.16$ 倍正常步长；$\beta_2 = 0.95$ 时是 0.447 倍。$t$ 到几百步后 $\beta^t$ 趋于 0，修正不再起作用。有 warmup 时前几步 lr 很小，影响被掩盖，但仍然该加。
:::

::: details Q：AdamW 和 Adam + L2 正则有什么区别？
L2 把 $\lambda\theta$ 加进梯度，会被 $\sqrt{\hat v}$ 除，历史梯度大的参数几乎不被衰减。AdamW 把衰减从梯度里拿出来，直接 $\theta \leftarrow \theta - \eta\lambda\theta$，所有参数按同样比例衰减。PyTorch 的 `torch.optim.Adam(weight_decay=...)` 是 L2，`AdamW` 才是 decoupled。
:::

::: details Q：为什么 LLM 用 β2 = 0.95？
Adam 单步更新量的上界：$1-\beta_1 > \sqrt{1-\beta_2}$ 时是 $\eta(1-\beta_1)/\sqrt{1-\beta_2}$，0.999 时 3.16$\eta$；0.95 时 $\sqrt{0.05} > 0.1$，上界是 $\eta$。而且 $v$ 的记忆从约 1000 步缩到约 20 步，梯度尺度突变时 $v$ 跟得上。代价是 $v$ 噪声更大。GPT-3、Llama 2 都用 0.95。
:::

::: details Q：WSD 相比 cosine 好在哪？
cosine 的形状绑定总步数 $T$，中途 checkpoint 处在「lr 降了一部分」的状态，既不是收敛好的模型，也不适合续训。WSD 的 stable 段 lr 恒定，任意 checkpoint 都能：(1) 分出去 decay 约 10% 的步数得到一个收敛的模型；(2) 接着用峰值 lr 续训。所以适合不确定总 token 数的 continual training，也能用一条主线测多个数据量下的 loss。
:::

::: details Q：batch 越大越好吗？
不是。超过 critical batch size 后，再加 batch 步数几乎不减，算力白花：$B = 4B_{\text{crit}}$ 时数据量是最优的 5 倍，步数只比 $B_{\text{crit}}$ 少 37.5%。$B_{\text{noise}}$ 随训练增大，所以 GPT-3、Llama 3 都在早期用小 batch、之后再加大。
:::

::: details Q：Muon 为什么不能直接套在 ZeRO-1 上？
ZeRO-1 把优化器状态按元素切到各 DP rank，AdamW 逐元素更新所以没问题。Muon 的 Newton–Schulz 要整块矩阵乘法，单个切片算不了。Moonlight 的做法是更新前在 DP 组内 gather 出完整矩阵（bf16），算完只留本地那片，额外通信约为 AdamW 的 0–25%（作者估计）。
:::

## 手撕

手写 AdamW 和 warmup-cosine，与 `torch.optim.AdamW` + `LambdaLR` 逐步对比：

```python
import math
import torch

def lr_at(step, peak, warmup, total, min_ratio=0.1):
    """step 从 0 开始数，对应公式里的 t = step + 1"""
    t = step + 1
    if t <= warmup:
        return peak * t / warmup                              # 线性 warmup
    p = (t - warmup) / max(1, total - warmup)                 # cosine 进度 0 → 1
    lo = peak * min_ratio
    return lo + 0.5 * (peak - lo) * (1 + math.cos(math.pi * min(p, 1.0)))

class MyAdamW:
    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = list(params)
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t                           # 偏差修正分母
        bc2 = 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            p.mul_(1 - self.lr * self.wd)                     # decoupled weight decay：不进 m、v
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)        # m = b1 m + (1-b1) g
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2) # v = b2 v + (1-b2) g^2
            m_hat, v_hat = m / bc1, v / bc2
            p.add_(-self.lr * m_hat / (v_hat.sqrt() + self.eps))

torch.manual_seed(0)
w_ref = torch.randn(4, 3, requires_grad=True)
w_my = w_ref.detach().clone().requires_grad_(True)
x, y = torch.randn(8, 3), torch.randn(8, 4)

peak, warmup, total = 1e-2, 3, 10
ref = torch.optim.AdamW([w_ref], lr=peak, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
sched = torch.optim.lr_scheduler.LambdaLR(ref, lambda s: lr_at(s, peak, warmup, total) / peak)
my = MyAdamW([w_my], lr=peak)

for step in range(total):
    for w in (w_ref, w_my):
        ((x @ w.T - y) ** 2).mean().backward()
    my.lr = lr_at(step, peak, warmup, total)
    ref.step(); sched.step(); my.step()
    ref.zero_grad(); w_my.grad = None
    print(step, f"{my.lr:.5f}", f"{(w_ref - w_my).abs().max().item():.1e}")
# 0 0.00333 0.0e+00
# 1 0.00667 0.0e+00
# 2 0.01000 0.0e+00      ← warmup 结束，到峰值
# 3 0.00955 6.0e-08
# ...
# 9 0.00100 6.0e-08      ← t = T，降到 min_ratio × peak
```

差值是 fp32 舍入误差（torch 2.5.1 CPU 实测）：PyTorch 内部先算 $\sqrt{v}/\sqrt{1-\beta_2^t}$ 再加 $\epsilon$、把 $1/(1-\beta_1^t)$ 并进步长，数学上和上面一样，运算顺序不同。

常见变体：加梯度裁剪（`step` 前对所有 `p.grad` 求全局 norm 再缩放）；weight decay 只加在 2D 矩阵上、不加在 bias 和 norm 的 scale 上（常见的参数分组）；改成 WSD。

## 参考

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)：更新公式、偏差修正、§2.1 的单步上界
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [On the Variance of the Adaptive Learning Rate and Beyond (RAdam)](https://arxiv.org/abs/1908.03265)：warmup 的解释
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) 附录 B：Adam 超参、cosine、batch ramp-up
- [Llama 2](https://arxiv.org/abs/2307.09288) §2.2、[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) §3.4.1
- [MiniCPM (WSD)](https://arxiv.org/abs/2404.06395) §4
- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)：gradient noise scale、critical batch size
- [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322)：warmup 与 lr 敏感度
- [Muon (Keller Jordan blog)](https://kellerjordan.github.io/posts/muon/)、[Muon is Scalable for LLM Training (Moonlight)](https://arxiv.org/abs/2502.16982)
- [PyTorch `clip_grad_norm_` 源码](https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/clip_grad.py)、[Megatron-LM `clip_grads.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/clip_grads.py)
- [HF blog: Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation)
