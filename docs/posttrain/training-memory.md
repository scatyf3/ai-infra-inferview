---
title: 混合精度、Grad Checkpointing 与优化器状态显存账
status: draft
tags: [mixed-precision, checkpointing, optimizer, activation-memory]
difficulty: 3
order: 5
related: [/parallel/zero-fsdp, /posttrain/lora-qlora, /inference/memory-accounting, /inference/flash-attention]
stack: []
---

# 混合精度、Grad Checkpointing 与优化器状态显存账

> 训练显存的完整账本

## 一句话结论

训练显存 = **模型状态**（参数、梯度、优化器状态，和数据无关，只看参数量）+ **激活**（forward 留给 backward 用的中间结果，随 batch、序列长度、层数涨）+ 临时 buffer 与碎片。混合精度 Adam 下模型状态是**每参数 16 字节**，7B 就是 112 GB；激活按 [Korthikanti et al. 2022](https://arxiv.org/abs/2205.05198) 的公式是每层 $sbh(34 + 5as/h)$ 字节，7B、seq 4k、micro-batch 1 时约 104 GB。FlashAttention 去掉 $s^2$ 项后剩约 18 GB，gradient checkpointing 再压到约 1 GB，代价是多算一次 forward（约 +33% 计算）。

## 推导

### 记号约定

| 符号 | 含义 | 7B（Llama-2-7B 结构）取值 |
|---|---|---|
| $\Psi$ | 参数量 | $7\times10^9$ |
| $L$ | Transformer 层数 | 32 |
| $h$ | hidden size | 4096 |
| $a$ | attention head 数 | 32 |
| $s$ | 序列长度 | 4096 |
| $b$ | micro-batch size（一次 forward 的样本数） | 1 |
| $V$ | 词表大小 | 32000 |
| $t$ | tensor parallel 度 | 1 |

单位：本页 GB 指 $10^9$ 字节，和 [ZeRO / FSDP](/parallel/zero-fsdp) 页的 $7 \times 16 = 112$ GB 保持一致。

### 第 0 步：训练比推理多存了什么

推理只要权重（加 KV cache，见 [推理显存账](/inference/memory-accounting)）。训练一步是 forward → backward → optimizer step，多出三类东西：

1. **梯度**：backward 给每个参数算一个梯度，和参数一样大。
2. **优化器状态**：Adam 给每个参数维护一阶动量 $m$（梯度的滑动平均）和二阶动量 $v$（梯度平方的滑动平均）。
3. **激活**：backward 算梯度要用 forward 的中间结果（比如 $y = Wx$ 的 $\partial L/\partial W = \delta \cdot x^\top$ 需要 $x$），所以 forward 时不能算完就扔。

前两类加上参数本身叫**模型状态**，只看 $\Psi$；激活只看 $s, b, h, L$。两者分开算。

### 模型状态：每参数 16 字节是怎么来的

| 东西 | 精度 | 字节 | 干什么用 |
|---|---|---|---|
| 参数 | bf16 | 2 | forward / backward 的矩阵乘 |
| 梯度 | bf16 | 2 | backward 产出，喂给优化器 |
| 主权重（master weight） | fp32 | 4 | 优化器在这份上做更新 |
| Adam $m$ | fp32 | 4 | 优化器状态 |
| Adam $v$ | fp32 | 4 | 优化器状态 |
| **合计** | | **16** | |

这是 [ZeRO 论文](https://arxiv.org/abs/1910.02054) 的记法：$2\Psi + 2\Psi + K\Psi$，Adam 的 $K = 12$。7B：$7 \times 16 = 112$ GB，一张 80 GB 卡放不下，还没算激活。

不同实现会有出入，算之前先问清楚：
- Megatron 等框架把梯度累加在 fp32 的 `main_grad` 里，梯度变成 4 字节，合计 18。
- PyTorch AMP（`autocast`）的默认用法是参数本身就存 fp32、算的时候临时 cast：fp32 参数 4 + fp32 梯度 4 + $m$ 4 + $v$ 4，也是 16，只是没有单独的「bf16 参数」。
- 8-bit Adam（[Dettmers et al. 2021](https://arxiv.org/abs/2110.02861)）把 $m$、$v$ 各压到 1 字节：$2+2+4+1+1 = 10$。
- SGD 无动量：$K = 4$（只有主权重），合计 8。

### 为什么要留一份 fp32 主权重

[Mixed Precision Training](https://arxiv.org/abs/1710.03740) 提出的做法：计算用半精度，更新在 fp32 主权重上做，再 cast 回半精度给下一步 forward 用。原因是**更新量太小，半精度加不上去**。

浮点数在 1.0 附近能分辨的最小间隔（ulp）是 $2^{-\text{尾数位数}}$：

| 格式 | 指数位 | 尾数位 | 1.0 附近的 ulp | 最小正规数 |
|---|---|---|---|---|
| fp32 | 8 | 23 | $2^{-23} \approx 1.2\times10^{-7}$ | $\approx 1.2\times10^{-38}$ |
| bf16 | 8 | 7 | $2^{-7} \approx 7.8\times10^{-3}$ | $\approx 1.2\times10^{-38}$ |
| fp16 | 5 | 10 | $2^{-10} \approx 9.8\times10^{-4}$ | $2^{-14} \approx 6.1\times10^{-5}$ |

自己算一下：参数 $w = 1.0$，lr $= 10^{-5}$，梯度 $\approx 1$，更新量 $10^{-5}$。bf16 里比半个 ulp（$\approx 3.9\times10^{-3}$）小得多，$1.0 + 10^{-5}$ 舍入后还是 $1.0$，**这一步白更新**；每步都这样，参数就永远不动。fp32 的 ulp 是 $10^{-7}$ 量级，能把这些小更新累积起来。

所以主权重这 4 字节买的是「更新不丢」，$m$、$v$ 用 fp32 也是同一个理由（$v$ 是梯度平方，数值更小）。

### fp16 要 loss scaling，bf16 不要

fp16 的问题不只是精度，还有**范围**：指数位只有 5 位，最小正规数 $6.1\times10^{-5}$，最小非正规数 $2^{-24} \approx 6\times10^{-8}$，再小就下溢成 0。[Micikevicius et al.](https://arxiv.org/abs/1710.03740) 统计了激活梯度的分布，发现有相当一部分落在 fp16 表示范围之下（作者自测）。

Loss scaling：反向前把 loss 乘一个大数 $S$（比如 $2^{16}$），由链式法则所有梯度都放大 $S$ 倍，挪进 fp16 的表示范围；更新前在 fp32 里除回 $S$。动态版本：梯度出现 inf/NaN 就跳过这一步并把 $S$ 减半，连续若干步正常就加倍。

bf16 的指数位和 fp32 一样是 8 位，范围相同，小梯度不会下溢，所以**不需要 loss scaling**（[Kalamkar et al. 2019](https://arxiv.org/abs/1905.12322) 的实验里 bf16 不改超参就能训到和 fp32 相当的精度，作者自测）。代价是尾数只有 7 位，精度更差，这正是上一节靠 fp32 主权重补的东西。

### 激活：一层存了哪些张量

[Korthikanti et al. 2022](https://arxiv.org/abs/2205.05198) 对 GPT 式结构（LayerNorm、4h 的 GeLU MLP、带 dropout）逐个数了 backward 要用的张量。约定：激活存 16 位（每元素 2 字节），dropout mask 每元素 1 字节；$sbh$ 是一个 $[s, b, h]$ 张量的元素数，所以一个 bf16 的 $[s,b,h]$ 张量是 $2sbh$ 字节。

| 位置 | 存的东西 | 字节 |
|---|---|---|
| Attention | QKV 线性层的输入 $x$ | $2sbh$ |
| | $QK^\top$ 要用的 $Q$、$K$ | $4sbh$ |
| | softmax 输出（每个 head 一张 $s\times s$） | $2as^2b$ |
| | softmax 后 dropout 的 mask | $as^2b$ |
| | 乘 $V$ 要用的 dropout 输出和 $V$ | $2as^2b + 2sbh$ |
| | 输出投影的输入 | $2sbh$ |
| | 输出 dropout 的 mask | $sbh$ |
| | **小计** | $11sbh + 5as^2b$ |
| MLP | 两个线性层的输入（$h$ 维、$4h$ 维） | $2sbh + 8sbh$ |
| | GeLU 的输入（$4h$ 维） | $8sbh$ |
| | dropout mask | $sbh$ |
| | **小计** | $19sbh$ |
| LayerNorm ×2 | 各自的输入 | $4sbh$ |

加起来就是论文的式 (1)：

$$
\text{每层激活} = sbh\left(34 + \frac{5as}{h}\right) \ \text{字节}
$$

两项的含义要分清：
- $34sbh$：和 $s$ **线性**，来自各种 $[s, b, h]$ 或 $[s, b, 4h]$ 张量。
- $5as^2b$：和 $s$ **平方**，来自 $a$ 张 $s \times s$ 的 attention 矩阵（softmax 输出、dropout mask、dropout 输出）。$5as/h$ 就是它和线性项的比值；7B、$s=4096$ 时 $5 \times 32 \times 4096 / 4096 = 160$，是线性项 34 的近 5 倍。

再乘层数 $L$ 就是总激活（论文式 (5)，$t=1$ 时）。embedding、最后的 LayerNorm、输出 logits 不在式里，单独算：logits 是 $[s, b, V]$，bf16 下 $2sbV$ 字节，7B、$s=4096$ 是 0.26 GB，cross-entropy 若 upcast 到 fp32 再翻倍。

::: tip Llama 式结构的常数
Llama 没有 dropout、用 RMSNorm 和 SwiGLU（中间维 $i = 11008 \approx 2.69h$），同样的方法自己数（非论文数字）：attention 去掉两个 mask 和 dropout 输出，剩 $10sbh + 2as^2b$；SwiGLU 存 gate/up 共享输入 $2sbh$、gate 输出 $2sbi$、up 输出 $2sbi$、down 的输入 $2sbi$，约 $2sbh + 6sbi \approx 18sbh$；两个 norm $4sbh$。合计约 $32sbh + 2as^2b$。和 34 差得不多，下面的例子直接用论文公式。
:::

### FlashAttention 为什么能去掉 $s^2$ 项

[FlashAttention](https://arxiv.org/abs/2205.14135) forward 分块算 softmax，从不把 $s \times s$ 矩阵写回显存，只存输出 $O$ 和每行的 logsumexp（$a \cdot s \cdot b$ 个 fp32，相对 $sbh$ 可忽略）；backward 时用 $Q, K, V$ 和 logsumexp 分块**重算** attention 矩阵。所以 $5as^2b$ 整项消失，每层只剩约 $34sbh$。

这和下面的 selective recomputation 是同一件事：attention 矩阵占显存大、重算 FLOPs 少，最值得重算。见 [FlashAttention](/inference/flash-attention)。

### Gradient checkpointing：用计算换显存

思路：forward 时只**存每层的输入**（checkpoint），其他中间结果算完就扔；backward 到某层时，从它的输入**重新跑一遍这一层的 forward**，把中间结果临时重建出来，再算梯度。

```python
# full recomputation，每层一个 checkpoint
def forward(x):
    saved = []
    for layer in layers:
        saved.append(x)                      # 只存层输入：2sbh 字节
        with torch.no_grad():
            x = layer(x)                     # 中间激活不留
    return x, saved

def backward(grad, saved):
    for layer, x_in in reversed(list(zip(layers, saved))):
        x_in.requires_grad_(True)
        y = layer(x_in)                      # 重算这一层 forward，临时重建激活
        y.backward(grad)                     # 算这一层的参数梯度和输入梯度
        grad = x_in.grad                     # 用完这一层的激活就释放
```

PyTorch 里对应 `torch.utils.checkpoint.checkpoint(layer, x)`，HF 模型里是 `model.gradient_checkpointing_enable()`。

**显存**：常驻激活从 $L \cdot sbh(34 + 5as/h)$ 降到 $2sbhL$，外加 backward 时**一层**的完整激活（重建出来、用完即扔）。峰值约为

$$
2sbhL + sbh\left(34 + \frac{5as}{h}\right)
$$

**计算**：训练每 token 约 $6\Psi$ FLOPs（forward $2\Psi$，backward $4\Psi$，backward 要对输入和权重各求一次梯度）。full recomputation 多一次 forward，变成 $8\Psi$，**约 +33%**。Korthikanti et al. 实测 full recomputation 开销 30–40%（作者自测；22B 模型单层 39%，530B / 1T 为 36%）。

**粒度**：不必每层都 checkpoint。[Chen et al. 2016](https://arxiv.org/abs/1604.06174) 指出每 $\sqrt{L}$ 层存一个 checkpoint，激活显存降到 $O(\sqrt{L})$，计算仍只多约一次 forward。实践中常按显存余量只给一部分层开 checkpoint。

### Full vs selective recomputation

[Korthikanti et al.](https://arxiv.org/abs/2205.05198) 的观察：$5as^2b$ 那部分（$QK^\top$、softmax、dropout、乘 $V$）显存占大头，但每个元素的 FLOPs 很少；$34sbh$ 那部分来自矩阵乘，重算贵。所以**只重算 attention 核心那一段**，其余照常存：

| 方案 | 每层常驻激活 | 额外计算 |
|---|---|---|
| 不重算 | $sbh(34 + 5as/h)$ | 0 |
| selective recomputation | $34sbh$ | 只重算 attention 核心；GPT-3 估算为 2.7% FLOPs（论文附录 A） |
| full recomputation | $2sbh$ | 一次完整 forward，约 +33% |

论文实测（作者自测）：22B 模型单层 forward+backward，selective 开销 7%，full 39%；配合 sequence parallel 后 selective 降到 4%。有 TP / SP 时每层激活还会再除以 $t$，完整表见论文 Table 2，TP / SP 本身见 [Megatron TP](/parallel/megatron-tp)。

用了 FlashAttention 就等于已经白拿了 selective recomputation，再要省只能上 full。

### 算一笔：7B 全参微调，seq 4096，micro-batch 1，单卡

模型状态：$7\times10^9 \times 16 = 112$ GB。

激活（$sbh = 4096 \times 1 \times 4096 = 1.68\times10^7$）：

| 项 | 公式 | 每层 | ×32 层 |
|---|---|---|---|
| 线性项 | $34sbh$ | 0.57 GB | 18.3 GB |
| 平方项 | $5as^2b = 5 \times 32 \times 4096^2$ | 2.68 GB | 85.9 GB |
| 合计（不重算） | | 3.25 GB | **104.2 GB** |
| FlashAttention / selective | $34sbh$ | 0.57 GB | **18.3 GB** |
| full recomputation | $2sbh$ | 0.034 GB | **1.07 GB**（+ 重算时一层的 0.57 GB） |

再加 logits 约 0.26–0.5 GB。汇总：

| 配置 | 模型状态 | 激活 | 合计（不含 buffer） |
|---|---|---|---|
| 朴素 attention，不重算 | 112 | 104 | ~216 GB |
| FlashAttention | 112 | 18 | ~131 GB |
| FlashAttention + full checkpointing | 112 | ~1.6 | ~114 GB |
| 上一行 + ZeRO-3 8 卡 | 14 | ~1.6 | ~16 GB / 卡 |

结论：
- seq 4k 下，不用 FlashAttention 时激活和模型状态一样大，$s^2$ 项占了激活的 80%。
- 把激活压干净以后，单卡依然放不下 112 GB 的模型状态。激活省不了模型状态，模型状态要靠 ZeRO 切。
- 序列翻倍：平方项 ×4、线性项 ×2。$s=8192$ 时不重算要 380 GB 激活，FlashAttention 下 36.5 GB。

### 通用公式

单卡显存（不含 buffer），数据并行度 $N$：

$$
M \approx \underbrace{\Psi \cdot B_{\text{state}}}_{\text{模型状态}} + \underbrace{L \cdot sbh \cdot c}_{\text{激活}} + \underbrace{2sbV}_{\text{logits}}
$$

- $B_{\text{state}}$：每参数字节数，按下节的表取。
- $c$：每层激活系数。不重算 $34 + 5as/h$；FlashAttention 或 selective $34$；full recomputation $2$（另加重算时一层的峰值）。
- 有 TP 度 $t$ 且开 sequence parallel 时，$c$ 再除以 $t$。
- **激活和 $b$ 成正比**：gradient accumulation 就是用小 $b$ 多跑几次 forward/backward 再更新，等效 batch 不变，激活按小 $b$ 算。

### ZeRO / FSDP、LoRA 怎么改这本账

只改 $B_{\text{state}}$，**不改激活**：

| 方案 | $B_{\text{state}}$（每参数） | 7B 时 |
|---|---|---|
| 全参，DDP | 16 | 112 GB |
| ZeRO-1，$N$ 卡 | $4 + 12/N$ | $N=8$：38.5 GB |
| ZeRO-2 | $2 + 14/N$ | 26.3 GB |
| ZeRO-3 / FSDP | $16/N$ | 14 GB |
| LoRA（bf16 底座冻结） | $2 + 16 \cdot r_{\text{frac}}$ | ≈ 14.3 GB |
| QLoRA（NF4 底座） | $\approx 0.5 + 16 \cdot r_{\text{frac}}$ | ≈ 3.8 GB + 量化常数 |

ZeRO 各级的推导见 [ZeRO / FSDP](/parallel/zero-fsdp)。

LoRA（[Hu et al. 2021](https://arxiv.org/abs/2106.09685)）：底座冻结，只存 bf16 权重 2 字节、不需要梯度和优化器状态；只有 adapter 参数要 16 字节全套。$r_{\text{frac}}$ 是可训练参数占比。例：$r=16$，加在 $W_q, W_k, W_v, W_o$ 上，每个矩阵加 $r(h + h)$ 个参数，共 $4 \times 16 \times 8192 \times 32 = 1.68\times10^7$，占 0.24%，16 字节全套才 0.27 GB。QLoRA（[Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)）把底座量化到 4 bit，底座从 14 GB 降到约 3.5 GB。

**LoRA 不省激活**：backward 仍要穿过每一层把梯度传回 adapter，各层的输入照样要存。所以 LoRA 之后的大头往往变成激活，还得靠 FlashAttention 和 checkpointing。见 [LoRA / QLoRA](/posttrain/lora-qlora)。

### 临时 buffer 与 allocator 碎片

公式之外还有：
- DDP 梯度 all-reduce 的 bucket、ZeRO-3 预取的下一层完整参数、cuBLAS / attention 的 workspace。
- **PyTorch caching allocator 的碎片**。PyTorch 不会每次都 `cudaFree`，释放的块留在缓存池里给下次复用（[CUDA semantics: Memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)）。`torch.cuda.memory_allocated()` 是张量实际占用，`memory_reserved()` 是 allocator 手里的总量，两者之差就是缓存加碎片，`nvidia-smi` 看到的是后者。序列长度每步变化时，不同大小的块被切碎，可能出现「总空闲够、但没有一块连续的够大」而 OOM。缓解：设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（让 segment 可以扩展，减少这类碎片）、`max_split_size_mb`、按长度分桶 padding 让形状更稳定。

## 面试追问

::: details Q：为什么 bf16 训练不需要 loss scaling 而 fp16 需要？
fp16 的指数位只有 5 位，最小正规数约 $6\times10^{-5}$，小梯度会下溢成零，所以要把 loss 乘一个大数 $S$，梯度跟着放大 $S$ 倍挪进可表示范围，更新前在 fp32 里除回来。bf16 的指数位和 fp32 一样是 8 位，动态范围够大不会下溢，代价是尾数只有 7 位、精度低，靠 fp32 主权重补回来。
:::

::: details Q：既然 bf16 范围够，主权重能不能也用 bf16，省 2 字节？
直接用会丢更新：bf16 在 1.0 附近的 ulp 是 $2^{-7} \approx 0.0078$，lr $\times$ grad 在 $10^{-5}$ 量级时加上去会被舍入掉。纯 bf16 训练需要额外手段，比如 stochastic rounding，或者用 Kahan summation 维护一个补偿项，这些不在标准 16 字节账里。
:::

::: details Q：gradient checkpointing 为什么开销是「多一次 forward」而不是两倍？
训练每 token 约 $6\Psi$ FLOPs，forward 只占 $2\Psi$，backward 占 $4\Psi$（对输入和对权重各一次矩阵乘）。重算只多一次 forward，所以是 $8\Psi / 6\Psi \approx 1.33$。Korthikanti et al. 实测 full recomputation 30–40%（作者自测）。
:::

::: details Q：开了 FlashAttention 还要 gradient checkpointing 吗？
看余量。FlashAttention 只去掉了 $5as^2b$ 的平方项，还剩 $34sbh \cdot L$，7B、seq 4k 是 18 GB，batch 变大或序列更长还会线性涨。放不下时再开 full recomputation（或只给部分层开），降到 $2sbhL$。
:::

::: details Q：ZeRO-3 能省激活吗？LoRA 呢？
都不能。两者改的都是模型状态那一项。激活和 $s, b, h, L$ 有关，要靠 FlashAttention、checkpointing、sequence parallel / context parallel，或者减小 micro-batch 加 gradient accumulation。
:::

::: details Q：为什么 nvidia-smi 显示占用比我算的大？
nvidia-smi 看到的是 CUDA context（几百 MB）加 PyTorch allocator 保留的全部内存（`memory_reserved`），里面含缓存起来没还的块和碎片，不只是张量实际占用（`memory_allocated`）。用 `torch.cuda.memory_summary()` 或 memory snapshot 看具体分布。
:::

::: details Q：micro-batch 从 1 改到 4，显存涨多少？
模型状态不变，激活和 logits 线性 ×4。7B、seq 4k、FlashAttention 不重算时激活从 18 GB 到 73 GB。所以 OOM 时第一个旋钮是减小 micro-batch、用 gradient accumulation 保持等效 batch。
:::

## 手撕

给定配置算单卡显存，照「通用公式」节写成函数：

```python
def train_mem_gb(P, L, h, a, s, b, V, N=1, t=1,
                 zero=0, flash=True, recompute="none", lora_frac=None):
    # 模型状态：每参数字节
    if lora_frac is not None:                  # LoRA：底座 bf16 冻结 + adapter 全套
        state = 2 + 16 * lora_frac
    else:
        state = {0: 16, 1: 4 + 12 / N, 2: 2 + 14 / N, 3: 16 / N}[zero]
    model = P * state

    # 激活：每层系数 c（Korthikanti et al. 2022 式 (1)）
    sbh = s * b * h
    if recompute == "full":
        c, peak = 2, sbh * (34 + (0 if flash else 5 * a * s / h)) / t
    else:
        c = 34 + (0 if flash or recompute == "selective" else 5 * a * s / h)
        c, peak = c / t, 0
    act = L * sbh * c + peak
    logits = 2 * s * b * V
    return (model + act + logits) / 1e9

# 7B，seq 4k，micro-batch 1
print(train_mem_gb(7e9, 32, 4096, 32, 4096, 1, 32000, flash=False))      # ~216
print(train_mem_gb(7e9, 32, 4096, 32, 4096, 1, 32000))                   # ~131
print(train_mem_gb(7e9, 32, 4096, 32, 4096, 1, 32000, recompute="full")) # ~114
print(train_mem_gb(7e9, 32, 4096, 32, 4096, 1, 32000, N=8, zero=3,
                   recompute="full"))                                     # ~16
```

常见变体：问 70B 在 64 卡 ZeRO-3 下每卡多少、LoRA 后激活是不是变成大头、seq 从 4k 到 32k 时哪一项先爆。推理侧的账见 [显存账](/inference/memory-accounting)。

## 参考

- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)：fp32 主权重、loss scaling
- [A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)：每层激活公式、selective recomputation
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)：gradient checkpointing
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)：16Ψ 的记法
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861)
- [LoRA](https://arxiv.org/abs/2106.09685)、[QLoRA](https://arxiv.org/abs/2305.14314)
- [PyTorch CUDA semantics: Memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)
