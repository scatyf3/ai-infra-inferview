---
title: SFT 训练提速：时间和显存都花在哪，按什么顺序优化
status: draft
tags: [sft, mfu, fused-kernel, cross-entropy, liger]
difficulty: 3
order: 10
related: [/train/compute-mfu, /train/parallel-config, /posttrain/sft, /posttrain/lora-qlora, /posttrain/training-memory, /parallel/zero-fsdp, /framework/torch-compile]
stack: []
train: [tr-r-sft, tr-k-fused]
---

# SFT 训练提速：时间和显存都花在哪，按什么顺序优化

> 先算理论上限，再一项项找浪费：padding、logits、小算子、recompute、并行配置

## 一句话结论

SFT 每 token 的算力和预训练一样是 $6N$，8B 在 8 张 H100 上的上限由 $n_{\text{gpu}} \cdot \text{peak} \cdot u / 6N$ 决定，MFU 40% 时约 6.6 万 token/s。实际跑不到的原因按影响从大到小：

1. **padding**：不 packing 时有效 token 比例常在 60% 以下，直接浪费算力。packing 后到 99%。
2. **logits 显存**：Llama 3 词表 128256，一个 8192 token 的 micro-batch 的 logits 在 fp32 下 4.2 GB，算 loss 时同时有好几份，比开了 recompute 的全部层激活还大。fused linear cross-entropy 分块算，不物化完整 logits。
3. **小算子**：RMSNorm、RoPE、SwiGLU 是 memory-bound 的，用融合 kernel 或 `torch.compile`。
4. **recompute**：full recompute 让每 token 变成 $8N$，省下的显存优先拿来关掉它。

顺序是先量（有效 token 比例、峰值显存、profile），再按上面的顺序改，最后调并行配置。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $N$ | 模型参数量 | Llama 3 8B：$8.03\times10^9$ |
| $n_{\text{gpu}}$ | GPU 数 | 8 |
| peak | 单卡 bf16 dense 峰值 | H100 SXM：989 TFLOP/s |
| $u$ | MFU = 按 $6N$ 算的有效 FLOP/s ÷ peak | 0.4 |
| $h$ | hidden size | 4096 |
| $L$ | 层数 | 32 |
| $V$ | 词表大小 | 128256 |
| $s$ | 一个 micro-batch 的 token 数（packing 后） | 8192 |
| $l_i$ | 第 $i$ 条样本的真实长度 | 800 |
| $\eta$ | 有效 token 比例 = 真实 token ÷ 实际计算的位置数 | 0.6 |

单位：GB = $10^9$ 字节。

### 1. 先算理论吞吐

训练每 token $6N$ FLOPs：forward $2N$；backward $4N$，对输入求梯度（dgrad）和对权重求梯度（wgrad）各一次同样大小的矩阵乘。推导和 MFU 的定义见 [训练算力账](/train/compute-mfu)。

$$
\text{tokens/s} = \frac{n_{\text{gpu}} \cdot \text{peak} \cdot u}{6N}
$$

8B、8 张 H100：

| $u$ | 每卡 token/s | 8 卡 token/s |
|---|---|---|
| 1.0（不可能达到的上限） | 20500 | 16.4 万 |
| 0.4 | 8200 | 6.6 万 |

实测参考：torchtitan 在 8 张 H100 上训 Llama 3.1 8B（FSDP + selective activation checkpointing，local batch 2），每卡 6258 token/s，加 `torch.compile` 到 6674（+6.6%）；论文说 8B 在 8 或 128 张 H100 上不用 fp8 时 MFU 为 33–42%（[TorchTitan](https://arxiv.org/abs/2410.06511) Table 1，作者自测；他们的 H100 是 95 GiB 版本）。这是预训练数字，SFT 每 token 的算力相同，差别在 padding 和 loss mask 带来的浪费。

用法：先拿这个公式和你的 tokens/s 比，算出自己的 $u$。$u$ 远低于 0.3 就说明有大的浪费，按下面的顺序找。公式忽略了 attention 的 FLOPs，长序列时要加上（见 [SFT](/posttrain/sft) 第 6 节）。

### 2. 浪费一：padding

$$
\eta = \frac{\sum_i l_i}{\text{实际计算的位置数}}
$$

不 packing 时，一个 micro-batch 的 $B$ 条样本 pad 到组内最长 $l_{\max}$，分母是 $B \cdot l_{\max}$。

例：4 条样本长度 200、300、500、2000，pad 到 2000：$\eta = 3000 / 8000 = 37.5\%$，62.5% 的算力花在 pad 上。padding 位置的线性层照算，attention 的 mask 只是把分数置 $-\infty$。

手撕节的模拟：1000 条长度均匀分布在 100–2047 的样本，每 8 条 pad 一次，$\eta = 58.7\%$；first-fit 装进 8192 的箱子，$\eta = 99.3\%$。真实 SFT 数据是长尾分布，不 packing 时更差。

怎么做 packing、配套的 `cu_seqlens` 和 position_ids，见 [SFT](/posttrain/sft) 第 4、5 节。这里只要求**每步把 $\eta$ 打到日志里**：`attention_mask.sum() / input_ids.numel()`，或 packing 后 `cu_seqlens[-1] / input_ids.numel()`。

### 3. 浪费二：logits 显存

最后一层 hidden $x \in \mathbb{R}^{s \times h}$ 乘 lm_head $W \in \mathbb{R}^{V \times h}$ 得到 logits $\in \mathbb{R}^{s \times V}$。一份 logits 的大小：

$$
M_{\text{logits}} = s \cdot V \cdot \text{每元素字节}
$$

Llama 3 8B，$s = 8192$：$8192 \times 128256 = 1.05\times10^9$ 个元素，bf16 是 2.1 GB，**fp32 是 4.2 GB**。

HF 的 `ForCausalLMLoss` 会先 `logits.float()` 再调 `F.cross_entropy`（内部是 log_softmax + nll）。按代码数前向峰值同时存在的张量（自己数，非实测）：

1. bf16 logits，作为模型输出返回：$2sV$
2. fp32 副本：$4sV$
3. log_softmax 的输出，留给 backward：$4sV$

合计约 $10sV$ 字节，$s = 8192$ 时 **10.5 GB**；backward 还要再分配 fp32 的 logits 梯度。对比 8B 其余各层的激活：开 full recompute 后只剩 $2shL = 2.1$ GB（[训练显存账](/posttrain/training-memory)）。也就是说**开了 recompute 以后，logits 是最大的一块**。

**Fused linear cross-entropy**：把 lm_head 的矩阵乘和 cross-entropy 一起沿 token 维分块做，任何时候只有一块的 logits：

1. 取一块 token $x_c \in \mathbb{R}^{c \times h}$，算 $z = x_c W^\top$（$c \times V$，fp32）。
2. 算 $\text{lse} = \log\sum_v e^{z_v}$，loss 累加 $\text{lse} - z_{y}$。
3. 当场算梯度：$\partial \ell / \partial z = \text{softmax}(z) - \text{onehot}(y)$，原地写回 $z$ 的内存，除以全局有效 token 数。
4. $\partial \ell / \partial x_c = g W$，$\partial \ell / \partial W \mathrel{+}= g^\top x_c$。丢掉这一块，下一块。

因为梯度在 forward 里就算好了，backward 只是把存好的 $\partial x$、$\partial W$ 乘上上游梯度返回，lm_head 不用再做矩阵乘。代价是 loss 必须是 logits 之后的最后一步，并且拿不到完整 logits。

块多大：Liger 取 $c = 2^{\lceil \log_2 \lceil s / \lceil V/h \rceil \rceil \rceil}$，让一块 logits 的元素数 $c \cdot V \approx s \cdot h$，即和 hidden states 一样大。代入：$\lceil 128256 / 4096 \rceil = 32$，$8192 / 32 = 256$，$c = 256$，一块 fp32 logits $256 \times 128256 \times 4 = 131$ MB，比 10.5 GB 小两个数量级。

两个实现，作者自测的数字：

1. **Liger Kernel**（[Hsu et al. 2024](https://arxiv.org/abs/2410.10989)）：就是上面的做法（FusedLinearCrossEntropy）。论文举例 Gemma 词表 256k、batch 8、序列 4096 时 bf16 logits 16.8 GB。整套 Liger kernel 在 4 张 A100 上微调 Llama 3 8B（seq 512、batch 64）吞吐 +42.8%、显存 −54.8%，这是所有 kernel 合计，不是 FLCE 单独的（作者自测）。
2. **Cut Cross-Entropy**（[Wijmans et al. 2024](https://arxiv.org/abs/2411.09009)）：在 kernel 的片上内存里做矩阵乘和 logsumexp，只把正确 token 的 logit 写出；backward 跳过数值上可忽略的梯度元素。Gemma 2 2B 上 loss 计算的显存从 24 GB 降到 1 MB，分类头的训练显存从 28 GB 降到 1 GB，且不损失速度和收敛（作者自测）。

### 4. 浪费三：小算子

RMSNorm、RoPE、SwiGLU 的乘法和激活都是逐元素或按行归约，算术强度远低于 H100 的 ridge point（295 FLOP/B），是 memory-bound 的。eager 模式下每个算子一次 HBM 往返；融合成一个 kernel 后只读一次输入、写一次输出。字节数怎么数见 [torch.compile](/framework/torch-compile) 第 0 节的 RMSNorm 例子：约 3 次往返变 1 次。

Liger 的单 kernel 测试（A100，作者自测）：

| kernel | 速度 | 峰值显存 |
|---|---|---|
| RMSNorm（hidden 16384） | 约 7 倍快 | 约 1/3 |
| RoPE（hidden 16384） | 约 8 倍快 | 约 1/3 |
| SwiGLU（seq 16384） | 持平 | 约 1/1.6，backward 时重算 SiLU 而不存它的输出 |

这些单算子的倍数不等于端到端的倍数，因为大头时间在 GEMM 和 attention 里。端到端看第 3 节那组 +42.8% 的合计数字，或 torchtitan 的 `torch.compile` +6.6%（第 1 节）。

做法：HF 模型直接用 Liger 的补丁（`apply_liger_kernel_to_llama()`，替换 RMSNorm、RoPE、SwiGLU 和 loss），一次拿到第 3、4 节的收益；PyTorch 原生写的模型按 Transformer block 用 `torch.compile`。

### 5. 显存换吞吐：省下的显存先给谁

省下的显存有两个用处：关掉 recompute，或开更大的 micro-batch。判断规则：**先关 recompute，再加 micro-batch**。

**为什么不先加 micro-batch。** 线性层是 $[s, h] \times [h, k]$ 的 GEMM，FLOPs $2shk$，读写字节约 $2(hk + sh + sk)$。$s \ll h, k$ 时字节由权重 $2hk$ 主导，算术强度约为 $s$ FLOP/B。$s = 8192$ 远大于 H100 的 ridge point 295，GEMM 早就是 compute-bound，再加大 $s$ 每 token 也不会更快（roofline 见 [Prefill vs Decode](/inference/prefill-decode-roofline)）。加大 micro-batch 只摊薄**每个 micro-batch 的固定开销**：FSDP 每个 micro-batch 都要把参数 all-gather 两次、梯度 reduce-scatter 一次，和 token 数无关。

**为什么先关 recompute。** full recompute 让每 token 从 $6N$ 变 $8N$，关掉最多快 $8/6 = 1.33$ 倍，收益是确定的。可以按层逐步关：

1. 每少 checkpoint 一层，多存 $(34 - 2) sh$ 字节激活。8B、$s = 8192$：$32 \times 8192 \times 4096 = 1.07$ GB。
2. 每少 checkpoint 一层，省掉这一层的一次重算，即全部计算的 $\frac{1}{L} \cdot \frac{2}{8}$。$L = 32$ 时约 0.8%。

所以显存余量 $M$ GB 大约能关掉 $M / 1.07$ 层的 recompute。全部关掉以后还有余量，再加 micro-batch。

**按 token 数组 batch。** micro-batch 的大小用 token 数（比如 8192）而不是样本数定，packing 后自然如此。这样每步的激活和 logits 大小固定，显存峰值稳定，也减少 allocator 碎片（[SFT](/posttrain/sft) 第 7 节）。

### 6. 分布式选择：单机 8 卡全参 8B

模型状态按每参数 16 字节（[ZeRO / FSDP](/parallel/zero-fsdp)），$N = 8.03\times10^9$：

| 方案 | 每参数字节 | 每卡模型状态 |
|---|---|---|
| DDP | 16 | 128.5 GB，放不下 |
| ZeRO-2 | $2 + 14/8 = 3.75$ | 30.1 GB |
| ZeRO-3 / FSDP2 | $16/8 = 2$ | 16.1 GB |

加上激活（$s = 8192$，FlashAttention，不 recompute：$34shL = 36.5$ GB）和 logits：

1. ZeRO-2 + HF loss：$30.1 + 36.5 + 10.5 = 77$ GB，80 GB 卡还没算临时 buffer，放不下，只能开 recompute。
2. FSDP2 + fused CE：$16.1 + 36.5 + 0.1 \approx 53$ GB，不开 recompute 也有余量。

FSDP2 多出的通信：每个 micro-batch 参数 all-gather 两次、梯度 reduce-scatter 一次，每次每卡发 $\frac{7}{8} \times 2N$ 字节（bf16），共 $3 \times 0.875 \times 16 = 42$ GB，按 NVLink 单方向 450 GB/s 约 93 ms（带宽见 [集合通信](/parallel/collective-comm)）。同一个 micro-batch 的计算 $8192 \times 6N / (0.4 \times 989\text{T}) \approx 1.0$ s，通信约 9%，而且 FSDP 预取下一层参数，可以藏在计算后面。对比 recompute 的 33%。

规则：**能不开 recompute 放下就用 ZeRO-2，否则用 FSDP2，而不是 ZeRO-2 + recompute。** 更大规模的并行配置怎么搜见 [并行配置](/train/parallel-config)。

**什么时候要 TP / CP。** 8B 单机不需要 TP：参数靠 FSDP 切已经够小。需要切的是**单条序列的激活**放不下的时候，也就是长上下文：

1. $s = 32$K：不 recompute 激活 $34shL = 146$ GB，放不下；full recompute 后 $2shL = 8.6$ GB 加重算时一层的 4.6 GB，可以。但 HF loss 的 logits $10sV = 42$ GB，必须用 fused CE。
2. $s = 128$K：full recompute 后还有 34 GB 加一层 18 GB，再加模型状态就到顶了。这时用 CP 把序列切到多卡。

CP 的做法见 [长上下文训练](/train/long-context) 和 [并行总览](/parallel/parallelism-overview)。

### 7. LoRA vs 全参的吞吐

LoRA 冻结底座 $W_0$。backward 仍然要对每层算 dgrad（$\partial L/\partial x = W_0^\top g$），梯度才能传回下面各层的 adapter；省掉的只有冻结权重的 wgrad 那 $2N$：

| | forward | dgrad | wgrad | 合计 | 开 full recompute |
|---|---|---|---|---|---|
| 全参 | $2N$ | $2N$ | $2N$ | $6N$ | $8N$ |
| LoRA | $2N$ | $2N$ | ≈0（只有 adapter） | $4N$ | $6N$ |

所以每 token 的计算**最多省 1/3**（理论 1.5 倍），开 recompute 时只省 1/4。实测更少：LoRA 论文在 GPT-3 175B 上报告训练吞吐 +25%（作者自测）。对比显存：模型状态从 16 字节/参数降到约 2 字节/参数，省 8 倍。

结论：**LoRA 主要省的是显存，不是时间**。激活和 logits 一点没少（[LoRA / QLoRA](/posttrain/lora-qlora) 第 4 节），所以第 2–5 节的优化对 LoRA 同样适用；省下的模型状态显存按第 5 节的规则拿去关 recompute，才能把 1.33 倍的那部分拿回来。QLoRA 每次 GEMM 前要把 NF4 反量化，比 bf16 LoRA 更慢。

### 8. 按顺序的 checklist

每一步改完都重新量，改动和指标一一对应：

1. **量有效 token 比例**：日志打 $\eta$。低于 90% 就做下一步。
2. **packing**：看 $\eta$ 升到 95% 以上，tokens/s（只数真实 token）同比例上升。
3. **FlashAttention varlen**：profiler 里 attention 的 kernel 是 varlen 版本（传了 `cu_seqlens`）；看 attention 时间占比，FLOPs 从 $(\sum l)^2$ 降到 $\sum l^2$。
4. **fused linear CE**：看 `torch.cuda.max_memory_allocated()`，峰值应降约 $10sV$ 字节（8B、$s = 8192$ 时约 10 GB）；loss 曲线和改之前重合。
5. **融合 kernel / torch.compile**：profiler 里每步 kernel 数下降，非 GEMM kernel 的总时间下降；tokens/s 上升。
6. **调 micro-batch 的 token 数**：先确认显存余量（峰值离上限多少 GB），再看 tokens/s 是否还涨；不涨就停，见第 5 节。
7. **recompute 策略**：按第 5 节逐层关 checkpoint，看 MFU（只算 $6N$）上升；HFU（把重算也算进去）和 MFU 的差距就是重算的浪费。
8. **并行配置**：profiler 时间线上 NCCL kernel 有多少没和计算重叠（exposed communication）。暴露多就换并行方式或调预取，见 [计算通信 overlap](/parallel/comm-overlap)。

## 面试追问

::: details Q：8B 全参 SFT，8 张 H100，理论上每秒多少 token？
$n_{\text{gpu}} \cdot \text{peak} \cdot u / 6N = 8 \times 989\times10^{12} \times 0.4 / (6 \times 8.03\times10^9) \approx 6.6$ 万 token/s。$6N$ 是 forward $2N$ 加 backward $4N$（dgrad、wgrad 各 $2N$）。开 full recompute 是 $8N$，降到约 4.9 万。torchtitan 实测 8B 在 8 张 H100 上每卡 6258 token/s（FSDP + selective AC，作者自测），即 8 卡约 5 万。
:::

::: details Q：为什么开了 gradient checkpointing 以后，lm_head 的 logits 反而成了显存大头？
full recompute 后每层只存输入 $2sh$ 字节，8B、$s = 8192$ 时 32 层合计 2.1 GB。logits 是 $s \times V$，Llama 3 的 $V = 128256$ 是 $h = 4096$ 的 31 倍，一份 fp32 就 4.2 GB，HF 的 loss 路径同时有 bf16 logits、fp32 副本、log_softmax 输出三份，约 $10sV = 10.5$ GB。词表越大越严重。
:::

::: details Q：fused linear cross-entropy 为什么能省显存而不增加计算？
沿 token 维分块：每块算 logits、logsumexp、loss，并当场算出 $\text{softmax} - \text{onehot}$ 这个梯度，乘出 $\partial x$ 和 $\partial W$ 后把这块 logits 丢掉。任何时刻只有一块 logits（Liger 让一块和 hidden states 一样大）。梯度在 forward 里已经算完，backward 不用再做 lm_head 的矩阵乘，所以总 FLOPs 不变。
:::

::: details Q：省下 20 GB 显存，是加大 micro-batch 还是关 recompute？
先关 recompute。micro-batch 已有几千个 token 时 GEMM 的算术强度约等于 token 数，早超过 ridge point，加大不会让每 token 更快，只摊薄 FSDP 通信这类每 micro-batch 的固定开销。关 recompute 每层省掉一次重算，8B 每层约多存 1.07 GB、省 0.8% 计算，20 GB 能关约 18 层。全部关完还有余量再加 micro-batch。
:::

::: details Q：单机 8 卡全参 8B，ZeRO-2 还是 FSDP？
看不开 recompute 能不能放下。ZeRO-2 每卡模型状态 30.1 GB，加 $s = 8192$ 的激活 36.5 GB 和未融合的 logits 约 10 GB，已经接近 80 GB；FSDP2 每卡 16.1 GB，加 fused CE 约 53 GB，不开 recompute 也有余量。FSDP2 多 50% 通信，单机 NVLink 上每个 micro-batch 约 93 ms，可以被预取盖住；recompute 却是 33% 的计算，所以选 FSDP2。
:::

::: details Q：LoRA 为什么不能让训练快 3 倍？
LoRA 只省掉冻结权重的 wgrad（$2N$），forward（$2N$）和 dgrad（$2N$）一点没少，因为梯度必须穿过每一层传回下面的 adapter。理论从 $6N$ 降到 $4N$，最多 1.5 倍；LoRA 论文在 GPT-3 上实测 +25%（作者自测）。它省的主要是模型状态显存（约 8 倍）。
:::

## 手撕

两段：(1) 算 packing 前后的有效 token 比例；(2) 分块 cross-entropy，按 token 维切块、每块算 logits → loss → 梯度 → 丢弃，并验证 loss 和梯度与完整版一致。

```python
import torch
import torch.nn.functional as F

# ---- 1. 有效 token 比例 ----
def eff_padded(lengths, mbs):
    """按顺序每 mbs 条一组，每组 pad 到组内最长"""
    real = sum(lengths)
    padded = sum(max(lengths[i:i + mbs]) * len(lengths[i:i + mbs])
                 for i in range(0, len(lengths), mbs))
    return real / padded

def eff_packed(lengths, cap):
    """first-fit decreasing 装箱，每箱 cap 个 token"""
    bins = []
    for L in sorted(lengths, reverse=True):
        for b in bins:
            if b[0] >= L:
                b[0] -= L
                break
        else:
            bins.append([cap - L])
    return sum(lengths) / (len(bins) * cap)

torch.manual_seed(0)
lengths = torch.randint(100, 2048, (1000,)).tolist()     # 1000 条样本，长度 100–2047
print(f"padded  {eff_padded(lengths, mbs=8):.3f}")       # padded  0.587
print(f"packed  {eff_packed(lengths, cap=8192):.3f}")    # packed  0.993

# ---- 2. 分块 cross-entropy：不物化 [T, V] 的 logits ----
class ChunkedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, y, chunk):
        # x: [T, h] 最后一层 hidden；W: [V, h] lm_head；y: [T] 标签，-100 不算 loss
        n = (y != -100).sum().clamp(min=1)                   # 分母：有效 token 数
        loss = torch.zeros((), dtype=torch.float32)
        gx = torch.empty_like(x)
        gW = torch.zeros_like(W, dtype=torch.float32)
        for s in range(0, x.shape[0], chunk):
            xc, yc = x[s:s + chunk], y[s:s + chunk]
            logits = (xc @ W.T).float()                      # [c, V]，只活在这一块里
            lse = torch.logsumexp(logits, dim=-1)            # [c]
            ok = yc != -100
            tgt = yc.clamp(min=0)
            r = torch.arange(len(yc))
            loss += (lse - logits[r, tgt])[ok].sum()
            # d(loss_sum)/d(logits) = softmax - onehot，原地写回 logits 省一份
            g = logits.sub_(lse[:, None]).exp_()
            g[r, tgt] -= 1
            g[~ok] = 0
            g /= n
            gx[s:s + chunk] = (g.to(x.dtype) @ W)            # [c, h]
            gW += g.T @ xc.float()                           # [V, h]
        ctx.save_for_backward(gx, gW.to(W.dtype))
        return loss / n

    @staticmethod
    def backward(ctx, gout):
        gx, gW = ctx.saved_tensors
        return gx * gout, gW * gout, None, None

T, h, V = 512, 64, 1000
x = torch.randn(T, h, requires_grad=True)
W = (0.1 * torch.randn(V, h)).requires_grad_()
y = torch.randint(0, V, (T,))
y[:100] = -100                                           # 前 100 个是 prompt，不算 loss

ref = F.cross_entropy(x @ W.T, y, ignore_index=-100)
gx_ref, gW_ref = torch.autograd.grad(ref, (x, W))
out = ChunkedLinearCE.apply(x, W, y, 128)
gx, gW = torch.autograd.grad(out, (x, W))
print(f"{ref.item():.6f} {out.item():.6f}")              # 7.186985 7.186985
print(torch.allclose(gx, gx_ref, atol=1e-5),
      torch.allclose(gW, gW_ref, atol=1e-4))             # True True
```

要点：分母 $n$ 必须是**全部块**的有效 token 数，不能每块各自平均（和 [SFT](/posttrain/sft) 第 3 节 gradient accumulation 的坑是同一件事）。常见变体：问为什么梯度可以在 forward 里算（loss 是标量、是最后一步，上游梯度 `gout` 只是一个缩放）；问块大小怎么选（让一块 logits 和 hidden states 一样大）。

## 参考

- [Liger Kernel: Efficient Triton Kernels for LLM Training](https://arxiv.org/abs/2410.10989)：FusedLinearCrossEntropy、RMSNorm / RoPE / SwiGLU kernel
- [Cut Your Losses in Large-Vocabulary Language Models (Cut Cross-Entropy)](https://arxiv.org/abs/2411.09009)
- [TorchTitan](https://arxiv.org/abs/2410.06511)：Llama 3.1 8B 在 8 张 H100 上的吞吐与 MFU
- [LoRA](https://arxiv.org/abs/2106.09685)：GPT-3 上的训练吞吐
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)：激活公式 $34sbh$
- [HF Transformers loss_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py)：`ForCausalLMLoss` 的 fp32 upcast
