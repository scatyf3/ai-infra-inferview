---
title: FP8 训练：哪些算子用 FP8、scale 怎么定、DeepSeek-V3 改了什么
status: draft
tags: [fp8, mixed-precision, quantization, transformer-engine, deepseek-v3, mxfp8]
difficulty: 3
order: 5
related: [/posttrain/training-memory, /inference/quantization, /gpu/tensor-core-gemm, /parallel/zero-fsdp, /parallel/moe-ep]
stack: []
train: [tr-m-precision, tr-k-gemm]
---

# FP8 训练：哪些算子用 FP8、scale 怎么定、DeepSeek-V3 改了什么

> 只有 Linear 层的三个 GEMM 换成 FP8，其余照旧 bf16 / fp32；难点全在 scale 的粒度和累加精度

## 一句话结论

FP8 训练把每个 Linear 层的三个 GEMM（Fprop、Dgrad、Wgrad）的输入 cast 成 fp8，用 Tensor Core 的 fp8 峰值（H100 dense 1979 TFLOP/s，bf16 的 2 倍）算，其余算子、主权重和优化器状态保持高精度。每个 fp8 张量配一个 fp32 的 scale $= 448 / \text{amax}$，把数值挪进 e4m3 的范围。scale 粒度从粗到细：per-tensor（Transformer Engine 的 delayed / current scaling）→ DeepSeek-V3 的激活 $1\times128$、权重 $128\times128$ → Blackwell 硬件原生的 MXFP8（每 32 个元素一个 2 的幂 scale）。端到端加速到不了 2×，公开的 Llama3-70B 实测约 1.5×（作者自测）。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $X$ | Linear 层输入（激活），$[M, K]$ | $M = 4096$ 个 token，$K = 7168$ |
| $W$ | 权重，$[K, N]$ | $N = 2048$ |
| $Y$ | 输出 $Y = XW$，$[M, N]$ | |
| $dY, dX, dW$ | loss 对 $Y, X, W$ 的梯度，形状同各自张量 | |
| $\text{amax}$ | 一组元素的 $\max\lvert x\rvert$ | 3.5 |
| $s$ | scale，fp32，量化时乘、反量化时除 | $448 / 3.5 = 128$ |
| $q$ | fp8 编码后的值 $q = \text{fp8}(x \cdot s)$ | |
| $q_{\max}$ | fp8 格式能表示的最大值 | e4m3 为 448 |
| $N_C$ | DeepSeek-V3 提升到 fp32 累加的间隔（沿 $K$ 的元素数） | 128 |
| $\Psi$ | 参数量 | $7\times10^9$ |

fp8 两种格式（e4m3 最大 448、e5m2 最大 57344）的位数、范围、为什么要两种，见 [训练显存账](/posttrain/training-memory#fp8-为什么要-e4m3-和-e5m2-两种)，这里不重复。GB 指 $10^9$ 字节。

### 1. 一个 Linear 层的三个 GEMM

| GEMM | 公式 | 沿哪一维求和 | 输入格式（TE 默认 `HYBRID`） |
|---|---|---|---|
| Fprop | $Y = XW$ | $K$ | $X$ e4m3，$W$ e4m3 |
| Dgrad | $dX = dY \cdot W^\top$ | $N$ | $dY$ e5m2，$W$ e4m3 |
| Wgrad | $dW = X^\top \cdot dY$ | $M$（token） | $X$ e4m3，$dY$ e5m2 |

1. 每个 GEMM 都是 $2MKN$ FLOPs，三个合计 $6MKN$。每个参数（$KN$ 个）每个 token（$M$ 个）6 FLOPs，这就是训练 $6\Psi$ FLOPs/token 的来源：forward 一个，backward 对输入和对权重各一个。
2. 输出 $Y, dX, dW$ 都在 fp32 里累加，写出成 bf16（$dW$ 交给 fp32 的梯度累加）。fp8 只出现在 GEMM 的**输入**上。
3. Transformer Engine（TE）的 `Format.HYBRID`：forward 的张量用 e4m3，backward 的梯度用 e5m2（[TE recipe 源码](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe/__init__.py)）。

**算一笔收益**：H100 SXM 的 dense 峰值 bf16 989 TFLOP/s、fp8 1979 TFLOP/s（[NVIDIA 规格](https://www.nvidia.com/en-us/data-center/h100/)，datasheet 标的 1979 / 3958 是带 2:4 稀疏的数，dense 减半）。GEMM 快 2 倍，端到端按 Amdahl：设 bf16 下 GEMM 占一步时间的比例 $f$，

$$
\text{加速比} = \frac{1}{(1 - f) + f/2}
$$

$f = 0.7$（假设）时是 $1/(0.3 + 0.35) = 1.54$。到不了 2× 的原因：

1. 非 GEMM 部分不变：attention 核心、norm、激活函数、优化器、通信。
2. 量化本身有开销：每个 fp8 输入要求 amax、乘 scale、cast，backward 还要转置版本，这些都是 memory-bound 的小 kernel。
3. 小矩阵的 GEMM 本来就到不了峰值。

实测：PyTorch 团队在 128 张 H100 上预训练 Llama3-70B，float8 对 bf16 快 1.50×，其中约 20% 的收益来自 FSDP 的 fp8 all-gather，其余来自 fp8 计算（[dev-discuss](https://dev-discuss.pytorch.org/t/enabling-float8-all-gather-in-fsdp2/2359)，作者自测）。

### 2. 什么留在高精度

| 部分 | 精度 | 为什么 |
|---|---|---|
| embedding | bf16 | 是查表不是 GEMM，fp8 没有加速可拿 |
| LM head（输出投影） | bf16 | 后面直接接 softmax 和 loss，误差直接进 loss；DeepSeek-V3 也保留 |
| norm、softmax、激活函数 | bf16 / fp32 | 不是 GEMM；含求和、exp、除法，对精度敏感 |
| attention 核心（$QK^\top$、$PV$） | 多数实现 bf16 | FlashAttention 里和 softmax 融在一起；DeepSeek-V3 保留高精度，TE 的 block scaling 也不支持 fp8 attention |
| MoE gating | bf16 / fp32 | 路由是离散决策，小误差会改变选中的专家 |
| 主权重、Adam $m$ / $v$ | fp32 | 更新量太小，低精度加不上去，见 [训练显存账](/posttrain/training-memory#为什么要留一份-fp32-主权重) |
| 梯度累加（跨 micro-batch） | fp32 | 很多个小数连加，和主权重是同一个理由 |

DeepSeek-V3 列出的保留原精度的部分是 embedding、output head、MoE gating、normalization、attention 算子，以及主权重、权重梯度、优化器状态（[arXiv 2412.19437](https://arxiv.org/abs/2412.19437) §3.3.1）。它把 AdamW 的 $m$、$v$ 降到了 bf16，主权重和用于累加的梯度仍是 fp32（§3.3.3）。

### 3. Scale 怎么定

**定义**：一组元素共用一个 scale，

$$
s = \frac{q_{\max}}{\text{amax}}, \qquad q = \text{fp8}(x \cdot s), \qquad \hat{x} = q / s
$$

e4m3 的 $q_{\max} = 448$（fp8 格式的最大值）。乘 $s$ 让这组里最大的数正好落在 448，整组数都在可表示范围内。

**算一遍**：一组数的 $\text{amax} = 3.5$，$s = 448 / 3.5 = 128$。

1. $x = 3.5$：$3.5 \times 128 = 448$，正好是 e4m3 的最大值，无误差。
2. $x = 0.01$：$0.01 \times 128 = 1.28$。e4m3 在 $[1, 2)$ 里的间隔是 $2^{-3} = 0.125$，最近的格点是 1.25，反量化 $1.25 / 128 = 0.00977$，相对误差 2.3%。
3. e4m3 正规数的最大相对舍入误差是半个间隔 $2^{-4} = 6.25\%$，和 scale 无关；scale 只决定数落不落在正规范围里。

**什么时候 per-tensor 不够**：e4m3 正规数的范围是 $[2^{-6}, 448]$，跨度 $448 / 2^{-6} \approx 2.9\times10^4$。一组里最大值和普通值之比超过这个数，普通值会掉进非正规数区（间隔固定 $2^{-9}$，相对误差变大）甚至变成 0。手撕一节的实验：前 8 个 token 放大 $10^3$ 倍，per-tensor 和 $1\times128$ 误差一样；放大 $10^5$ 倍，per-tensor 的误差从 2.6% 涨到 50%。

**per-tensor 的两种 amax 取法**（TE 都支持）：

| 做法 | amax 从哪来 | 代价 |
|---|---|---|
| delayed scaling | 前若干步记录的 amax 历史取最大值 | 不用额外读一遍张量；但 scale 落后于当前数据 |
| current scaling | 当前张量现算 | cast 前要先扫一遍求 amax，多一次读 |

TE 的 `DelayedScaling` 当前默认 `amax_history_len=1024`、`amax_compute_algo="max"`、`margin=0`，$s = (q_{\max}/\text{amax}) / 2^{\text{margin}}$（[TE recipe 源码](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe/__init__.py)）。

**delayed 的风险**：

1. **突然出现的大值**：历史 amax 是 3.5，$s = 128$；这一步出现 10，$10 \times 128 = 1280 > 448$，只能截到 448，反量化成 3.5，误差 65%。e4m3 没有 inf，不先 clamp 就 cast 会直接变成 NaN（PyTorch 的 `float8_e4m3fn` 就是这样）。
2. **数值变小后反应慢**：历史窗口取最大值，amax 下降后要等旧的大值滑出 1024 步窗口，scale 才跟上；期间整体数值偏小，精度没用满。

### 4. DeepSeek-V3 改了什么

出处都是 [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) §3.3 和附录 B。

**1. 细粒度 scale**：

| 张量 | 一组 | 一个 scale 管多少元素 |
|---|---|---|
| 激活 $X$（Fprop 的输入） | $1\times128$：每个 token、每 128 个通道 | 128 |
| 权重 $W$ | $128\times128$ 的块 | 16384 |

1. amax 每步在线算（current），不用历史。
2. 激活按 token 分组，一个离群 token 只影响它自己那几组。附录 B.2 是反例：把激活梯度也改成 $128\times128$ 块量化，约 16B 的 MoE 模型训练约 300B token 后发散；作者归因于激活梯度在 token 间极不均衡，离群值跟着 token 走，跨 token 的方块管不住（作者自测）。
3. 权重用方块：$W$ 和 $W^\top$ 的分块完全一样，Dgrad 用 $W^\top$ 时不用重新量化。
4. 激活在 backward 要沿 token 维求和（Wgrad），论文 §3.5.2 写明 forward 存的 $1\times128$ tile 要读出、反量化、转置、再量化成 $128\times1$。
5. 算一笔 scale 的个数：$X$ 是 $[4096, 7168]$，$4096 \times 7168/128 = 229{,}376$ 个 fp32 scale，每元素摊 $32/128 = 0.25$ bit；$W$ 是 $[7168, 2048]$，$56 \times 16 = 896$ 个。

**2. 累加精度**：

1. 论文测得 H800 上 FP8 GEMM 的 Tensor Core 累加器只保留约 14 bit，远低于 fp32；$K = 4096$ 的随机矩阵乘，最大相对误差接近 2%（作者自测）。论文指出这个有限精度的累加仍是一些 fp8 框架的默认选项。
2. 做法：沿 $K$ 每 $N_C = 128$ 个元素（4 个 WGMMA）就把 Tensor Core 的部分和拷到 CUDA core 的 fp32 寄存器里累加。per-block 的 scale 正好也在这一步乘上去（反量化），几乎不额外花钱。
3. 为了不让 Tensor Core 闲着：一个 warpgroup 做提升时，另一个 warpgroup 继续做 MMA。
4. 算一笔：$K = 7168$ 时每个输出元素提升 $7168/128 = 56$ 次。为什么 fp32 累加器重要见 [Tensor Core GEMM](/gpu/tensor-core-gemm)。

**3. 全部用 e4m3**：Fprop、Dgrad、Wgrad 的所有输入都用 e4m3，不用 HYBRID 的 e5m2。论文把可行性归功于细粒度 scale：组越小，组内的数值跨度越小，e4m3 的范围就够了，于是可以选精度更高的格式。

**4. 存什么、传什么**：

1. Linear 层为 backward 缓存的激活存 fp8（Wgrad 直接用）。
2. attention 后那个 Linear 的输入还要给 attention 的 backward 用，对精度更敏感，用自定义的 E5M6 格式，scale 取 2 的整数次幂。
3. MoE 里 SwiGLU 的输入存 fp8，输出在 backward 时重算。
4. 通信：MoE up-projection 之前的激活 dispatch 用 fp8，combine 保留 bf16（通信量算法见 [MoE / EP](/parallel/moe-ep)）。

**5. 效果**：在约 16B（1.33T token）和约 230B（约 0.9T token）两个规模的 MoE 上，和 bf16 基线相比 loss 的相对误差保持在 0.25% 以下（附录 B.1，作者自测）。

### 5. MXFP8：硬件做第 4 节的事

OCP 的 MX 规范（[OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)）：

1. 每 32 个连续元素一块，共享一个 scale。
2. scale 是 E8M0：8 位指数、0 位尾数，只能是 2 的幂。乘 scale 等于改指数，没有舍入误差。
3. 元素是 fp8（e4m3 / e5m2），同一规范还定义了 fp6、fp4、int8。
4. Blackwell 的 Tensor Core 原生支持：块 scale 在 Tensor Core 内部乘，不用像 DeepSeek-V3 那样回 CUDA core。

**和第 4 节的对应**：

| | DeepSeek-V3（Hopper） | MXFP8（Blackwell） |
|---|---|---|
| 块大小 | 激活 $1\times128$，权重 $128\times128$ | 沿归约维 32 个元素 |
| scale 格式 | fp32 | E8M0（2 的幂） |
| 乘 scale 在哪 | CUDA core，每 $N_C = 128$ 一次 | Tensor Core 内部 |
| 格式 | 全部 e4m3 | 推荐全部 e4m3 |
| 每元素 scale 开销 | $32/128 = 0.25$ bit（激活） | $8/32 = 0.25$ bit |

**scale 怎么取 2 的幂**：NVIDIA 的 MXFP8 预训练配方（[arXiv 2506.08027](https://arxiv.org/abs/2506.08027)）把指数**向上取整**：$X = 2^{\lceil \log_2(\text{amax}/448) \rceil}$，$q = \text{fp8}(x / X)$。例：amax = 5，$5/448 = 0.0112$，$\log_2 = -6.49$，向上取 $-6$，$X = 1/64$，$5 \times 64 = 320 \le 448$。向下取 $-7$ 的话 $5 \times 128 = 640$ 溢出。论文认为 OCP v1 的取法相当于向下取，会溢出，这是 MXFP8 训练出问题的原因之一。

同一篇论文的其他结论（作者自测）：所有张量（含激活梯度）都用 e4m3；8B 模型上梯度用 e5m2 困惑度明显变差；最大 8B、15T token 的训练和 bf16 对齐。代价：MX 的块沿归约维划分，backward 要用转置的张量，所以行、列两个方向各量化一份。

### 6. 显存账：省了什么，没省什么

**没省的**：模型状态。bf16 参数 2 + 梯度 2 + fp32 主权重 4 + Adam 8 = 16 字节/参数（[训练显存账](/posttrain/training-memory#模型状态-每参数-16-字节是怎么来的)），fp8 一个字节都没动。7B 还是 112 GB，70B 还是 1120 GB。反而多出 fp8 权重的缓存（行、列方向各一份时每参数 2 字节）。

**省的**：

1. **激活缓存**：每层激活（FlashAttention 下）$34sbh$ 字节里，Linear 层的输入是 QKV 输入 $2sbh$、输出投影输入 $2sbh$、MLP 两个输入 $2sbh + 8sbh$，共 $14sbh$。存成 fp8 减半，每层变 $27sbh$，省约 21%。
2. **通信**：FSDP 每次 all-gather 权重，bf16 是 $2\Psi$ 字节，fp8 是 $\Psi$；MoE 的 dispatch 也减半。

算一笔（$s = 4096$，$b = 1$，不开 TP，用 Korthikanti 公式的系数）：

| | 7B（$L=32, h=4096$） | 70B（$L=80, h=8192$） |
|---|---|---|
| 模型状态 | 112 GB（不变） | 1120 GB（不变） |
| 激活，bf16（$34sbh \cdot L$） | 18.3 GB | 91.3 GB |
| 激活，Linear 输入 fp8（$27sbh \cdot L$） | 14.5 GB | 72.5 GB |
| FSDP 一次 all-gather 的权重 | 14 → 7 GB | 140 → 70 GB |

结论：fp8 主要买的是**计算速度**，显存只在激活和通信上小幅受益；模型状态还得靠 [ZeRO / FSDP](/parallel/zero-fsdp) 切。

### 7. 排障：fp8 的 loss 和 bf16 对不齐

按顺序查：

1. **先量 bf16 自己的噪声**：换个 seed 跑两次 bf16，loss 差多少。fp8 的差距在这个带内就不是问题。参考量级：DeepSeek-V3 的相对误差 < 0.25%（作者自测）。
2. **看 scale 统计**：每个 fp8 张量记录 amax、scale、饱和比例（$\lvert x \cdot s\rvert$ 超过 448 的比例）、下溢比例（变成 0 的比例）。delayed scaling 下先换成 current scaling，问题消失就是 amax 历史跟不上。
3. **二分定位**：先把首尾几层、再把某一类 GEMM（Wgrad 最常见）退回 bf16，看差距在哪一步消失。
4. **查累加**：确认 fp8 GEMM 有提升到 fp32 累加（第 4 节），$K$ 很大的 GEMM 最容易出问题。
5. **训练后期才出现的发散**：Fishman et al.（[arXiv 2409.12517](https://arxiv.org/abs/2409.12517)）在 2T token 的 fp8 训练中发现，SwiGLU 会在长时间训练后放大离群值，造成短训练看不出的不稳定；他们的 Smooth-SwiGLU 修复后 7B 模型和 bf16 持平（作者自测）。

## 面试追问

::: details Q：为什么梯度用 e5m2、权重和激活用 e4m3？DeepSeek-V3 为什么全用 e4m3？
per-tensor 一个 scale 管整个梯度张量，梯度的数值跨度大，需要 e5m2 的范围。DeepSeek-V3 每 128 个元素一个 scale，组内跨度小，e4m3 的范围够用，于是选尾数多一位的 e4m3 换精度。MXFP8 每 32 个元素一个 scale，同样推荐全用 e4m3。
:::

::: details Q：delayed scaling 和 current scaling 怎么选？
delayed 省一次读张量求 amax，但 scale 来自过去的步数：突然的大值会饱和，数值变小时 scale 跟得慢。current 每次现算，多一次 memory-bound 的扫描，但没有这两个问题。块级 scale（DeepSeek-V3、MXFP8）都是现算。
:::

::: details Q：既然 fp8 GEMM 是 bf16 的 2 倍，为什么端到端只有 1.3–1.5×？
GEMM 只占一部分时间，attention 核心、norm、通信、优化器不变；cast 和求 amax 是额外的 memory-bound kernel；小 GEMM 本来就到不了峰值。GEMM 占 70% 时 Amdahl 上限是 1.54×。
:::

::: details Q：Hopper 上 fp8 GEMM 的累加有什么问题？DeepSeek-V3 怎么解决？
DeepSeek-V3 测得 H800 的 fp8 Tensor Core 累加器只保留约 14 bit，$K = 4096$ 时最大相对误差接近 2%（作者自测）。解决：沿 $K$ 每 128 个元素把部分和拷到 CUDA core 上用 fp32 累加，顺便乘上这一块的 scale；两个 warpgroup 交替做 MMA 和提升，把开销藏起来。
:::

::: details Q：fp8 训练能省多少显存？
模型状态（16 字节/参数）不省；Linear 层缓存的输入激活减半，每层约省 21%；FSDP 的权重 all-gather 和 MoE dispatch 的通信量减半。fp8 主要是为了算得快。
:::

::: details Q：为什么激活用 $1\times128$、权重用 $128\times128$？
激活的离群值按 token 出现，按 token 分组能把离群 token 隔离开。权重用方块，$W$ 和 $W^\top$ 的分块一样，Dgrad 用 $W^\top$ 时不用重新量化。激活在 Wgrad 里要转置，DeepSeek-V3 是把 $1\times128$ 反量化、转置、再量化成 $128\times1$。
:::

## 手撕

实现 per-tensor 和 per-block（$1\times128$）的量化 / 反量化，对比误差：

```python
import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max                      # 448.0

def quant_per_tensor(x):
    amax = x.abs().max().clamp(min=1e-12)
    scale = FP8_MAX / amax                          # 一个 fp32 标量
    q = (x * scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)  # 先 clamp：越界的直接 cast 会变 NaN
    return q, scale

def dequant_per_tensor(q, scale):
    return q.to(torch.float32) / scale

def quant_1x128(x, block=128):
    M, K = x.shape
    xb = x.view(M, K // block, block)               # 每行每 128 个元素一组
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = FP8_MAX / amax                          # [M, K/128, 1]
    q = (xb * scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q.view(M, K), scale.squeeze(-1)

def dequant_1x128(q, scale, block=128):
    M, K = q.shape
    qb = q.to(torch.float32).view(M, K // block, block)
    return (qb / scale.unsqueeze(-1)).view(M, K)

def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()

torch.manual_seed(0)
M, K = 4096, 4096
for outlier in [1, 1000, 10000, 100000]:
    x = torch.randn(M, K)
    x[:8] *= outlier                                # 前 8 个 token 整行放大
    q, s = quant_per_tensor(x)
    q2, s2 = quant_1x128(x)
    e1 = rel_err(dequant_per_tensor(q, s)[8:], x[8:])   # 只看正常的行
    e2 = rel_err(dequant_1x128(q2, s2)[8:], x[8:])
    print(f"outlier x{outlier:>6}: per-tensor {e1:.4f}  1x128 {e2:.4f}")

# outlier x     1: per-tensor 0.0265  1x128 0.0257
# outlier x  1000: per-tensor 0.0265  1x128 0.0257
# outlier x 10000: per-tensor 0.0538  1x128 0.0257
# outlier x100000: per-tensor 0.5000  1x128 0.0257
```

读结果：

1. 放大 $10^3$ 倍时两者一样：e4m3 是浮点，只要普通值还在正规数范围（$\ge 2^{-6}$）里，相对误差就是尾数决定的约 2.6%，和 scale 大小无关。这是 fp8 比 int8 更抗离群值的原因（int8 的对比见 [量化](/inference/quantization)）。
2. 放大 $10^4$ 倍以上，普通值乘 scale 后跌进非正规数区或变成 0，per-tensor 误差迅速变大；$1\times128$ 的组里没有离群 token，不受影响。

常见追问：把 `quant_1x128` 改成 $128\times128$ 的块；写一个 delayed scaling（维护长度为 $H$ 的 amax 历史，取最大值算下一步的 scale），构造一步突然变大的输入看饱和；把块 scale 改成 E8M0（$2^{\lceil\log_2(\text{amax}/448)\rceil}$）。

## 参考

- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)：e4m3 / e5m2 的定义
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：§3.3 FP8 训练框架、§3.5.2 硬件建议、附录 B
- [Transformer Engine recipe 源码](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe/__init__.py)：`DelayedScaling`、`Float8CurrentScaling`、`MXFP8BlockScaling`、`Float8BlockScaling`
- [OCP Microscaling Formats (MX) v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027)
- [Scaling FP8 training to trillion-token LLMs](https://arxiv.org/abs/2409.12517)：SwiGLU 离群值放大
- [Enabling Float8 All-Gather in FSDP2](https://dev-discuss.pytorch.org/t/enabling-float8-all-gather-in-fsdp2/2359)：Llama3-70B 1.50×
- [NVIDIA H100 规格](https://www.nvidia.com/en-us/data-center/h100/)
