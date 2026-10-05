---
title: 量化：GPTQ / AWQ / SmoothQuant / FP8
status: draft
tags: [quantization, fp8]
difficulty: 3
order: 8
related: [/inference/memory-accounting, /inference/prefill-decode-roofline]
stack: [ld-format, ld-dtype, k-gemm]
---

# 量化

1. 不同的格式
2. 不同的粒度
3. 各种消除outlier的算法

三件事都围绕同一个式子。对称量化把实数 $x$ 映到整数 $x_q$：

$$
x_q = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right), -q_{\max}, q_{\max}\right), \quad \hat{x} = s \cdot x_q, \quad s = \frac{\max|x|}{q_{\max}}
$$

$s$ 是 scale，$q_{\max}$ 是格式能表示的最大值（int8 取 127，int4 取 7），$\hat{x}$ 是反量化回来的近似值。格式决定 $q_{\max}$ 和格点怎么分布，粒度决定多少个元素共用一个 $s$，outlier 决定 $\max|x|$ 有多大。

tldr：把高精度的 $x$（如 bf16）压缩成低精度的 $x_q$（如 int4），再额外存一份 metadata（scale $s$），用 $\hat{x} = s \cdot x_q$ 重建。目标是 reconstruction error 最小：

$$
\min_{s} \ \| x - \hat{x} \|^2 = \min_{s} \ \| x - s \cdot x_q \|^2
$$

## 1. 格式

| 格式       | 位宽  | 符号/指数/尾数  | 最大值                  | 用途                |     |
| -------- | --- | --------- | -------------------- | ----------------- | --- |
| BF16     | 16  | 1 / 8 / 7 | $3.4 \times 10^{38}$ | 基线，范围同 FP32       |     |
| FP8 E4M3 | 8   | 1 / 4 / 3 | 448                  | 推理的权重、激活、KV       |     |
| FP8 E5M2 | 8   | 1 / 5 / 2 | 57344                | 训练的梯度（范围大、精度低）    |     |
| INT8     | 8   | 整数        | 127                  | W8A8              |     |
| INT4     | 4   | 整数        | 7                    | W4A16 的权重         |     |
| FP4 E2M1 | 4   | 1 / 2 / 1 | 6                    | MXFP4 / NVFP4 的元素 |     |

1. int和fp的分布区别
2. fp8训练vs推理
3. 不同的粒度，w8a8和w4a16

**INT 和 FP 的区别在格点分布。** INT 的格点等间距，绝对误差处处一样，适合均匀分布的数。FP 的格点在 0 附近密、远处疏，相对误差大致恒定，适合「大部分很小、少数很大」的长尾分布，LLM 的激活正是这样。E2M1 能表示的正数只有 8 个：0, 0.5, 1, 1.5, 2, 3, 4, 6，可以看出越往大越稀。

**E4M3 和 E5M2 是拿 1 bit 在精度和范围之间换。** E4M3 多 1 位尾数，1 后面的下一个格点是 1.125（E5M2 是 1.25）；E5M2 多 1 位指数，最大值与最小正规数之比从约 $2^{15}$（$448 / 2^{-6}$）扩到约 $2^{30}$（$57344 / 2^{-14}$）。权重和激活除以 scale 后落在窄范围里，缺的是精度，所以用 E4M3；梯度在一个张量里就跨很多个数量级，缺的是范围，所以用 E5M2。

**FP4 必须配块 scale。** E2M1 的最大值是 6，所以 $s = \max|x| / 6$，能表示的最小非零值是 $0.5 s$。如果整个张量共用一个 scale，里面有一个 60，$s = 10$，最小非零值就是 5，所有 $|x| < 2.5$ 的数都会变成 0，而张量里绝大多数数都在 1 左右。解法是把张量切成很小的块，每块单独算 $s$：一块里最大是 1.2，$s = 0.2$，最小非零值 0.1，块里的数就都保得住；那个 60 只影响它自己那一块。有两种标准：

1. **MXFP4**：每 32 个元素一块，块 scale 用 E8M0（8 bit 全是指数），只能取 2 的幂。上例的 0.2 只能向上取到 0.25，块内最大值变成 $1.2 / 0.25 = 4.8$，用不满到 6 的格点。
2. **NVFP4**：每 16 个元素一块，块 scale 用 E4M3，可以取 0.203 这种非 2 的幂的值，更贴合。但 E4M3 最大只有 448，表示不了太大或太小的 scale，所以整个张量再配一个 FP32 的 scale 先把数值缩放到合适范围：$\hat{x} = s_{\text{tensor}} \cdot s_{\text{block}} \cdot x_q$。

块 scale 的开销：MXFP4 是 $4 + 8/32 = 4.25$ bit/元素，NVFP4 是 $4 + 8/16 = 4.5$ bit/元素。

**记号 W$x$A$y$**：权重 $x$ bit，激活 $y$ bit。

- **W4A16**：权重 int4 存储，算之前反量化成 bf16，再走 bf16 Tensor Core。只省读权重的带宽，计算量不变。decode 是 memory-bound，正好受益，见 [Roofline](/inference/prefill-decode-roofline)。
- **W8A8（INT8 / FP8）**：权重和激活都是 8 bit，直接走 8 bit Tensor Core，吞吐是 bf16 的 2 倍。prefill 是 compute-bound，只有这类方案能加速。H100 原生支持 FP8，所以 FP8 W8A8 是现在的默认。
- **W4A4（FP4）**：Blackwell 原生支持 FP4 Tensor Core。

### 量化哪里、怎么用

**量化哪里。** 只量化 Linear 层：attention 的 QKV 和 O 投影、FFN 的 gate / up / down、MoE 的专家。它们占了几乎全部的权重和 FLOPs。具体是三类张量：

1. Linear 的权重 $W$
2. 进 Linear 的激活 $X$（只有 W8A8、W4A4 才量化）
3. KV cache

一般保持 bf16 的：embedding、lm_head、RMSNorm、softmax、RoPE、MoE router。它们要么是逐元素运算、量化了也省不了多少，要么直接决定输出（logits、选哪个专家），对误差敏感。

不同的部分需要quantize和dequantize，这里也需要高效的实现


**怎么用 scale。** 记 Linear 为 $Y = XW$，$X \in \mathbb{R}^{M \times K}$，$W \in \mathbb{R}^{K \times N}$，$s_w$、$s_x$ 分别是权重和激活的 scale。

W4A16 在 kernel 里先反量化、再乘：

$$
Y = X \cdot \hat{W}, \quad \hat{W} = s_w \cdot W_q
$$

kernel 从 HBM 读 int4 的 $W_q$ 和 $s_w$，在片上（寄存器 / shared memory）乘出 bf16 的 $\hat{W}$，再和 bf16 的 $X$ 做 GEMM。$\hat{W}$ 不写回 HBM，所以 HBM 上只读了 1/4 的权重字节。

W8A8 先做整数乘加、最后乘 scale：

$$
Y_{ij} \approx s_x[i] \cdot s_w[j] \cdot \sum_k X_{q,ik} W_{q,kj}
$$

1. 离线：$W$ 量化成 $W_q$ 和 per-channel 的 $s_w[j]$，存进 checkpoint。
2. 在线：$X$ 量化成 $X_q$，per-token 的 $s_x[i]$ 在每次 forward 时现算（dynamic）；这一步融进前一个 kernel（如 RMSNorm），不多读写一遍 $X$。也可以用校准集预先定一个 per-tensor 的 $s_x$（static），省掉求 max。
3. GEMM：$\sum_k X_{q,ik} W_{q,kj}$ 在 Tensor Core 里累加，INT8 用 int32，FP8 用 fp32。
4. epilogue：累加结果乘 $s_x[i] \cdot s_w[j]$，转成 bf16 写回 HBM。

KV cache 存的时候量化、读的时候反量化：

1. 写入：新 token 的 $K$、$V$ 算出来后量化成 FP8，连同 scale 存进 cache。
2. 读出：attention kernel 从 HBM 读 FP8 的 $K_q$、$V_q$，在片上乘 scale 还原，再算 $QK^\top$ 和 $PV$。

省的是 KV 的显存和 decode 时读 KV 的带宽，都是一半，见 [显存账](/inference/memory-accounting)。

## 2. 粒度

粒度就是「多少个元素共用一个 $s$」。越细，每组的 $\max|x|$ 越贴近组内真实数值，误差越小，但 scale 越多。

记 GEMM 为 $Y = XW$，$X \in \mathbb{R}^{M \times K}$ 是激活（$M$ 个 token，$K$ 个输入通道），$W \in \mathbb{R}^{K \times N}$ 是权重（$N$ 个输出通道）。

| 粒度          | 共用 scale 的范围                       | 典型用法                     |
| ----------- | ---------------------------------- | ------------------------ |
| per-tensor  | 整个 $X$ 或 $W$                       | FP8 W8A8 的静态 scale       |
| per-token   | $X$ 的一行（一个 token）                  | W8A8 的激活                 |
| per-channel | $W$ 的一列（一个输出通道）                    | W8A8 的权重                 |
| per-group   | $W$ 一列里沿 $K$ 连续 $g$ 个元素，$g$ 常取 128 | W4A16 的权重                |
| per-block   | 一个小块，如 MX 的 32 个元素                 | FP4、DeepSeek-V3 的 FP8 训练 |

<QuantGranularity />


W8A8的quantize粒度非常直觉
1. a'c'ti'vv'a'ti
### 为什么激活不能 per-channel

整数 GEMM 要求 scale 能提到 $K$ 维求和外面。激活 per-token（scale $s_x[i]$）、权重 per-output-channel（scale $s_w[j]$）时：

$$
Y_{ij} = \sum_k X_{ik} W_{kj} \approx s_x[i] \cdot s_w[j] \cdot \sum_k X_{q,ik} W_{q,kj}
$$

求和全是整数乘加，最后乘一次 scale 就行。但如果激活按输入通道 $k$ 量化，scale 变成 $s_x[k]$，它在求和号里面，提不出来，整数 GEMM 就做不了。

问题是激活的 outlier 偏偏集中在固定的几个输入通道 $k$ 上，最该按 $k$ 分 scale，却不能这么做。第 3 节的 SmoothQuant 就是为了绕开这个限制。

per-group 的 scale 也在 $K$ 维上，W4A16 能用是因为它先把权重反量化成 bf16 再乘，scale 在反量化时就乘进去了，不进整数求和。

### scale 的开销

per-group $g = 128$，每组一个 16 bit 的 scale，摊到每个权重上是 $16 / 128 = 0.125$ bit。int4 权重的实际位宽就是 $4 + 0.125 = 4.125$ bit。int4 权重通常还带一个 zero point（非对称量化，$\hat{x} = s \cdot (x_q - z)$），再加一点。

## 3. 各种消除 outlier 的算法

### outlier 是什么，为什么致命

LLM 的激活里有少数几个**固定的通道**，在几乎所有 token 上的数值都比其他通道大 100 倍左右（LLM.int8() 发现模型超过 6.7B 后普遍出现）。权重本身比较平坦。

per-token int8 量化一个 token 的激活 $x = [0.1, -0.2, 0.3, 60]$：

$$
s = 60 / 127 = 0.472, \quad x / s = [0.21, -0.42, 0.64, 127] \to [0, 0, 1, 127]
$$

前两个值直接变成 0，第三个从 0.3 变成 0.472。一个 outlier 撑大了 $s$，其他值全部被舍入掉。

下面的算法按「怎么处理 outlier」分成四类。

### 拆出去：LLM.int8()

把含 outlier 的通道（$|x| > 6$）单独拿出来用 16 bit 浮点算，其余通道走 int8，结果相加。精度好，但拆分和两路 GEMM 让 kernel 很不规整，速度经常比不量化还慢。

### 挪到权重上：SmoothQuant

利用一个恒等变换：给 $X$ 的每个输入通道 $k$ 除以 $s_k$，同时给 $W$ 的第 $k$ 行乘以 $s_k$，乘积不变：

$$
Y = XW = \left(X \operatorname{diag}(s)^{-1}\right) \cdot \left(\operatorname{diag}(s) W\right)
$$

outlier 通道除以一个大的 $s_k$ 就被压平了，代价是 $W$ 的对应行被放大。$s_k$ 在两边之间折中：

$$
s_k = \frac{\max|X_{:,k}|^{\alpha}}{\max|W_{k,:}|^{1-\alpha}}, \quad \alpha = 0.5
$$

$\max|X_{:,k}|$ 是第 $k$ 个输入通道在校准数据上的最大绝对值，$\max|W_{k,:}|$ 是权重第 $k$ 行的最大绝对值。$\alpha$ 越大，越多难度挪给权重。

$\operatorname{diag}(s)^{-1}$ 不需要在线算：它离线乘进前一个 LayerNorm 的 weight 里。之后激活就能 per-token、权重 per-channel，走标准 W8A8 GEMM。

### 保护重要权重：AWQ

AWQ 做 weight-only（W4A16）。观察是：只有约 1% 的权重通道重要，而重要与否看的是**激活**的幅度，不是权重自己的大小，因为激活大的输入通道，对应权重行的量化误差会被放大。

做法和 SmoothQuant 是同一个恒等变换，只是目的不同：把重要通道的权重行乘以 $s_k > 1$ 再量化，相对舍入误差就变小了；激活侧除以 $s_k$，同样离线融进前一层。

$$
s_k = \overline{|X_{:,k}|}^{\,\alpha}
$$

$\overline{|X_{:,k}|}$ 是第 $k$ 个输入通道激活绝对值的均值，$\alpha \in [0, 1]$ 在校准集上网格搜索，取输出误差最小的那个。

### 补偿误差：GPTQ

GPTQ 也做 weight-only，思路不是改 outlier，而是让后量化的权重去补偿先量化的权重的误差。它按 $K$ 维一行一行地量化 $W$，每量化完一行，就用 Hessian $H = 2X^\top X$ 的逆把误差分摊到还没量化的行上，使 $\|XW - X\hat{W}\|^2$ 最小。

它和 outlier 的联系在 act-order 选项：按 $H$ 的对角线（即该输入通道激活的平方和）从大到小的顺序量化，先处理 outlier 通道，让它们的误差有更多的行来补偿。

### 转散：QuaRot / SpinQuant

前面的方法都在 outlier 通道上做文章，旋转直接把 outlier 抹匀。对任意正交矩阵 $R$（$RR^\top = I$）：

$$
Y = XW = (XR)(R^\top W)
$$

取 $R$ 为归一化的 Hadamard 矩阵，它把每个输出都变成所有输入的 $\pm$ 等权组合，一个通道里的大值被摊到所有通道。4 维的例子：

$$
R = \frac{1}{2}\begin{bmatrix} 1 & 1 & 1 & 1 \\ 1 & -1 & 1 & -1 \\ 1 & 1 & -1 & -1 \\ 1 & -1 & -1 & 1 \end{bmatrix}, \quad [1, -1, 1, 9] \, R = [5, -3, -5, 5]
$$

最大绝对值从 9 降到 5，平方和都是 84（正交变换保范数），数值变平，scale 变小。

$R^\top W$ 离线算进权重，$XR$ 大部分也能融进前一层的权重；少数位置（如 FFN 的 down_proj 输入、KV cache）需要在线乘 Hadamard，有快速算法，开销是 $O(K \log K)$。QuaRot 用固定的随机 Hadamard，SpinQuant 把 $R$ 当参数学出来。这类方法让 W4A4 和 4 bit KV cache 变得可用。

## 面试追问

::: details Q：为什么 W4A16 在大 batch 下反而可能比 bf16 慢？
W4A16 的收益来自少读权重。batch 大了以后 GEMM 变成 compute-bound，瓶颈不再是读权重；而 W4A16 的 kernel 要先反量化再乘，多了额外的计算，Tensor Core 利用率低于直接的 bf16 GEMM。大 batch 场景要换成 W8A8 / FP8，让计算本身变快。
:::

::: details Q：SmoothQuant 和 AWQ 用的是同一个变换，区别在哪？
变换都是 $X \operatorname{diag}(s)^{-1} \cdot \operatorname{diag}(s) W$，目标不同：

1. SmoothQuant 要量化激活（W8A8），用 $s$ 把激活的 outlier 压平。
2. AWQ 只量化权重（W4A16），激活保持 bf16，用 $s$ 放大重要的权重行来降低它们的相对误差。
:::

::: details Q：per-tensor FP8 为什么比 per-tensor INT8 更抗 outlier？
INT8 等间距，scale 被 outlier 撑大后，小值全部挤进 0 附近的几个格点。E4M3 的格点在 0 附近更密，同样的 scale 下小值仍有足够的相对精度。所以 FP8 常常 per-tensor 就够用，INT8 一般要 per-token × per-channel 再加 SmoothQuant。
:::

## 参考

- [LLM.int8()](https://arxiv.org/abs/2208.07339)、[SmoothQuant](https://arxiv.org/abs/2211.10438)、[AWQ](https://arxiv.org/abs/2306.00978)、[GPTQ](https://arxiv.org/abs/2210.17323)
- [QuaRot](https://arxiv.org/abs/2404.00456)、[SpinQuant](https://arxiv.org/abs/2405.16406)
- [OCP Microscaling (MX) Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [vLLM 文档：Quantization](https://docs.vllm.ai/en/latest/features/quantization/)
