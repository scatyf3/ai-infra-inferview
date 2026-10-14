---
title: FlashAttention backward：分块重算与梯度推导
status: draft
tags: [flash-attention, kernel, backward, recompute]
difficulty: 4
order: 5.5
related: [/inference/flash-attention, /handson/online-softmax, /handson/stable-softmax, /train/long-context, /train/fp8-training]
stack: []
train: [tr-k-attn]
---

# FlashAttention backward：分块重算与梯度推导

> 前向不存 $s \times s$ 的 attention 矩阵，反向怎么拿到梯度；多花多少 FLOPs，省下多少显存

前向的分块和 online softmax 见 [FlashAttention v1 / v2 / v3](/inference/flash-attention) 和 [Online Softmax](/handson/online-softmax)。这一页只讲反向。

## 记号

下面按单个 head 写，序列长 $s$，head 维度 $d$，$Q, K, V, O \in \mathbb{R}^{s \times d}$：

$$S = QK^\top/\sqrt d,\qquad P = \text{softmax}(S)\ (\text{按行}),\qquad O = PV$$

反向的输入是上游传下来的 $dO = \partial L / \partial O$，要求 $dQ, dK, dV$。

## 反向为什么需要 $P$

先看标准实现（autograd 直接展开）每一步要什么：

| 梯度 | 公式 | 需要 |
|---|---|---|
| $dV$ | $P^\top dO$ | $P$ |
| $dP$ | $dO\,V^\top$ | $V$ |
| $dS$ | softmax 反向（下一节） | $P$、$dP$ |
| $dQ$ | $dS\,K/\sqrt d$ | $K$ |
| $dK$ | $dS^\top Q/\sqrt d$ | $Q$ |

$P$ 出现了两次，所以标准实现在前向把 $P$ 存进 HBM 留给反向。它是 $s \times s$ 的：$s = 8192$、32 个 head、bf16 时，一层一个样本就是 $8192^2 \times 2\text{B} \times 32 \approx 4.3$ GB。几十层叠起来，长序列训练根本放不下。

## 前向只存 LSE

FlashAttention 前向不存 $P$，只存两样（Dao et al. 2022, FlashAttention, Alg. 1）：

- $O$：本来就是输出，要传给下一层，不额外占地方。
- 每行一个 $\text{LSE}_i = m_i + \log \ell_i$，其中 $m_i$ 是这一行的最大值，$\ell_i = \sum_j e^{S_{ij} - m_i}$ 是 online softmax 结束时的分母。用 fp32 存，上面那个例子里是 $8192 \times 4\text{B} \times 32 = 1$ MB。

反向重算 $P$ 时，有了 LSE 就不用再做 online softmax 那套逐块 rescale，一个 tile 一步就能还原：

$$e^{S_{ij} - \text{LSE}_i} = \frac{e^{S_{ij} - m_i}}{e^{\log \ell_i}} = \frac{e^{S_{ij} - m_i}}{\ell_i} = P_{ij}$$

前向要 online 是因为算到第一块时还不知道整行的最大值和分母；反向时这两个数已经由 LSE 给出，每个 tile 互不依赖。

## softmax 的反向

对一行 $p = \text{softmax}(x)$，$p_j = e^{x_j} / \sum_k e^{x_k}$，求偏导：

$$\frac{\partial p_j}{\partial x_k} = p_j(\delta_{jk} - p_k)$$

链式法则，已知 $dp$ 求 $dx$：

$$dx_k = \sum_j dp_j \, p_j(\delta_{jk} - p_k) = p_k\Big(dp_k - \sum_j p_j\, dp_j\Big)$$

对 attention 的每一行 $i$：

$$dS_{ij} = P_{ij}\big(dP_{ij} - D_i\big),\qquad D_i = \sum_k P_{ik}\, dP_{ik}$$

## 难点：$D_i$ 要整行

$dS$ 的每个元素都要减去同一个 $D_i$，而 $D_i$ 是对整行求和。分块时一个 tile 只看到这一行的 $B_c$ 列，看不到整行的 $P$ 和 $dP$。

把 $dP_{ik} = dO_i \cdot V_k$（$dO$ 第 $i$ 行和 $V$ 第 $k$ 行的点积）代进去：

$$D_i = \sum_k P_{ik}\,(dO_i \cdot V_k) = dO_i \cdot \Big(\sum_k P_{ik} V_k\Big) = dO_i \cdot O_i$$

所以

$$D = \text{rowsum}(dO \circ O)$$

只用到两个 $s \times d$ 的张量，和 $P$ 无关。FlashAttention v1 的反向已经用了这个恒等式（Dao et al. 2022, Appendix B.2），v2 把它挪到一个单独的预处理 kernel 里先算好（Dao 2023, FlashAttention-2, Alg. 2）。之后每个 tile 用 $P_{ij}$、$dP_{ij}$、$D_i$ 就能独立算出 $dS_{ij}$。

## 分块算法

tile 大小记 $B_r$（Q 方向）和 $B_c$（K/V 方向）。下面是 FA2 的循环结构：

```python
# 预处理 kernel
D = (dO * O).sum(-1)                         # [s]，fp32

# 主 kernel：每个 CTA 负责一个 K/V 块 j，并行铺到各 SM 上
for j in K/V 块:
    load K[j], V[j] 到 SRAM                   # [Bc, d]
    dK_j = dV_j = 0                           # 留在寄存器里累加，fp32
    for i in Q 块:                            # causal：跳过整块都在 j 之前的 i
        load Q[i], dO[i], LSE[i], D[i]
        S  = Q[i] @ K[j].T / sqrt(d)          # ① 重算 [Br, Bc]
        P  = exp(S - LSE[i])                  #    causal 的对角块要先 mask 成 -inf
        dV_j += P.T @ dO[i]                   # ②
        dP = dO[i] @ V[j].T                   # ③
        dS = P * (dP - D[i])
        dQ[i] += dS @ K[j] / sqrt(d)          # ④ 不同 j 写同一个 dQ[i] → atomicAdd（fp32）
        dK_j  += dS.T @ Q[i] / sqrt(d)        # ⑤
    写回 dK[j], dV[j]
```

整个过程中 $S$、$P$、$dP$、$dS$ 都只以 $B_r \times B_c$ 的 tile 存在 SRAM / 寄存器里，从不写回 HBM。

### 循环顺序

$dK_j$、$dV_j$ 要对所有 Q 块 $i$ 求和，$dQ_i$ 要对所有 K/V 块 $j$ 求和，所以总有一边得跨 CTA 累加：

| 外层 | 留在片上的 | 要跨 CTA 累加的 |
|---|---|---|
| K/V（FA1、FA2） | $dK_j$、$dV_j$ | $dQ$ |
| Q | $dQ_i$ | $dK$、$dV$ |

外层按 K/V 循环，只有 $dQ$ 一个要 atomic，所以 FA1 和 FA2 都这么排。FA1 的外层循环在 CTA 内串行，$dQ$ 是对 HBM 做读-改-写；FA2 把不同的 $j$ 分给不同的 CTA 并行（按序列维度并行），$dQ$ 就改成 fp32 的 atomicAdd，最后再转回 bf16。FA3 在 Hopper 上把 $dQ$ 的累加交给单独的 warp 处理，和主计算的 WGMMA 流水线重叠（Shah et al. 2024, FlashAttention-3）。

### GQA / MQA

多个 Q head 共享一组 K/V 时，这组 K/V 的 $dK$、$dV$ 是所有共享它的 Q head 的梯度之和。实现上要么让一个 CTA 依次处理同组的几个 Q head，在片上累加；要么每个 head 先写各自的 $dK$、$dV$，最后再做一次 reduce。

### dropout

attention dropout 的 mask 也是 $s \times s$ 的，同样不存。前向只记下 RNG 的 seed 和 offset，反向按同样的 tile 坐标重新生成同一个 mask（Dao et al. 2022, Alg. 2、Alg. 4）。现在的大模型预训练大多不开 attention dropout，这一步通常用不上。

## FLOPs 和显存账

每个 $s \times s \times d$ 的 matmul 记 $2s^2d$ FLOPs（一次乘加算 2 FLOPs）：

| | matmul | FLOPs |
|---|---|---|
| 前向 | $QK^\top$、$PV$ | $4s^2d$ |
| 标准反向（存了 $P$） | ②③④⑤ | $8s^2d$ |
| FlashAttention 反向 | ①②③④⑤ | $10s^2d$ |

一个训练 step 的 attention matmul，标准实现是 $12s^2d$，FlashAttention 是 $14s^2d$，多了约 17%。反向是前向的 2.5 倍，这也是 FA 反向的 MFU 通常比前向低的一个原因：同样的 tile 要做 5 个 matmul，加上 $dQ$ 的 atomic，寄存器压力更大。

换来的是 HBM 上不再有 $s^2$ 的东西：

| | 前向存给反向的 | 反向读写的 $s^2$ 张量 |
|---|---|---|
| 标准实现 | $P$（$s^2$） | 读 $P$，写读 $dP$、$dS$ |
| FlashAttention | $O$、LSE（$s$） | 无 |

标准实现的 attention 是 memory-bound，多算 17% 的 FLOPs 去换掉这些 $s^2$ 的读写，是净赚的。causal 时前向、反向都跳过约一半的 tile，表里每一项都近似减半。

这里的重算是 kernel 内部的事，和整层的 activation checkpointing（见 [tr-m-recompute](/stack/tr-m-recompute)）是两回事。后者是框架在反向时把整个 attention 的前向（包括 FA 前向 kernel）再跑一遍，两者可以叠加。

## 数值对拍

拿整矩阵的标准反向当参考，检查分块版本只用 $O$、$dO$、LSE 能不能得到同样的梯度（numpy，fp64，causal）：

```python
import numpy as np
rng = np.random.default_rng(0)
s, d, Br, Bc = 256, 64, 64, 64
Q, K, V, dO = (rng.standard_normal((s, d)) for _ in range(4))
scale = 1 / np.sqrt(d)
i = np.arange(s)[:, None]; j = np.arange(s)[None, :]
allowed = j <= i

# 前向（参考）
S = np.where(allowed, Q @ K.T * scale, -np.inf)
m = S.max(-1, keepdims=True)
LSE = (m + np.log(np.exp(S - m).sum(-1, keepdims=True)))[:, 0]
P = np.exp(S - LSE[:, None])
O = P @ V

# 标准反向：用整个 P
dV_ref = P.T @ dO
dP = dO @ V.T
dS = P * (dP - (P * dP).sum(-1, keepdims=True))
dQ_ref, dK_ref = dS @ K * scale, dS.T @ Q * scale

# 分块反向：只用 O、dO、LSE
D = (dO * O).sum(-1)
dQ, dK, dV = np.zeros_like(Q), np.zeros_like(K), np.zeros_like(V)
for k0 in range(0, s, Bc):
    Kj, Vj = K[k0:k0+Bc], V[k0:k0+Bc]
    dKj, dVj = np.zeros_like(Kj), np.zeros_like(Vj)
    for q0 in range(0, s, Br):
        if q0 + Br - 1 < k0:                  # 整块被 causal mask 掉
            continue
        qi = np.arange(q0, q0+Br)[:, None]; kj = np.arange(k0, k0+Bc)[None, :]
        Sij = np.where(kj <= qi, Q[q0:q0+Br] @ Kj.T * scale, -np.inf)
        Pij = np.exp(Sij - LSE[q0:q0+Br, None])
        dVj += Pij.T @ dO[q0:q0+Br]
        dSij = Pij * (dO[q0:q0+Br] @ Vj.T - D[q0:q0+Br, None])
        dQ[q0:q0+Br] += dSij @ Kj * scale
        dKj += dSij.T @ Q[q0:q0+Br] * scale
    dK[k0:k0+Bc], dV[k0:k0+Bc] = dKj, dVj

for n, a, b in (("Q", dQ, dQ_ref), ("K", dK, dK_ref), ("V", dV, dV_ref)):
    print(f"d{n}: {np.abs(a - b).max():.1e}")
# dQ: 7.8e-16  dK: 1.1e-15  dV: 1.8e-15
```

参考实现本身用有限差分抽查过，误差在 $10^{-9}$ 量级。

## 常见追问

- **为什么存 LSE 而不是分别存 $m$ 和 $\ell$？** 一个数就够还原 $P$，省一半，反向也少一次除法。
- **$D$ 为什么不在主 kernel 里算？** 每个 K/V 块的 CTA 都要用到所有行的 $D_i$。放在主 kernel 里就会被重复算 $s/B_c$ 遍；先单独算好，只要 $O(sd)$。
- **反向比前向慢多少？** matmul FLOPs 是 2.5 倍，实测墙钟时间一般是前向的 2.5–3 倍，多出来的来自 atomic 和更低的 Tensor Core 利用率。
