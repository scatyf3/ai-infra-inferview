---
title: Kimi Delta Attention
status: draft
tags: [attention, linear-attention, leetgpu]
difficulty: 2
order: 4.85
related: [/leetgpu/linear-self-attention]
stack: []
leetgpu: [120]
---

# Kimi Delta Attention

> [LeetGPU #120](https://leetgpu.com/challenges/kimi-delta-attention) · 出自 Kimi Linear（Moonshot AI 2025, [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)），是 Gated DeltaNet（Yang et al. 2024, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)）的改进版 · 前置：[Linear Self-Attention](./linear-self-attention)

每个 batch 的状态从 $S_0 = 0$ 开始，按 $t = 1, \dots, L$ 逐步更新：

$$
S_t = \left(I - \beta_t k_t k_t^\top\right) \mathrm{Diag}(\alpha_t)\, S_{t-1} + \beta_t k_t v_t^\top,\qquad o_t = S_t^\top q_t
$$

- `Q K V alpha output`：(B, L, d)；`beta`：(B, L)；状态 $S_t$：(d, d)
- $\alpha_t \in (0,1]^d$：按通道的遗忘门。Gated DeltaNet 每个 head 只有一个标量 $\alpha_t$，KDA 把它换成长度为 d 的向量
- $\beta_t \in [0,1]$：写入强度
- K 的每一行都做了 L2 归一化

## v1：逐步递推

题面怎么写就怎么算，每步 O(d²)。

```python
import torch


# Q, K, V, alpha, beta, output are tensors on the GPU
# KDA = gated delta rule + channel-wise decay
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    batch: int,
    seq_len: int,
    d: int,
):
    # Q, K, V, alpha, and output :(B,L,d)
    S = torch.zeros(batch, d, d, device=Q.device, dtype=Q.dtype)
    for t in range(seq_len):
        # diag => elementwise mul
        # (B,d,1) * (B,d,d),in dim=1, which is k dim
        S = alpha[:, t, :, None] * S  # diag(alpha) * s
        # qkv (B, d), slice for cur seq
        k, v, q = K[:, t], V[:, t], Q[:, t]
        #  v - S^T k
        # (B,1,d)@(B,d,d) =>(B,1,d) =>(B,d), which is V
        kS = torch.matmul(k.unsqueeze(1), S).squeeze(1)
        err = v - kS
        # update s controlled by beta
        # beta(B,1,1)
        # k(B,d) => (B, d, 1)
        # err(B,d) => (B, 1, d)
        # S(B,d,d), beta * k * err : (B,d,d)
        S = S + beta[:, t, None, None] * k[:, :, None] * err[:, None, :]
        # o : (B,d,d) @ (B,d,1) => (B,d,1) => (B,d)
        output[:, t] = torch.matmul(S.transpose(-1,-2),q.unsqueeze(-1)).squeeze(-1)
        # output[:, t] = torch.einsum('bi,bij->bj', q, S)
```

本地验证了两项：题面例子输出 `[[1,2],[1.5,1]]`；随机数据（B=2，L=64，d=16，float64）和照原式显式构造 $(I-\beta kk^\top)\mathrm{Diag}(\alpha)$ 的参考实现相比，最大误差 1.6e-15。

## 公式怎么变成代码

**1. $\mathrm{Diag}(\alpha)$ 左乘就是逐行缩放**

$$
(\mathrm{Diag}(\alpha)\,S)_{ij} = \sum_m \mathrm{Diag}(\alpha)_{im} S_{mj} = \alpha_i S_{ij}
$$

第 i 行整行乘 $\alpha_i$，所以把 α 变成列向量 `(B,d,1)` 去广播就行，不用 `diag_embed` 构造 d×d 矩阵再做 O(d³) 的矩阵乘。S 的行对应 key 维，这是因为写入的是 $kv^\top$，行下标来自 k。所以 α 衰减的是 key 的通道。

右乘 $S\,\mathrm{Diag}(\alpha)$ 才是缩放列，那时要写 `alpha[:, t, None, :]`。

**2. 把 delta 更新展开**

记 $S' = \mathrm{Diag}(\alpha_t) S_{t-1}$：

$$
\begin{aligned}
S_t &= (I - \beta k k^\top)\,S' + \beta k v^\top \\
&= S' - \beta\,k\,(k^\top S') + \beta\,k\,v^\top \\
&= S' + \beta\,k\,\big(v - S'^\top k\big)^\top
\end{aligned}
$$

- `err` $= v - S'^\top k$：用 k 从当前状态读出来的值，和想存进去的 v 之差。delta rule 只把这个差值写回去，等价于在 $\frac12\lVert S^\top k - v\rVert^2$ 上做一步步长为 β 的梯度下降。
- 原式要构造 $I - \beta kk^\top$ 再和 S 做矩阵乘，O(d³)；展开后只剩一次矩阵乘向量和一次外积，O(d²)。

**3. 外积靠两个 `None` 放在不同位置**

| 写法 | 形状 |
|---|---|
| `beta[:, t, None, None]` | (B, 1, 1) |
| `k[:, :, None]` | (B, d, 1)，列向量 |
| `err[:, None, :]` | (B, 1, d)，行向量 |
| 乘积 | (B, d, d)，第 `[i,j]` 个元素是 $\beta\,k_i\,\text{err}_j$ |

## 这题的坑

1. **用 `matmul` 算出来的向量没 `squeeze`**：`(B,d,d) @ (B,d,1)` 得到 `(B,d,1)`，赋给 `output[:, t]` 这个 `(B,d)` 会报 `expand ... the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (3)`。广播只能补维度，不能删维度。
2. **衰减和读出的顺序反了**：`err` 必须用衰减后的 $S'$ 来算，所以先乘 α，再算 `kS`。题面例子第 2 步用的就是衰减后的状态 `[[0.5,1],[0,0]]`。
3. **外积写成逐元素乘**：`k` 和 `err` 都写成 `[:, :, None]`，结果是 `(B,d,1)`，静默算错。
4. **把 `t` 当切片**：`K[:, t]` 用整数索引，会删掉 L 这一维得到 `(B,d)`；`K[:, t:t+1]` 才会保留成 `(B,1,d)`。

其余见 [通用语法坑](./#通用语法坑)。

## v1 的性能问题

性能测试的规模是 B=4，L=8192，d=128。v1 要在 Python 里循环 8192 步，每步启动好几个小 kernel，时间基本都花在 launch 上。

这题没法套 linear attention 的 parallel 形式 `mask(QKᵀ)V`。那种写法要求状态转移是标量或对角阵，而 $(I-\beta kk^\top)\mathrm{Diag}(\alpha)$ 两样都不是。所以走 chunkwise：块内并行，块间仍按递推传递 S，循环从 L 步降到 L/C 步。

## v2：chunkwise

思路来自 DeltaNet 的分块算法（Yang et al. 2024, *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)），再加上按通道的衰减。

### 推导

只看一个块：块内位置 $r = 1..C$，块初状态 $S_0$。块内累积衰减为

$$
\gamma_r = \prod_{s \le r} \alpha_s \in \mathbb{R}^d,\qquad g_r = \log \gamma_r = \sum_{s\le r} \log\alpha_s
$$

**1. 把 delta rule 改写成"带衰减的 linear attention + 新的 value"。** 由 v1 的展开式，令

$$
u_r = \beta_r\,(v_r - S_r'^\top k_r),\qquad S_r = \mathrm{Diag}(\alpha_r) S_{r-1} + k_r u_r^\top
$$

这样每步只是衰减再加一个外积，和 gated linear attention 的形式一样，只不过写进去的是 $u_r$ 而不是 $v_r$。

**2. 展开递推。** 衰减从 s 传到 r 累积成 $\gamma_r/\gamma_s$：

$$
S_r = \mathrm{Diag}(\gamma_r)\,S_0 + \sum_{s \le r} \mathrm{Diag}(\gamma_r/\gamma_s)\,k_s u_s^\top
$$

**3. 解出 U。** $u_r$ 里用到的 $S_r' = \mathrm{Diag}(\alpha_r)S_{r-1}$ 只包含 $s<r$ 的项，代入得到

$$
u_r = \beta_r\Big(v_r - S_0^\top(\gamma_r \odot k_r) - \sum_{s<r} \underbrace{k_r^\top \mathrm{Diag}(\gamma_r/\gamma_s)\,k_s}_{\text{标量}}\; u_s\Big)
$$

$u_r$ 依赖前面所有的 $u_s$。写成矩阵形式是一个下三角线性方程组：

$$
(I + A)\,U = \mathrm{diag}(\beta)\,(V - \tilde K S_0),\qquad
A_{rs} = \beta_r \sum_c k_{rc}\,k_{sc}\,e^{g_{rc} - g_{sc}}\ \ (s<r)
$$

其中 $\tilde K$ 的第 r 行是 $\gamma_r \odot k_r$。$I + A$ 是单位下三角矩阵，用 `solve_triangular` 求解，复杂度 O(C²d)。

**4. 输出。** 读的是写入之后的 $S_r$，所以求和包含 $s=r$：

$$
O = \tilde Q S_0 + P\,U,\qquad P_{rs} = \sum_c q_{rc}\,k_{sc}\,e^{g_{rc}-g_{sc}}\ \ (s \le r)
$$

**5. 传给下一块的状态。** 在步骤 2 里取 $r = C$：

$$
S_C = \mathrm{Diag}(\gamma_C)\,S_0 + \sum_s (\gamma_C/\gamma_s \odot k_s)\,u_s^\top
$$

### 代码

```python
import torch


def solve(Q, K, V, alpha, beta, output, batch, seq_len, d, C=64):
    S = torch.zeros(batch, d, d, device=Q.device, dtype=Q.dtype)
    for st in range(0, seq_len, C):
        ed = min(st + C, seq_len)
        c = ed - st                                                # 最后一块可能不满 C
        q, k, v = Q[:, st:ed], K[:, st:ed], V[:, st:ed]           # (B,c,d)
        b = beta[:, st:ed]                                         # (B,c)
        # 块内累积衰减，log 空间：g_r = sum_{s<=r} log alpha_s
        g = torch.cumsum(torch.log(alpha[:, st:ed]), dim=1)        # (B,c,d)

        # 两两衰减 exp(g_r - g_s)：s <= r 时 <= 1，不会溢出；s > r 先填 -inf，exp 后为 0
        pos = torch.arange(c, device=Q.device)
        incl = pos[:, None] >= pos[None, :]                        # (c,c) 含对角
        strict = pos[:, None] > pos[None, :]                       # (c,c) 不含对角
        diff = g[:, :, None, :] - g[:, None, :, :]                 # (B,c,c,d)  r 行 s 列
        decay = torch.exp(diff.masked_fill(~incl[None, :, :, None], float('-inf')))

        # A[r,s] = beta_r * sum_c k_r k_s exp(g_r - g_s), s < r
        kk = (k[:, :, None, :] * k[:, None, :, :] * decay).sum(-1)   # (B,c,c)
        A = b[:, :, None] * kk.masked_fill(~strict, 0)
        # P[r,s] = sum_c q_r k_s exp(g_r - g_s), s <= r
        P = (q[:, :, None, :] * k[:, None, :, :] * decay).sum(-1)    # (B,c,c)

        k_dec = k * torch.exp(g)                                   # (B,c,d) gamma_r * k_r
        q_dec = q * torch.exp(g)

        # (I + A) U = beta * (V - K_dec S0)
        rhs = b[:, :, None] * (v - torch.matmul(k_dec, S))         # (B,c,d)
        I = torch.eye(c, device=Q.device, dtype=Q.dtype)
        U = torch.linalg.solve_triangular(I + A, rhs, upper=False, unitriangular=True)

        # O = Q_dec S0 + P U : (B,c,d)@(B,d,d) + (B,c,c)@(B,c,d)
        output[:, st:ed] = torch.matmul(q_dec, S) + torch.matmul(P, U)

        # S_C = Diag(gamma_C) S0 + K_end^T U
        g_last = g[:, -1:, :]                                      # (B,1,d)
        k_end = k * torch.exp(g_last - g)                          # (B,c,d) gamma_C/gamma_s * k_s
        S = torch.exp(g_last).transpose(-1, -2) * S + torch.matmul(k_end.transpose(-1, -2), U)
```

本地和 v1 逐元素对拍（CPU）：

| 设置 | float64 最大误差 | float32 最大误差 |
|---|---|---|
| B=2, L=200, d=16, C=64, α~U(0.5,1) | 4.7e-15 | 2.6e-6 |
| 同上，C=16 | 2.7e-15 | 1.9e-6 |
| B=2, L=257, d=32, C=64, α~U(0.001,1)，最后一块不满 | 1.2e-14 | 6.4e-6 |
| L=1 | 3.5e-17 | 7.5e-8 |

题面例子也输出 `[[1,2],[1.5,1]]`。float32 的误差在 1e-6 量级，来自求和顺序不同，题面没写误差容限。GPU 上没测过速度。

### v2 的坑

1. **两两衰减拆成 $(k_r\gamma_r)\cdot(k_s/\gamma_s)$ 来算**：数学上等价，也能直接用一次 matmul。但 $1/\gamma_s = e^{-g_s}$ 在 float32 下超过 $e^{88}$ 就溢出了。实测 α~U(0.001, 0.2)、C=64 时输出出现 inf/NaN；C=16 时没出现，log 空间版本在 C=64 下也正常。所以要先算差 $g_r - g_s \le 0$ 再取 exp。代价是多一个 `(B,c,c,d)` 的中间张量。
2. **mask 要在 exp 之前用 -inf 填**：$s>r$ 的位置 $g_r - g_s > 0$，先 exp 可能已经溢出成 inf。
3. **A 不含对角，P 含对角**：$u_r$ 用的是写入 r 之前的状态，所以只到 $s<r$；$o_r$ 读的是写入之后的 $S_r$，所以包含 $s=r$。
4. **最后一块不满 C**：mask 和 `eye` 都要用实际长度 `c`。
5. **块末衰减要缩放行**：`exp(g_last)` 是 `(B,1,d)`，要转置成 `(B,d,1)` 再乘 S，和 v1 里 `alpha[:, t, :, None]` 是同一回事。

### 还能再快

v2 还剩 L/C 步的串行循环，每步十几个 kernel。下一步是 WY 拆分：把 U 拆成 $U = \underbrace{T\,\mathrm{diag}(\beta) V}_{U_v} - \underbrace{T\,\mathrm{diag}(\beta)\tilde K}_{W}\,S_0$，其中 $T = (I+A)^{-1}$。

$T$、$W$、$U_v$、$P$ 都只依赖块内数据，可以对所有块一次性批量算完。循环里只剩 $U = U_v - WS_0$、$O$、$S_C$ 三次 matmul。

代价是 `(B, L/C, C, C, d)` 的衰减张量，在 B=4、L=8192、C=64 时约 1 GB。FLA 的 KDA kernel 用块内再分子块来避开它，参考题面附的 [KDA walkthrough](https://jianyuh.github.io/attention/2025/12/13/KDA.html)。
