---
title: Linear Self-Attention
status: draft
tags: [attention, leetgpu]
difficulty: 3
order: 4.8
related: [/handson/naive-attention, /leetgpu/kimi-delta-attention]
stack: []
leetgpu: [56]
---

# Linear Self-Attention

> [LeetGPU #56](https://leetgpu.com/challenges/linear-self-attention) · 单头、无 mask，用特征映射 φ 代替 softmax · 出自 Katharopoulos et al. 2020, *Transformers are RNNs*

$$
\text{LinearAttention}(Q,K,V) = \frac{\phi(Q)\,\big(\phi(K)^T V\big)}{\phi(Q)\,\big(\sum_j \phi(K_j)\big)},\qquad
\phi(x) = \text{ELU}(x)+1 = \begin{cases} x+1 & x>0 \\ e^x & x\le 0 \end{cases}
$$

## 题解

```python
import torch


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    # Q/K/V: (M,d)
    # phi = elu(x)+1
    # clamp to avoid nan
    phi_q = torch.where(Q <= 0, torch.exp(Q.clamp(max=0)), Q + 1)
    phi_k = torch.where(K <= 0, torch.exp(K.clamp(max=0)), K + 1)
    S = torch.matmul(phi_k.transpose(-1,-2),V)
    unnorm_o = torch.matmul(phi_q,S)
    sum_k=torch.sum(phi_k,dim=0) # sum over seq (j), not over d -> (d,)
    # phi_q (M,d)@(d,) ->(M,)
    norm = torch.matmul(phi_q,sum_k)
    output.copy_(unnorm_o/norm.unsqueeze(-1))
```

φ 也可以直接写 `F.elu(x) + 1`。

## shape

| 量 | shape | 说明 |
|---|---|---|
| `phi_q`, `phi_k` | (M, d) | 逐元素映射，不改 shape |
| `S = phi_k^T V` | (d, d) | 先算这个，复杂度 O(M d²)，不出现 M×M 矩阵 |
| `unnorm_o = phi_q S` | (M, d) | 分子 |
| `sum_k` | (d,) | $\sum_j \phi(K_j)$，对序列维求和 |
| `norm = phi_q @ sum_k` | (M,) | 分母，每个 query 一个标量 |

## 这题的坑

1. **`sum_k` 的求和维度错了**：$\sum_j$ 是对序列维 j 求和，应该用 `dim=0`，得到 `(d,)`。写成 `dim=-1` 就成了对 d 求和，得到 `(M,)`。
2. **分母写成逐元素乘**：分母是 $\phi(Q_i)$ 和 `sum_k` 的点积，要用 `phi_q @ sum_k`。写成 `phi_q * sum_k` 是 `(M,d) * (M,)`：M ≠ d 时报错，M == d 时静默算错。
3. **除法前没 `unsqueeze`**：`(M,d) / (M,)` 会按最后一维对齐，变成 M 和 d 去对。要写成 `norm.unsqueeze(-1)`，即 `(M,1)`。
4. **`torch.where` 里的 exp 溢出**：`where` 两个分支都会对全部元素计算。x 较大时（float32 下大约 x > 88）`exp(x)` 得到 inf，前向不受影响，但反向会出现 `0 * inf = nan`。所以先 `clamp(max=0)` 再取 exp。
5. **照着 softmax attention 加了 `/ sqrt(d)`**：公式里没有缩放，这里也没有 softmax。

其余见 [通用语法坑](./#通用语法坑)。

## Linear Attention 家族

### 统一成状态递推

这题算的 `S = phi_k^T V` 是对整条序列求和。如果只累加到第 t 步，就得到 causal 版本，也就是一个 RNN（Katharopoulos et al. 2020 标题里 *Transformers are RNNs* 说的就是这个）：

$$
S_t = S_{t-1} + \phi(k_t)\,v_t^\top,\qquad o_t = S_t^\top \phi(q_t)
$$

状态 $S_t$ 是 $d_k\times d_v$ 的矩阵，大小固定，和序列长度无关。这就是 linear attention 推理时不需要 KV cache 的原因。

后来的变体都只在改状态更新的公式，可以按两个独立的方向分类：
- **有没有遗忘门 α**：门是标量，还是按通道的向量；
- **写入方式**：直接累加 $kv^\top$，还是用 delta rule 只写入误差 $\beta k(v - S^\top k)^\top$。

| | 无门 | 标量门 $\alpha_t$ | 按通道门 $\mathrm{Diag}(\alpha_t)$ |
|---|---|---|---|
| 直接累加 $S \mathrel{+}= kv^\top$ | Linear Attention（Katharopoulos et al. 2020） | Mamba2（Dao & Gu 2024, [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)） | GLA（Yang et al. 2023, [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)） |
| delta rule | DeltaNet（Schlag et al. 2021, [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)） | Gated DeltaNet（Yang et al. 2024, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)） | KDA（Moonshot AI 2025, [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)），题解见 [Kimi Delta Attention](./kimi-delta-attention) |

右下角 KDA 的完整递推：

$$
S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top
$$

把 $\mathrm{Diag}(\alpha_t)$ 换成标量 $\alpha_t$ 就是 Gated DeltaNet，令 $\alpha_t = 1$ 就是 DeltaNet。去掉 $(I - \beta_t k_t k_t^\top)$、令 $\beta_t = 1$，就退化成第一行的三种。

后面这些模型都去掉了这题的分母 $\phi(Q)\sum_j\phi(K_j)$，改成对 q、k 做 L2 归一化，再对输出做 norm。

### chunkwise 怎么算

训练和 prefill 时不能逐步递推（太慢），也不能像这题一样一次 matmul 算完（causal 时块和块之间有依赖）。所以按长度 C 分块：块内并行，块间传递状态 $S_0$。记块内累积衰减 $g_r = \sum_{s\le r}\log\alpha_s$：

- **直接累加的那一行**：块内输出是 $O = \tilde Q S_0 + (QK^\top \odot \text{衰减 mask})\,V$，两次 matmul 就够了。
- **delta rule 那一行**：每步写入的 $u_r = \beta_r(v_r - S_r'^\top k_r)$ 依赖前面所有的 $u_s$，所以要多解一个下三角方程组 $(I+A)U = \mathrm{diag}(\beta)(V - \tilde K S_0)$，然后用 U 代替 V。推导见 [Kimi Delta Attention](./kimi-delta-attention#v2-chunkwise)。

### 标量门和按通道门的区别

在 chunk 版本里，这一列的区别体现在块内两两衰减的位置：

$$
\text{标量门（GDN）:}\quad A_{rs} = \beta_r\,e^{g_r-g_s}\,(k_r^\top k_s)
\qquad\qquad
\text{按通道门（KDA）:}\quad A_{rs} = \beta_r \sum_c k_{rc}\,k_{sc}\,e^{g_{rc}-g_{sc}}
$$

- **标量门**：衰减可以提到内积外面。先用一次 matmul 算 $KK^\top$，再逐元素乘一个 `(c,c)` 的衰减矩阵，在 GPU 上走 tensor core。
- **按通道门**：每个通道衰减不同，只能放进内积里。直接写会出现 `(B,c,c,d)` 的中间张量，运算量和显存都多 d 倍。要用 matmul 只能拆成 $(k\odot\gamma)(k/\gamma)^\top$，但 $1/\gamma$ 会溢出，所以 FLA 的 kernel 会把块再切成子块。

GLA 和 Mamba2 之间也是同样的区别。

[KDA 的 chunk 版](./kimi-delta-attention#代码)改成 GDN，需要改的就是这几行，解 U 和算输出不用动：

```python
# alpha: (B,L) 每个 head 一个标量
g = torch.cumsum(torch.log(alpha[:, st:ed]), dim=1)            # (B,c)    KDA 是 (B,c,d)
diff = g[:, :, None] - g[:, None, :]                            # (B,c,c)  KDA 是 (B,c,c,d)
decay = torch.exp(diff.masked_fill(~incl, float('-inf')))

kk = torch.matmul(k, k.transpose(-1, -2)) * decay                # 先 matmul 再乘衰减
A = b[:, :, None] * kk.masked_fill(~strict, 0)
P = torch.matmul(q, k.transpose(-1, -2)) * decay

k_dec = k * torch.exp(g)[..., None]                             # 标量广播到 d
q_dec = q * torch.exp(g)[..., None]
g_last = g[:, -1:]                                              # (B,1)
k_end = k * torch.exp(g_last - g)[..., None]
S = torch.exp(g_last)[..., None] * S + torch.matmul(k_end.transpose(-1, -2), U)   # 整个 S 一起缩放
```

验证方法：GDN 就是 KDA 在 α 各通道取相同值时的特例。把标量 α 复制成 d 维，交给 KDA 的逐步版当参考，B=2、L=257、d=32 时误差在 float64 下是 7e-15，float32 下是 4e-6。

从建模上看，标量门每一步只能让整个记忆一起变淡；按通道门可以让不同的 key 通道以不同速度遗忘，有的通道存长期信息，有的很快被覆盖。
