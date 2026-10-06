---
title: Speculative Decoding Verification
status: draft
tags: [speculative-decoding, sampling, leetgpu]
difficulty: 3
order: 7
related: [/inference/speculative-decoding]
stack: [o-sampling]
leetgpu: [87]
---

# Speculative Decoding Verification

> [LeetGPU #87](https://leetgpu.com/challenges/speculative-decoding-verification) · rejection sampling 的验证步 · 概念见 [Speculative Decoding](/inference/speculative-decoding)

::: warning 记号和概念页相反
这题 $p$ = **draft**、$q$ = **target**，接受率是 $\min(1, q/p)$。[概念页](/inference/speculative-decoding#验证与分布保证) 沿用论文记号，$q$ = draft、$p$ = target，写成 $\min(1, p/q)$。两边是同一个公式：**target 在分子**。
:::

## 题意

对每条序列 $b$，从左到右处理 $i = 0, \dots, T-1$：

1. $\alpha_i = \min\!\left(1, \dfrac{q_i(t_i)}{p_i(t_i)}\right)$，只在 draft token $t_i$ 那一列取值
2. $u_i < \alpha_i$：接受 $t_i$，继续
3. $u_i \ge \alpha_i$：拒绝，停。从残差分布 $\text{norm}(\max(0, q_i - p_i))$ 用逆 CDF 采一个 token，随机数是 $r = u[b, T]$；残差全 0 时用均匀分布 $1/V$
4. 全部接受：从 $q_{T-1}$ 采一个 bonus token，同样用 $r = u[b, T]$

输出 `output_tokens[b, :]`，shape `[B, T+1]`：位置 $0$ 到接受个数（含）依次是接受的 token 和最后一个重采样 / bonus token，其余为 0。

## 题解

按「for 循环 + 分支」直接翻译题意。T 是 draft window，通常只有几个，循环没问题。

```python
import torch


def sample(probs, r):                                   # 逆 CDF：第一个 cdf > r 的位置
    cdf = torch.cumsum(probs, dim=0)
    idx = torch.searchsorted(cdf, r.reshape(1), right=True)
    return idx.clamp(max=probs.numel() - 1)             # 浮点误差可能让 cdf[-1] < r


# draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are tensors on the GPU
def solve(
    draft_tokens: torch.Tensor,     # [B, T] int32
    draft_probs: torch.Tensor,      # [B, T, V]
    target_probs: torch.Tensor,     # [B, T, V]
    uniform_samples: torch.Tensor,  # [B, T+1]，最后一列给 resample / bonus
    output_tokens: torch.Tensor,    # [B, T+1] int32
    B: int,
    T: int,
    V: int,
):
    output_tokens.zero_()                               # 没写到的位置要是 0
    for b in range(B):
        accept_len = 0
        for i in range(T):
            t = draft_tokens[b, i]
            acceptance_rate = min(1.0, (target_probs[b, i, t] / draft_probs[b, i, t]).item())
            if uniform_samples[b, i] < acceptance_rate:
                # accept
                output_tokens[b, i] = t
                accept_len += 1
            else:
                # reject: 从 max(0, q - p) 归一化后重新采样
                adj = torch.clamp(target_probs[b, i] - draft_probs[b, i], min=0)
                if adj.sum() > 0:
                    adj = adj / adj.sum()
                else:
                    adj = torch.full_like(adj, 1.0 / V)
                output_tokens[b, i] = sample(adj, uniform_samples[b, T])
                break
        if accept_len == T:
            # bonus: 从 q_{T-1} 采样
            output_tokens[b, T] = sample(target_probs[b, T - 1], uniform_samples[b, T])
```

逆 CDF：`cumsum` 把 $[0, 1)$ 按概率切成 $V$ 段，$r$ 落在哪段就输出哪个 token，`searchsorted` 二分找段。

::: details 只在 T 上循环、B 并行的版本（B 大时更快）
上面每个 `.item()` 和 `if tensor` 都是一次 GPU→CPU 同步，B 大时慢。用 `alive` 掩码代替 `break`，B 条序列一起往前走：

```python
def solve(draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens, B, T, V):
    dev = draft_probs.device
    rows = torch.arange(B, device=dev)
    out = torch.zeros(B, T + 1, dtype=torch.int32, device=dev)
    alive = torch.ones(B, dtype=torch.bool, device=dev)       # 到目前为止还没被拒绝
    n_acc = torch.zeros(B, dtype=torch.long, device=dev)

    for i in range(T):
        t = draft_tokens[:, i].long()
        p = draft_probs[rows, i, t]                           # [B]
        q = target_probs[rows, i, t]                          # [B]
        alive = alive & (uniform_samples[:, i] < torch.clamp(q / p, max=1.0))
        out[:, i] = torch.where(alive, draft_tokens[:, i].int(), 0)
        n_acc += alive.long()

    # 位置 n_acc 采一个 token：被拒绝 → 残差分布；全接受 → bonus，用 q_{T-1}
    j = n_acc.clamp(max=T - 1)
    q_row, p_row = target_probs[rows, j], draft_probs[rows, j]   # [B, V]
    rejected = (n_acc < T)[:, None]
    dist = torch.where(rejected, torch.clamp(q_row - p_row, min=0), q_row)
    s = dist.sum(dim=-1, keepdim=True)
    dist = torch.where(s > 0, dist / s, torch.full_like(dist, 1.0 / V))

    cdf = dist.cumsum(dim=-1)
    r = uniform_samples[:, T].contiguous().unsqueeze(-1)      # [B, 1]
    tok = torch.searchsorted(cdf, r, right=True).clamp(max=V - 1).squeeze(-1)
    out[rows, n_acc] = tok.int()
    output_tokens.copy_(out)
```
:::

两个版本都和逐 token 的朴素实现对拍过（300 组随机 B / T / V，含残差全 0 的情况）。

## 这题的坑

**会报错的**

1. **`min(1, tensor)`**：Python 内置 `min` 比较不了多元素 tensor。取出标量后 `.item()`，或者用 `torch.clamp(x, max=1.0)`。
2. **整个 `[B, T, V]` 相除**：`target_probs / draft_probs` 不仅多算了 V 倍，0/0 还会出 NaN。只需要 draft token 那一列：`[b, i, t]`，或批量用 `gather`。

**不报错、结果静默错的**

3. **比值写反**：`draft / target`。target 在分子，target 比 draft 更看好这个 token 就一定接受。
4. **接受条件写反**：`u > α` 是拒绝，`u < α` 才是接受。
5. **`accept_len=+1`**：这是赋值成 $+1$，不是加一。用 `+=`。
6. **拒绝后没有 `break`**：后面的 draft token 全部作废，不能继续验证。
7. **重采样用了 `u[b, i]`**：题目规定 resample 和 bonus 都用最后一列 `u[b, T]`，所以 `uniform_samples` 是 `[B, T+1]`。
8. **漏了 B 维**：注释里写成 `[T]`、`[T, V]`，实际都多一维 B。
9. **残差没归一化 / 没处理全 0**：$\max(0, q - p)$ 的和小于 1，要除以和；$q = p$ 时全 0，按题意退化成均匀分布。
10. **没清零 output**：题目要求剩余位置为 0，`output_tokens` 进来不一定是 0。
