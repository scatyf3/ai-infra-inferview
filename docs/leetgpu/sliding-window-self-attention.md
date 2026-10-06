---
title: Sliding Window Self-Attention
status: draft
tags: [attention, sliding-window, leetgpu]
difficulty: 2
order: 4.5
related: [/handson/naive-attention, /inference/attention-variants, /leetgpu/causal-self-attention]
stack: []
leetgpu: [59]
---

# Sliding Window Self-Attention

> [LeetGPU #59](https://leetgpu.com/challenges/sliding-window-self-attention) · 单头、无投影、双向窗口 · 在 [Softmax Attention](./softmax-attention) 上加一个带状 mask

query $i$ 只看 $|i - j| \le w$ 的 key，**这题 $w$ 就是 `window_size`**（单侧宽度，总宽 $2w + 1$）。

## 题解

```python
import torch
import math


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d: int,
    window_size: int,
):
    attn = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d)          # [M, M]
    full = torch.full((M, M), float('-inf'), device=Q.device)
    mask_upper = torch.triu(full, diagonal=window_size + 1)    # j - i >= w+1 的位置 -inf，其余 0
    mask_lower = torch.tril(full, diagonal=-(window_size + 1)) # i - j >= w+1 的位置 -inf，其余 0
    mask = mask_upper + mask_lower                             # 带内 0，带外 -inf
    attn = attn + mask
    attn = torch.softmax(attn, dim=-1)                         # 沿 key 方向归一化
    torch.matmul(attn, V, out=output)                          # [M, d]
```

**标准写法**（[mask 怎么拼](./mask)）：下标比较，不推 diagonal：

```python
i = torch.arange(M, device=Q.device)[:, None]                # [M, 1] query
j = torch.arange(M, device=Q.device)[None, :]                # [1, M] key
allowed = (i - j).abs() <= window_size                       # [M, M] bool
attn = attn.masked_fill(~allowed, float('-inf'))
```

## `triu` / `tril` 的 diagonal

`torch.triu(x, diagonal=k)` **保留** $j - i \ge k$ 的元素，其余置 0；`tril(x, diagonal=k)` 保留 $j - i \le k$。
这里拿全 `-inf` 矩阵去做，保留下来的是**要屏蔽的**部分，所以屏蔽 $j - i > w$ 要写 $k = w + 1$，屏蔽 $i - j > w$ 要写 $k = -(w + 1)$。写成 $k = w$ 会把距离恰好为 $w$ 的边界也屏蔽掉。

## window_size 到底指什么

各家定义不统一，做题以题面为准：

| 出处 | 方向 | `window_size` 含义 | query $i$ 能看的 $j$ |
|---|---|---|---|
| LeetGPU #59 | 双向 | 单侧宽度 | $\lvert i - j\rvert \le W$ |
| Longformer | 双向 | 总宽度 | $\lvert i - j\rvert \le W/2$ |
| Mistral / Gemma 2（LLM 里的 SWA） | 因果 | 总宽度（含自己） | $0 \le i - j < W$ |

判断方法：看最小测例。本题 `M = 2, window_size = 1` 时期望输出是两行 V 的加权混合，说明两个位置互相可见，所以是单侧宽度。

LLM 里用的是因果窗口，好处是 KV cache 只需要留最近 $W$ 个 token，显存 $O(W)$ 而不是 $O(\text{seq})$，实现上用 ring buffer 按 `pos % W` 覆盖写。

## 这题的坑

**会报错的**

1. **`torch.triu(window_size // 2, (M, M), -inf)`**：`triu` / `tril` 不生成矩阵，签名是 `(input, diagonal=k)`。先 `torch.full((M, M), float('-inf'))` 再截。
2. **`torch.empty(1, (M, M))`**：参数格式不对；而且 `empty` 是未初始化内存，不能当 mask 用。

**不报错、结果静默错的**

3. **`softmax(attn, axis=0)`**：`attn[i, j]` 是 query $i$ 对 key $j$ 的分数，要在每个 query 的那一行上归一化，用 `dim=-1`。方阵沿错方向也不会报 shape 错。
4. **`w = window_size // 2`**：把单侧宽度当成了总宽度。`window_size = 1` 时 $w = 0$，每个 query 只看自己，输出原样等于 V。
5. **diagonal 写成 `w` 而不是 `w + 1`**：边界那条对角线也被屏蔽了。
