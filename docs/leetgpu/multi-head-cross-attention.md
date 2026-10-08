---
title: Multi-Head Cross-Attention
status: draft
tags: [attention, leetgpu]
difficulty: 2
order: 2.5
related: [/leetgpu/multi-head-attention, /handson/mha-gqa-forward]
stack: []
leetgpu: [26]
---

# Multi-Head Cross-Attention

> [LeetGPU #26](https://leetgpu.com/challenges/multi-head-cross-attention) · 无投影、无 mask · Q 有 M 个 token，K / V 有 N 个，head 已经拆好：`[*, H, D]` · 拆 head 的写法见 [Multi-Head Attention](./multi-head-attention)

## 题解

```python
import torch
import math


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,       # [M, H, D]
    K: torch.Tensor,       # [N, H, D]
    V: torch.Tensor,       # [N, H, D]
    output: torch.Tensor,  # [M, H, D]
    M: int,
    N: int,
    H: int,
    D: int,
):
    # 交换维度用 transpose；reshape 只改 shape、不搬数据
    q = Q.transpose(0, 1)                                    # [H, M, D]
    k = K.transpose(0, 1)                                    # [H, N, D]
    v = V.transpose(0, 1)                                    # [H, N, D]
    attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D), dim=-1)   # [H, M, N]
    o = torch.matmul(attn, v)                                # [H, M, D]
    output.copy_(o.transpose(0, 1))                          # [M, H, D]；copy_ 接受不连续的 tensor
```

提交的版本在每个 `transpose` 后面还跟了 `.view(H, N, D)` 之类，shape 没变，是空操作，可以删。

`einsum` 写法，不用想维度顺序：

```python
scores = torch.einsum('mhd,nhd->hmn', Q, K) / math.sqrt(D)   # 共有 h、d；d 不在输出里 → 求和
attn = torch.softmax(scores, dim=-1)
output.copy_(torch.einsum('hmn,nhd->mhd', attn, V))          # 对 n 求和，直接写出 [M, H, D]
```

## 和 #12 的区别

| | [#12 Multi-Head Attention](./multi-head-attention) | #26 Cross-Attention |
|---|---|---|
| 输入 | `[N, d_model]`，要自己拆成 `[N, h, dk]` | 已经是 `[*, H, D]` |
| Q 和 K/V 的长度 | 一样，都是 N | 不一样，M vs N |
| attention 矩阵 | `[h, N, N]` | `[H, M, N]` |

拆好之后都一样：把 H 挪到最前当 batch，`matmul` 只对最后两维做矩阵乘。

## 这题的坑

1. **用 `reshape(H, N, D)` 换维度**：`[N, H, D]` 在内存里是 token 优先排的，`reshape` 只是把同一串数据按新 shape 重新切开，切出来的「head 0」是好几个 token、好几个 head 混在一起。shape 对、数值错，静默算错。交换维度要 `transpose` / `permute`。
2. **输出也用 `reshape(M, H, D)` 换回来**：同一个错误，`[H, M, D]` 要 `transpose(0, 1)`。
3. **小测例测不出来**：M = N = H = 1 时 `reshape` 和 `transpose` 结果一样。

`reshape` / `view` 和 `transpose` / `permute` 的分工：

- 只**拆开或合并**相邻维度，顺序不变（`[M, H*D]` ↔ `[M, H, D]`）：`reshape` / `view`
- **调换**维度顺序（`[M, H, D]` → `[H, M, D]`）：`transpose` / `permute`

其余见 [通用语法坑](./#通用语法坑)。
