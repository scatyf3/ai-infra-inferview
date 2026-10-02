---
title: Causal Self-Attention
status: draft
tags: [attention, mask, leetgpu]
difficulty: 2
order: 4
related: [/handson/mha-gqa-forward, /handson/naive-attention]
stack: []
leetgpu: [53]
---

# Causal Self-Attention

> [LeetGPU #53](https://leetgpu.com/challenges/causal-self-attention) · 单头，在 [Softmax Attention](./softmax-attention) 基础上加 causal mask · 概念见 [MHA / GQA Forward](/handson/mha-gqa-forward)

## 题解

```python
import torch
import math


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    # Q/K/V: [M, d]
    kt = K.transpose(0, 1)
    attn = torch.matmul(Q, kt) / math.sqrt(d)                               # [M, M]
    # 严格上三角（j > i，未来位置）为 -inf，对角线和下三角为 0
    mask = torch.full((M, M), float('-inf'), device=Q.device).triu(1)
    attn = attn + mask
    attn_norm = torch.softmax(attn, dim=-1)                                  # exp(-inf) = 0
    torch.matmul(attn_norm, V, out=output)                                   # [M, d]，shape 一致可以 out=
```

## mask 的两种写法

```python
# 加法 mask：上面题解的写法
mask = torch.full((M, M), float('-inf'), device=Q.device).triu(1)
attn = attn + mask

# 布尔 mask：下三角（含对角线）= 可见，取反后填 -inf
allowed = torch.tril(torch.ones(M, M, dtype=torch.bool, device=Q.device))
attn = attn.masked_fill(~allowed, float('-inf'))
```

`triu(k)` 保留第 k 条对角线及其右上方，其余置 0。`k=0` 从主对角线开始，`k=1` 从主对角线右边一条开始（主对角线也被置 0）。`tril(k)` 是下三角版本。

## 这题的坑

1. **`triu(0)`**：把对角线也盖成 -inf，token 看不到自己。第一行全是 -inf，softmax 得到 NaN。
2. **mask 方向反了**：行是 query、列是 key，第 i 行只能看列 j ≤ i，可见区域是下三角。
3. **mask 没建在 GPU 上**：`torch.full` 默认在 CPU，和 attn 相加会报 device 不一致，要传 `device=Q.device`。

其余见 [通用语法坑](./#通用语法坑)。
