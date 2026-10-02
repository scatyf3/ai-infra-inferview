---
title: Multi-Head Attention
status: draft
tags: [attention, leetgpu]
difficulty: 2
order: 2
related: [/handson/mha-gqa-forward]
stack: []
leetgpu: [12]
---

# Multi-Head Attention

> [LeetGPU #12](https://leetgpu.com/challenges/multi-head-attention) · 无投影、无 mask、无 batch：Q/K/V 已经是 `[N, d_model]`，只练拆 head / 合并 head · 概念见 [MHA / GQA Forward](/handson/mha-gqa-forward)

## 题解

```python
import torch
import math


def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor,
    N: int, d_model: int, h: int,
):
    dk = d_model // h                                # 整除

    # [N, d_model] -> [N, h, dk] -> [h, N, dk]
    # h 挪到最前当 batch，matmul 只对最后两维做矩阵乘，等效于 for each head
    q = Q.reshape(N, h, dk).transpose(0, 1)
    k = K.reshape(N, h, dk).transpose(0, 1)
    v = V.reshape(N, h, dk).transpose(0, 1)

    attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk)   # [h, N, N]
    attn_norm = torch.softmax(attn, dim=-1)

    out = torch.matmul(attn_norm, v)                 # [h, N, dk]
    out = out.transpose(0, 1).reshape(N, d_model)    # [N, h, dk] -> [N, d_model]；transpose 后不连续，用 reshape
    output.copy_(out)
```

## 这题的坑

1. **不 transpose 直接 matmul**：`(N, h, dk) @ (N, dk, h)` 的 batch 维是 N，算出来是每个 token 内 head 两两之间的相似度，shape `(N, h, h)`，不报错但完全不是 attention。h 必须挪到 batch 位置。
2. **合并 head 时 transpose 错维度**：`[h, N, dk]` 换回 `[N, h, dk]` 是 `transpose(0, 1)`。写成 `transpose(2, 1)` 得到 `[h, dk, N]`，元素数一样，reshape 能过，结果静默错。
3. **缩放用 `sqrt(d_model)`**：除的是单 head 维度 `sqrt(dk)`。
4. **`matmul(..., out=output)`**：结果是 `[h, N, dk]`，和 output 的 `[N, d_model]` 对不上，output 会被 resize。这题必须局部算完再 `copy_`。

其余见 [通用语法坑](./#通用语法坑)。
