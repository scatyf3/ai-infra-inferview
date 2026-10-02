---
title: Softmax Attention
status: draft
tags: [attention, leetgpu]
difficulty: 1
order: 1
related: [/handson/naive-attention]
stack: []
leetgpu: [6]
---

# Softmax Attention

> [LeetGPU #6](https://leetgpu.com/challenges/softmax-attention) · 单头、无 mask、无投影 · 概念见 [Naive Attention](/handson/naive-attention)

## 题解

```python
import torch
import math


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int
):
    '''
    Q: M×d
    K: N×d
    V: N×d
    '''
    K_T = torch.transpose(K, 0, 1)                  # [d, N]
    attn = torch.matmul(Q, K_T) / math.sqrt(d)      # [M, N]
    attn_norm = torch.softmax(attn, dim=-1)         # 沿 N 归一化
    torch.matmul(attn_norm, V, out=output)          # [M, d]，shape 和 output 一致，可以直接 out=
```

## 这题的坑

- **`return` 结果而不写 `output`**：判题只看传入的 `output`。
- 其余见 [通用语法坑](./#通用语法坑)。
