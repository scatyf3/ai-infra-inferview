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
    torch.matmul(torch.softmax(torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(d),dim=-1),V,out=output)

```

1. leetgpu的坑，要把out写到output，用torch的那个out即可
2. softmax要指定dim=-1
3. 记得`/math.sqrt(d)`

