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

# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    
    attn_raw = torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(d)
    
    q_pos = torch.arange(M,device=Q.device).unsqueeze(1)
    kv_pos = torch.arange(M,device=K.device).unsqueeze(0)

    casual_allowed = q_pos>=kv_pos
    attn_raw_masked = attn_raw.masked_fill(~casual_allowed,float("-inf"))
    attn_normed = torch.softmax(attn_raw_masked,dim=-1)
    out = torch.matmul(attn_normed,V)
    output.copy_(out)
```

## mask 的写法

**标准写法**（[mask 怎么拼](./mask)）：下标比较。

```python
q_pos = torch.arange(M,device=Q.device).unsqueeze(1)
kv_pos = torch.arange(M,device=K.device).unsqueeze(0)

casual_allowed = q_pos>=kv_pos
attn_raw_masked = attn_raw.masked_fill(~casual_allowed,float("-inf"))
```


## 这题的坑

1. **`triu(0)`**：把对角线也盖成 -inf，token 看不到自己。第一行全是 -inf，softmax 得到 NaN。
2. **mask 方向反了**：行是 query、列是 key，第 i 行只能看列 j ≤ i，可见区域是下三角。
3. **mask 没建在 GPU 上**：`torch.full` 默认在 CPU，和 attn 相加会报 device 不一致，要传 `device=Q.device`。

其余见 [通用语法坑](./#通用语法坑)。
