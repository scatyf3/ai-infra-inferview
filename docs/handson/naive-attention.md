---
title: Naive Attention
status: draft
tags: [attention, handson]
difficulty: 1
order: 0.5
related: [/handson/torch-primitives, /handson/mha-gqa-forward, /handson/stable-softmax]
stack: [f-model]
leetgpu: [6]
---

# Naive Attention

> 单头、无 mask、无投影：$\text{softmax}(QK^T/\sqrt{d})\,V$ 三行写对

## 一句话结论

$Q \in \mathbb{R}^{M \times d}$，$K, V \in \mathbb{R}^{N \times d}$。score 是 $[M, N]$，**先缩放再 softmax**，softmax 沿最后一维（$N$，即对每个 query 在所有 key 上归一化），再乘 $V$ 得到 $[M, d]$。这是 [MHA / GQA](./mha-gqa-forward) 去掉多头和投影后的内核。

## 手撕

```python
import math
import torch


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Q: [M, d]，K/V: [N, d] -> [M, d]"""
    d = Q.shape[-1]
    attn = Q @ K.transpose(-1, -2) / math.sqrt(d)   # [M, N]；d 是 Python int，用 math.sqrt
    attn_norm = torch.softmax(attn, dim=-1)          # 沿 N 归一化：每个 query 对所有 key 的权重和为 1
    return attn_norm @ V                             # [M, d]
```

LeetGPU 版（写进传入的 `output`）见 [LeetGPU · Softmax Attention](/leetgpu/softmax-attention)。

### 常见错误

1. **缩放放在 softmax 之后**：结果每行和不再为 1，错。
2. **`torch.nn.Softmax(attn)`**：`nn.Softmax` 是 module 类，要实例化后再调用；demo 里直接用 `torch.softmax(x, dim=-1)`。
3. **`torch.sqrt(d)`**：`torch.sqrt` 只吃 tensor，标量用 `math.sqrt(d)` 或 `d ** 0.5`。
4. **softmax 的 dim 写错**：`dim=0` 是对每个 key 在所有 query 上归一化，语义错误但 shape 不报错。

## 面试追问

::: details Q：加上 batch 和 head 维要改什么？
上面已经用 `transpose(-1, -2)`，不用改——`matmul` 会把前面的维度当 batch 广播。完整版见 [MHA / GQA Forward](./mha-gqa-forward)。
:::

::: details Q：这段代码的显存瓶颈在哪？
`attn` 是 $M \times N$ 的中间矩阵，长序列时远大于 Q/K/V 本身，而且要写回 HBM 再读出来做 softmax 和第二次 matmul。FlashAttention 就是用分块 + [online softmax](./stable-softmax) 避免物化它。
:::
