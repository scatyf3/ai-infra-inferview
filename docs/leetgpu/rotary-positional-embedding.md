---
title: Rotary Positional Embedding
status: draft
tags: [rope, leetgpu]
difficulty: 2
order: 6.5
related: [/handson/torch-primitives]
stack: []
leetgpu: [61]
---

# Rotary Positional Embedding

> [LeetGPU #61](https://leetgpu.com/challenges/rotary-positional-embedding) · Q / cos / sin 都是 `[M, D]` · 纯逐元素，没有 reduce · 算子见 [torch 原语：拼接 / 切分](/handson/torch-primitives#拼接-切分)

$$
\text{RoPE}(x) = x \odot \cos + \text{rotate\_half}(x) \odot \sin
$$

$$
\text{rotate\_half}([x_1, \dots, x_{d/2}, x_{d/2+1}, \dots, x_d]) = [-x_{d/2+1}, \dots, -x_d, x_1, \dots, x_{d/2}]
$$

## 思路

两件事：

1. **逐元素乘**：`*` 就是 $\odot$，Q / cos / sin 形状相同，直接乘
2. **rotate_half**：每行沿最后一维切成前后两半，后半取负放前面、前半放后面，再 `cat` 回去

```
x    = [a, b, c, d]
rot  = [-c, -d, a, b]
```

## 题解

```python
import torch


# Q, cos, sin, output are tensors on the GPU
def solve(
    Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, output: torch.Tensor, M: int, D: int
):
    Q_cos = Q * cos                                   # 逐元素乘，[M, D]
    half = D // 2
    Q_prev = Q[:, :half]                              # 每行前一半 [M, D/2]
    Q_last = Q[:, half:]                              # 每行后一半 [M, D/2]
    Q_rot = torch.cat([-Q_last, Q_prev], dim=-1)      # 沿最后一维拼：D/2 + D/2 = D
    Q_sin = Q_rot * sin
    res = Q_cos + Q_sin
    output.copy_(res)                                 # 写进调用方给的 buffer
```

把两半展开写，就是 LLM 里常见的成对旋转：

$$
\begin{aligned}
y_{[:d/2]} &= x_1 \cos_{[:d/2]} - x_2 \sin_{[:d/2]} \\
y_{[d/2:]} &= x_2 \cos_{[d/2:]} + x_1 \sin_{[d/2:]}
\end{aligned}
$$

## 这题的坑

1. **负号放错半边**：`rotate_half` 是 `[-x2, x1]`，不是 `[x2, -x1]`。写反了不报错，结果整体差个符号。
2. **`cat` 的 `dim` 写成 0**：两个 `[M, D/2]` 上下叠成 `[2M, D/2]`，和 cos 相乘时报 shape 不匹配。要沿最后一维左右接，`dim=-1`。
3. **`torch.cat(a, b)`**：第一个参数是一组 tensor（元组或 list），`b` 会被当成 `dim` 报错。
4. **`output = res`**：只改了局部名字，调用方那块显存没写，要 `output.copy_(res)`。

其余见 [通用语法坑](./#通用语法坑)。

## 和 LLM 里的 RoPE

- 真实模型里 cos / sin 是按位置 `pos` 和频率 $\theta_i = 10000^{-2i/d}$ 预先算好的表，前后两半是同一组值（HF 里 `emb = cat((freqs, freqs), dim=-1)`）。本题直接把表给你，不用自己算。
- 输入是 `[B, H, S, D]` 时，`dim=-1` 和切片 `x[..., :half]` 的写法都不用改，cos / sin `[S, D]` 靠[广播](/handson/torch-primitives#broadcasting)对上。
- "前后两半配对"是 GPT-NeoX / HF LLaMA 的布局；原始 RoPE 论文和 GPT-J 是相邻两维 `(x0, x1), (x2, x3)…` 配对，权重布局不同，混用会静默出错。
- decode 时 RoPE 的位置要带上 KV cache 的偏移，对应 [Fused QKV Projection with RoPE and KV Cache Update](./fused-qkv-projection-with-rope-and-kv-cache-update)。
