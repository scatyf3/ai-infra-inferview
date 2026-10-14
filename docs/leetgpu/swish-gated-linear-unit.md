---
title: Swish-Gated Linear Unit
status: draft
tags: [swiglu, ffn, leetgpu]
difficulty: 1
order: 6.7
related: [/leetgpu/swiglu-mlp-block]
stack: []
leetgpu: [54]
---

# Swish-Gated Linear Unit

> [LeetGPU #54](https://leetgpu.com/challenges/swish-gated-linear-unit) · 1D 输入 `[N]`，前半是 gate、后半是 up · torch · 完整的 FFN 见 [SwiGLU MLP Block](./swiglu-mlp-block)

$$
x = [\,x_1 \mid x_2\,],\quad x_1, x_2 \in \mathbb{R}^{N/2}, \qquad
y = \text{SiLU}(x_1) \odot x_2, \qquad \text{SiLU}(z) = z \cdot \sigma(z)
$$

SwiGLU 出自 Shazeer 2020（arXiv 2002.05202），Swish 取 $\beta = 1$ 就是 SiLU。

## 题解

```python
# input = [gate | up]，长度 N；* 是逐元素乘
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    half = N // 2
    x1 = input[:half]                 # gate
    x2 = input[half:]                 # up
    x1 = x1 * torch.sigmoid(x1)       # SiLU，等价于 torch.nn.functional.silu(x1)
    output.copy_(x1 * x2)             # output 长度是 N/2
```

## 这题的坑

1. **`half = N / 2`**：得到 float，切片报 "slice indices must be integers"。用 `//`。
2. **gate 和 up 拿反**：SiLU 作用在**前半**上，`x2 * sigmoid(x2) * x1` 不报错但结果错。
3. **写 output**：`output.copy_(...)`，不要 `return`，见 [通用语法坑](./#通用语法坑)。

## 和推理引擎的对应

这题的输入布局 `[gate | up]` 就是推理引擎里 gate、up 合并成一个 GEMM 之后的输出：vLLM 把 `W_gate`、`W_up` 拼成一个 `[d_model, 2·d_ffn]` 的权重（`MergedColumnParallelLinear`），一次 matmul 出 `[M, 2·d_ffn]`，再交给 `SiluAndMul` 这个 kernel 做本题的事（按最后一维切半）。

所以 triton 版就是一个逐元素 kernel：每个 program 读 gate 一段、up 对应一段，算完写回，HBM 读 N 个、写 N/2 个元素。
