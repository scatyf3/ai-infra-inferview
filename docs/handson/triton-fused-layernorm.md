---
title: Triton 版 Fused LayerNorm
status: todo
tags: [triton, layernorm, handson]
difficulty: 3
order: 8
related: []
stack: [k-fused, k-lang]
leetgpu: [113]
---

# Triton 版 Fused LayerNorm

> forward 与 backward

## 一句话结论

LayerNorm 对每行算均值和方差再做 (x − μ) / sqrt(σ² + ε) · γ + β；fused 的意思是一个 kernel 里一次读 x、算两个统计量、写 y，不把 μ、σ² 或中间结果落到 HBM。Triton 下一行一个 program，用 `tl.sum` 做行内 reduce，整个 kernel 二十行，是 memory-bound 融合算子的标准练习。

## 推导

- **两个 reduce**：μ = sum(x)/N，σ² = sum((x − μ)²)/N；一遍算法（sum 和 sum of squares）数值不稳，Welford 或两遍更稳，行能放进一个 block 时两遍只是寄存器里多算一次。
- **和 RMSNorm 的差别**：多一个 μ 和 β，其他结构一样，见 [RMSNorm](/handson/rmsnorm)。
- **fp32 累加**：bf16 输入先 cast 到 fp32 做统计，输出再 cast 回去。
- **反向**：∂x 需要 γ·∂y 的行内均值和与 x̂ 的加权均值，Triton 教程里的 backward 还要对 ∂γ、∂β 做跨行归约，用 atomic 或分段 + 二次归约。

## 面试追问

::: details Q：hidden 维超过一个 block 能放下的大小怎么办？
BLOCK_SIZE 受寄存器限制，更大的行要分成多个 chunk 循环：第一遍循环累加 sum 和 sum of squares（或 Welford 合并），第二遍循环按统计量归一化写回。代价是 x 读两遍，但仍比非融合版本少写一次中间结果。
:::

## 手撕

对应 [LeetGPU #113](https://leetgpu.com/challenges)。框架：

```python
@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.)
    var = tl.sum(xc * xc, axis=0) / N
    y = xc / tl.sqrt(var + eps)
    # 乘 w 加 b，store
```
