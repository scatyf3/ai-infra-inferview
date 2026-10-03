---
title: RMSNorm
status: todo
tags: [rmsnorm, handson]
difficulty: 2
order: 4
related: []
stack: [k-fused]
leetgpu: [50, 83]
---

# RMSNorm

> 与 LayerNorm 的区别、fused 实现

## 一句话结论

RMSNorm 去掉了 LayerNorm 的减均值，只用均方根归一化再乘可学习的 γ：y = x / sqrt(mean(x²) + ε) · γ。少一次 reduce、不需要 β，效果和 LayerNorm 相当，所以 LLaMA 之后的模型全用它。fused 实现的意义在于它是 memory-bound 的：一个 kernel 里读一遍 x、算 rms、写一遍 y，别让 x² 和 mean 落到 HBM。

## 推导

- **和 LayerNorm 的区别**：LayerNorm 做 (x − μ) / σ · γ + β，两次 reduce（均值、方差）；RMSNorm 一次 reduce，假设均值本来就接近零。
- **数值**：用 fp32 累加 x²，输入 bf16 时先 cast；ε 放在 sqrt 里面。
- **fused**：一个 program 处理一行（hidden 维通常 4k–16k，一个 block 装得下），和残差加（先 `x = x + residual` 再 norm）融合是推理框架的标配，省一次读写。
- **反向**：∂y/∂x 涉及 rms 和 x·γ·y 的行内 reduce，训练框架里也常 fuse。

## 面试追问

::: details Q：为什么推理框架把残差加和 RMSNorm 融合成一个 kernel？
两者都是 memory-bound 的逐元素 / 行操作，分开写要把残差和的结果写到 HBM 再读回来做 norm，白白多一次读写。融合后读 x 和 residual 各一次、写新的 residual 和 norm 结果各一次，流量减少约三分之一，decode 下每层都省这一点，累积起来可观。
:::

## 手撕

对应 [LeetGPU #50、#83](https://leetgpu.com/challenges)。Triton 框架（一行一个 program）：

```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=1.)
    tl.store(y_ptr + row * N + cols, x / rms * w, mask=mask)
```
