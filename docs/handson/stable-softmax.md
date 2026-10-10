---
title: 数值稳定 Softmax
status: draft
tags: [softmax, numerics, handson]
difficulty: 2
order: 3
related: [/handson/online-softmax, /handson/triton-softmax, /inference/flash-attention, /handson/mha-gqa-forward]
stack: [k-attn]
leetgpu: [5]
---

# 数值稳定 Softmax

> max-subtraction、log-softmax、反向

## 一句话结论

减去最大值：$\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$，其中 $m = \max_j x_j$。数学上等价（分子分母同乘 $e^{-m}$），数值上把 `exp` 的输入压到 $\le 0$，彻底避免上溢。代价是要扫三遍 $x$；把前两遍合成一遍的 [online softmax](./online-softmax) 单独成篇，它是 FlashAttention 能分块计算的基础。

## 推导

### 为什么会炸

fp16 的最大值是 65504，$e^{11.09} = 65536$ 就溢出了。attention 的 score 经过 $\sqrt{d_h}$ 缩放后典型范围是 ±10，极端情况（长序列、某些 head）能到 ±30，直接 `exp` 必然 inf，之后 `inf / inf = nan`，整个前向污染。

下溢方向也有问题：$e^{-30}$ 在 fp16 下是 0，所有项都下溢时分母为 0。减最大值保证了至少有一项是 $e^0 = 1$，分母不可能为 0。

### 三遍法（safe softmax）

```
m = max(x)                  # 第一遍：求最大值
s = sum(exp(x_i - m))       # 第二遍：求和
y_i = exp(x_i - m) / s      # 第三遍：归一化
```

三次读取 $x$。对 attention 来说，$x$ 是 $S \times S$ 的 score，读三遍意味着三倍的 HBM 流量。第二遍要等第一遍的 $m$，怎么把这两遍合成一遍见 [Online Softmax](./online-softmax)。

## 手撕

```python
import numpy as np

def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)                      # 所有指数 <= 0，不会上溢
    return e / np.sum(e, axis=axis, keepdims=True)

def log_softmax(x, axis=-1):
    """算 loss 时用这个，别先 softmax 再 log（log(0) = -inf）"""
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))
```

### 常见错误

1. **`keepdims=False`** 导致广播 shape 错，减 max 时静默算错。
2. **先 softmax 再 log** 算交叉熵：概率下溢成 0 时 `log(0) = -inf`。用 `log_softmax`。
3. **在 fp16 下累加**：即使减了 max，$\sum e^{x_i - m}$ 在 $S$ 很大时（几万项）累加误差也可观。累加器用 fp32。

## 面试追问

::: details Q：减最大值会引入误差吗？
数学上完全等价，浮点上反而更精确。因为 $e^{x-m} \in (0, 1]$，落在浮点表示最密集的区间；而不减 max 的话 $e^x$ 可能在 $10^{30}$ 量级，相对精度差得多。所以是纯赚。
:::

::: details Q：为什么 FlashAttention 省的是访存不是 FLOPs？
它算的还是同样的 $QK^\top$ 和 $PV$，矩阵乘的 FLOPs 一个没少，rescale 还多了一点。省的是不把 $S \times S$ 的中间矩阵写回 HBM 再读回来。标准实现要写一次读两次（softmax 的两遍 + PV 的一遍），$S = 8192$、$H = 64$、bf16 下一层就是几十 GiB 的往返流量。
:::

::: details Q：softmax 的反向怎么写？
$\frac{\partial L}{\partial x_i} = y_i \left( \frac{\partial L}{\partial y_i} - \sum_j \frac{\partial L}{\partial y_j} y_j \right)$，其中 $y$ 是前向输出。只需要保存 $y$，不需要保存 $x$。FlashAttention 连 $y$ 都不存，只存每行一个 $\text{LSE} = m + \log s$，反向时重算 score 再用 $y = e^{x - \text{LSE}}$ 恢复（见 [Online Softmax](./online-softmax)）。
:::

## 参考

- [Online Softmax](./online-softmax)
