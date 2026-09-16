---
title: 数值稳定 Softmax
status: draft
tags: [softmax, numerics, handson]
difficulty: 2
order: 3
related: [/handson/triton-softmax, /inference/flash-attention, /handson/mha-gqa-forward]
---

# 数值稳定 Softmax

> max-subtraction、online softmax

## 一句话结论

减去最大值：$\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$，其中 $m = \max_j x_j$。数学上等价（分子分母同乘 $e^{-m}$），数值上把 `exp` 的输入压到 $\le 0$，彻底避免上溢。**online softmax** 再进一步，用一次遍历同时维护最大值和求和，这是 FlashAttention 能分块计算的基础。

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

三次读取 $x$。对 attention 来说，$x$ 是 $S \times S$ 的 score，读三遍意味着三倍的 HBM 流量。

### Online softmax（两遍法）

关键观察：如果已经处理了前 $k$ 个元素得到 $(m_k, s_k)$，来了新元素 $x_{k+1}$，新的最大值 $m_{k+1} = \max(m_k, x_{k+1})$，而旧的和可以**修正**：

$$
s_{k+1} = s_k \cdot e^{m_k - m_{k+1}} + e^{x_{k+1} - m_{k+1}}
$$

修正因子 $e^{m_k - m_{k+1}}$ 把之前用旧 max 算的和"换算"成新 max 下的和。这样一遍遍历就同时得到 $m$ 和 $s$，只需再一遍做归一化。

推广到分块（这就是 FlashAttention）：每块算出局部的 $(m^{(j)}, s^{(j)}, O^{(j)})$，合并时

$$
m^{new} = \max(m^{old}, m^{(j)}), \quad
O^{new} = O^{old} e^{m^{old} - m^{new}} + O^{(j)} e^{m^{(j)} - m^{new}}
$$

于是 $S \times S$ 的 score 矩阵**从不需要完整物化**，HBM 访问从 $O(S^2)$ 降到 $O(S^2 / M)$（$M$ 是 SRAM 能放下的块大小）。FlashAttention 省的正是这个，FLOPs 一点没省。

## 手撕

::: code-group

```python [numpy 三遍法]
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

```python [online softmax 一遍]
def online_softmax(x):
    """一遍扫描同时得到 max 和 sum，FlashAttention 的核心递推。"""
    m, s = -float("inf"), 0.0
    for xi in x:
        m_new = max(m, xi)
        s = s * np.exp(m - m_new) + np.exp(xi - m_new)   # 用修正因子换算旧的和
        m = m_new
    return np.exp(np.asarray(x) - m) / s
```

```python [分块合并（FlashAttention 的形式）]
def block_softmax_weighted_sum(scores_blocks, v_blocks):
    """每块算局部 softmax@V，再用 rescale 合并，等价于全局 softmax@V。"""
    m, s, o = -float("inf"), 0.0, None
    for sc, v in zip(scores_blocks, v_blocks):
        m_blk = sc.max()
        m_new = max(m, m_blk)
        alpha = np.exp(m - m_new)          # 旧结果的修正因子
        p = np.exp(sc - m_new)             # 本块的未归一化权重
        s = s * alpha + p.sum()
        o = (p @ v) if o is None else o * alpha + p @ v
        m = m_new
    return o / s                            # 最后才做一次归一化
```

:::

### 常见错误

1. **`keepdims=False`** 导致广播 shape 错，减 max 时静默算错。
2. **先 softmax 再 log** 算交叉熵：概率下溢成 0 时 `log(0) = -inf`。用 `log_softmax`。
3. **在 fp16 下累加**：即使减了 max，$\sum e^{x_i - m}$ 在 $S$ 很大时（几万项）累加误差也可观。累加器用 fp32。
4. **online 版忘了对 $o$ 也 rescale**：只 rescale $s$ 不 rescale $o$，结果错但不报错，很难查。

## 面试追问

::: details Q：减最大值会引入误差吗？
数学上完全等价，浮点上反而更精确。因为 $e^{x-m} \in (0, 1]$，落在浮点表示最密集的区间；而不减 max 的话 $e^x$ 可能在 $10^{30}$ 量级，相对精度差得多。所以是纯赚。
:::

::: details Q：online softmax 相比三遍法省了什么，代价是什么？
省一次对 score 的完整读取。代价是每步多了一次 `exp` 和一次乘法（rescale），也就是用少量算力换访存。在 memory-bound 的 attention 上这个交换非常划算，这正是 FlashAttention 的设计哲学。
:::

::: details Q：为什么 FlashAttention 省的是访存不是 FLOPs？
它算的还是同样的 $QK^\top$ 和 $PV$，矩阵乘的 FLOPs 一个没少，rescale 还多了一点。省的是不把 $S \times S$ 的中间矩阵写回 HBM 再读回来。标准实现要写一次读两次（softmax 的两遍 + PV 的一遍），$S = 8192$、$H = 64$、bf16 下一层就是几十 GiB 的往返流量。
:::

::: details Q：softmax 的反向怎么写？
$\frac{\partial L}{\partial x_i} = y_i \left( \frac{\partial L}{\partial y_i} - \sum_j \frac{\partial L}{\partial y_j} y_j \right)$，其中 $y$ 是前向输出。只需要保存 $y$，不需要保存 $x$，这也是 FlashAttention 反向只存 $(m, s)$ 两个标量就能重算的原因。
:::

## 参考

- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
