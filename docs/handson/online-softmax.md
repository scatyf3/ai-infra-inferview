---
title: Online Softmax
status: draft
tags: [softmax, numerics, handson, flash-attention]
difficulty: 3
order: 3.5
related: [/handson/stable-softmax, /inference/flash-attention, /handson/triton-softmax, /handson/naive-attention]
stack: [k-attn]
leetgpu: [5, 6]
---

# Online Softmax

> 一遍扫描同时求 max 和分母；两段结果怎么合并；带 $V$ 的输出为什么也要 rescale

## 一句话结论

[数值稳定 softmax](./stable-softmax) 要先扫一遍求 $m = \max_j x_j$，再扫一遍求 $s = \sum_j e^{x_j - m}$。online softmax 边扫边维护 $(m, s)$：max 变大时，把旧的 $s$ 乘上 $e^{m_{\text{old}} - m_{\text{new}}}$ 换算到新 max 下。两段各自的 $(m, s)$ 也能用同一个式子合并，所以可以分块、可以并行归约。FlashAttention、FlashDecoding、Ring Attention 用的都是这一个合并式。

## 推导

### 出发点：safe softmax 要读三遍

```
m = max(x)                  # 第一遍：求最大值
s = sum(exp(x_i - m))       # 第二遍：求和
y_i = exp(x_i - m) / s      # 第三遍：归一化
```

第二遍必须等第一遍结束，因为不知道 $m$ 就没法算 $e^{x_i - m}$。对 attention 来说，$x$ 是一行 score，长度是序列长度 $S$。读三遍就是三倍的 HBM 流量，而且整行 score 必须先算出来存着。

### 逐元素递推

符号：$x_1, \dots, x_n$ 是输入，每个 $x_i$ 是一行 score 中的单个元素（分块版本见下文「两段合并」）；处理完前 $k$ 个元素后，

$$
m_k = \max_{i \le k} x_i, \qquad s_k = \sum_{i \le k} e^{x_i - m_k}
$$

$s_k$ 是「以当前最大值 $m_k$ 为基准」的部分和。来了 $x_{k+1}$：

$$
m_{k+1} = \max(m_k,\ x_{k+1}), \qquad
s_{k+1} = s_k \cdot e^{m_k - m_{k+1}} + e^{x_{k+1} - m_{k+1}}
$$

**为什么对。** 按定义，$s_{k+1}$ 的每一项都是 $e^{x_i - m_{k+1}}$（减新 max），而手上 $s_k$ 的每一项是 $e^{x_i - m_k}$（减旧 max）。要做的就是把前 $k$ 项的基准从 $m_k$ 换成 $m_{k+1}$。

先看单独一项。乘上修正因子，用 $e^a \cdot e^b = e^{a+b}$：

$$
e^{x_i - m_k} \cdot e^{m_k - m_{k+1}}
= e^{(x_i - m_k) + (m_k - m_{k+1})}
= e^{x_i - m_{k+1}}
$$

指数里 $-m_k$ 和 $+m_k$ 抵消，剩下的就是新基准下的那一项。修正因子就是专门挑来消掉旧 max 的。

再看整个和。每一项乘的是同一个数，按分配律等于整个和乘一次：

$$
s_k \cdot e^{m_k - m_{k+1}} = \sum_{i \le k} e^{x_i - m_k} \cdot e^{m_k - m_{k+1}} = \sum_{i \le k} e^{x_i - m_{k+1}}
$$

所以不用回头找每个 $x_i$，拿手上的 $s_k$ 乘一下就行，online 成立靠的就是这一点。这样就得到了前 $k$ 项在新基准下的和，再加上第 $k+1$ 项 $e^{x_{k+1} - m_{k+1}}$，就是 $s_{k+1}$。下面数字例子的第 2 步就是这个过程：$s_1 = e^{1-1} = 1$，乘 $e^{1-3}$ 变成 $e^{1-3}$，正好是 $x_1$ 在新 max $3$ 下的那一项。

可以把 $s_k$ 看成「以 $m_k$ 为单位记的账」：max 变了，单位也变了，旧账要乘一个汇率 $e^{m_k - m_{k+1}}$ 才能和新的一项相加。

初值 $m_0 = -\infty$、$s_0 = 0$：第一步修正因子 $e^{-\infty} = 0$，旧的和直接清零。

数值上也安全：$m_{k+1} \ge m_k$，$m_{k+1} \ge x_{k+1}$，两个指数都 $\le 0$，`exp` 的结果在 $(0, 1]$，不会上溢。

**数字例子**，$x = [1, 3, 2]$：

| 步 | $x_k$ | $m_k$ | 修正因子 $e^{m_{k-1} - m_k}$ | $s_k$ |
|---|---|---|---|---|
| 1 | 1 | 1 | $e^{-\infty} = 0$ | $0 + e^{0} = 1$ |
| 2 | 3 | 3 | $e^{1-3} = 0.1353$ | $1 \times 0.1353 + e^{0} = 1.1353$ |
| 3 | 2 | 3 | $e^{0} = 1$ | $1.1353 + e^{-1} = 1.5032$ |

直接算：$e^{1-3} + e^{3-3} + e^{2-3} = 0.1353 + 1 + 0.3679 = 1.5032$，对上。第 3 步 max 没变，修正因子是 1，退化成普通累加。

扫完得到 $(m_n, s_n)$，再扫一遍算 $y_i = e^{x_i - m_n} / s_n$。所以求 softmax 本身是**两遍**（比三遍少一遍），不是一遍。

### 两段合并

把 $x$ 切成两段 $A$、$B$，各自扫完得到 $(m_A, s_A)$、$(m_B, s_B)$。合并：

$$
m = \max(m_A, m_B), \qquad
s = s_A \cdot e^{m_A - m} + s_B \cdot e^{m_B - m}
$$

理由和上面一样：两边各自换算到共同基准 $m$ 再相加。逐元素递推只是 $B$ 只有一个元素的特例（$m_B = x_{k+1}$，$s_B = 1$）。

这个合并满足结合律和交换律，所以：

- 可以按任意块大小分块，一块一块顺序合并（FlashAttention 沿 K/V 方向）；
- 可以做树形归约，比如 warp 内用 shuffle 两两合并（[reduce](./reduce) 里的套路，把加法换成这个合并）；
- 可以先切开并行算，最后一次合并（FlashDecoding 的 split-KV，Ring Attention 每卡持一段 KV）。

### 带上 $V$：attention 只要一遍

attention 要的不是 $y$ 本身，而是 $o = \sum_i y_i v_i$（$v_i$ 是 $V$ 的第 $i$ 行，长度 $d$）。再维护一个未归一化的输出

$$
\text{acc}_k = \sum_{i \le k} e^{x_i - m_k}\, v_i
$$

它和 $s_k$ 用的是同一个基准 $m_k$，所以换基准时乘同一个修正因子。按块写，第 $j$ 块 score 是 $x^{(j)}$、对应的 V 行是 $V^{(j)}$：

$$
m' = \max\big(m, \max x^{(j)}\big),\quad
\alpha = e^{m - m'},\quad
s' = \alpha\, s + \sum e^{x^{(j)} - m'},\quad
\text{acc}' = \alpha\, \text{acc} + e^{x^{(j)} - m'}\, V^{(j)}
$$

全部块处理完，$o = \text{acc} / s$。这里只扫了一遍：每块的 $e^{x^{(j)} - m'}$ 算完立刻乘 $V^{(j)}$ 累进 $\text{acc}$，然后就扔掉，不需要回头。除以 $s$ 留到最后做一次。

**数字例子**接上面，$d = 1$，$v = [10, 20, 30]$，一块一个元素：

| 步 | $\alpha$ | $\text{acc}_k$ | $s_k$ |
|---|---|---|---|
| 1 | 0 | $e^{0} \times 10 = 10$ | 1 |
| 2 | 0.1353 | $10 \times 0.1353 + e^{0} \times 20 = 21.353$ | 1.1353 |
| 3 | 1 | $21.353 + e^{-1} \times 30 = 32.389$ | 1.5032 |

$o = 32.389 / 1.5032 = 21.55$。直接算：$(0.1353 \times 10 + 1 \times 20 + 0.3679 \times 30) / 1.5032 = 21.55$，对上。

如果第 2 步忘了给 $\text{acc}$ 乘 $\alpha$，$\text{acc}_3 = 10 + 20 + 11.04 = 41.04$，$o = 27.30$：第一个元素的权重被放大了 $e^{2} \approx 7.4$ 倍，结果错了但不报错。

### 在 FlashAttention 里是什么样

一行 score 是 $Q$ 的一行和所有 $K$ 行的点积。Q 块常驻 SRAM，K/V 一块一块流过，每块只产生 $B_r \times B_c$ 的 score tile，用上面的块递推更新 $(m, s, \text{acc})$ 后就丢掉，$S \times S$ 的 score 矩阵从不写回 HBM。FLOPs 没少（每块还多了 rescale），省的是 HBM 读写。访存账和 v1/v2/v3 的区别见 [FlashAttention](/inference/flash-attention)。

反向时也不用存 score：前向只额外存每行的 $\text{LSE} = m + \log s$，反向重算 score 后 $P = e^{x - \text{LSE}}$ 一步就恢复出概率。

## 手撕

::: code-group

```python [逐元素：两遍求 softmax]
import numpy as np

def online_softmax(x):
    """第一遍同时求 max 和 sum，第二遍归一化。"""
    m, s = -np.inf, 0.0
    for xi in x:
        m_new = max(m, xi)
        s = s * np.exp(m - m_new) + np.exp(xi - m_new)   # 旧的和换算到新 max
        m = m_new
    return np.exp(np.asarray(x) - m) / s
```

```python [两段合并]
def merge(a, b):
    """a、b 是两段各自的 (m, s)，返回整段的 (m, s)。满足结合律，可以树形归约。"""
    (m_a, s_a), (m_b, s_b) = a, b
    m = max(m_a, m_b)
    return m, s_a * np.exp(m_a - m) + s_b * np.exp(m_b - m)
```

```python [分块 softmax @ V（FlashAttention 的形式）]
def block_softmax_weighted_sum(x, v, block=64):
    """x: [n] 一行 score，v: [n, d]。一遍扫描，等价于 softmax(x) @ v。"""
    m, s, acc = -np.inf, 0.0, np.zeros(v.shape[1])
    for j in range(0, len(x), block):
        xb, vb = x[j:j + block], v[j:j + block]
        m_new = max(m, xb.max())
        alpha = np.exp(m - m_new)          # 旧结果的修正因子
        p = np.exp(xb - m_new)             # 本块的未归一化权重
        s = s * alpha + p.sum()
        acc = acc * alpha + p @ vb         # acc 和 s 乘同一个 alpha
        m = m_new
    return acc / s                         # 最后才归一化一次
```

```python [验证]
rng = np.random.default_rng(0)
x = rng.normal(0, 10, 1000)
v = rng.normal(size=(1000, 64))
ref = np.exp(x - x.max()) / np.exp(x - x.max()).sum()

assert np.allclose(online_softmax(x), ref)
m, s = -np.inf, 0.0
for blk in np.split(x, 10):                # 切 10 段，各自求 (m, s) 再合并
    m, s = merge((m, s), (blk.max(), np.exp(blk - blk.max()).sum()))
assert np.isclose(s, np.exp(x - x.max()).sum())
assert np.allclose(block_softmax_weighted_sum(x, v, block=37), ref @ v)
```

:::

### 常见错误

1. **忘了给 $\text{acc}$ rescale**：只更新 $s$，上面的例子里 21.55 变成 27.30，不报错，很难查。
2. **每块都除以 $s$**：v1 的写法，结果对，但每块多一次除法；v2 改成最后只除一次。
3. **整行被 mask 掉**：causal 或 padding 让某行所有 score 都是 $-\infty$，$m = -\infty$，$e^{m - m'} = e^{-\infty - (-\infty)} = \text{nan}$。kernel 里要特判，比如把 $m'$ 为 $-\infty$ 时的 $\alpha$ 置 0、最后 $s = 0$ 时输出 0。
4. **累加器用 fp16**：$s$ 和 $\text{acc}$ 要累加几千上万项，用 fp32。

## 面试追问

::: details Q：online softmax 相比 safe softmax 省了什么，代价是什么？
单独求 softmax：省一遍对 $x$ 的读取（三遍变两遍）。融进 attention：一遍搞定，score 不用存。代价是每步多一次 `exp` 和两次乘法（$s$、$\text{acc}$ 各乘 $\alpha$），用少量算力换访存。attention 在长序列下是 memory-bound，这个交换很划算。
:::

::: details Q：为什么 $\text{acc}$ 也要乘修正因子？
$\text{acc}$ 里每一项的权重是 $e^{x_i - m}$，和 $s$ 用同一个基准 $m$。max 变了，权重的基准就变了，$\text{acc}$ 必须和 $s$ 一起换算，最后 $\text{acc}/s$ 里的基准才能约掉。
:::

::: details Q：FlashDecoding 把 KV 切成多段并行算，最后怎么合并？
每段算出 $(m_j, \ell_j, o_j)$，其中 $o_j$ 已经除过本段的 $\ell_j$。合并：$m = \max_j m_j$，$o = \dfrac{\sum_j e^{m_j - m} \ell_j o_j}{\sum_j e^{m_j - m} \ell_j}$。就是两段合并式推广到多段，$\ell_j o_j$ 还原成未归一化的 $\text{acc}_j$。
:::

::: details Q：online 版和一次性 softmax 结果完全一样吗？
数学上一样，浮点上有小差异：累加顺序不同，浮点加法不满足结合律。bf16 输入下通常在 1e-3 量级，所以开关 FlashAttention 生成结果可能不逐位一致。
:::

## 参考

- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)（Milakov & Gimelshein, 2018）
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
