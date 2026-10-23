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

>TLDR：利用unsafe版本的无依赖，所以一个直觉告诉我们safe版本一定也有一趟扫完的方法，利用一个中间变量T，从unsafe版本的递推推出safe版本的递推

**定义**：对前 $k$ 个元素，记

$$  
m_k = \max_{i \le k} x_i, \qquad  
T_k = \sum_{i \le k} e^{x_i}, \qquad  
s_k = \sum_{i \le k} e^{x_i - m_k}  
$$

其中 $T_k$ 是 unsafe 的部分和，可能溢出，只在推导中使用。由 $e^{x_i - m_k} = e^{x_i} / e^{m_k}$ 可得

$$s_k = \frac{T_k}{e^{m_k}}, \qquad \text{即}\quad T_k = s_k \cdot e^{m_k}$$

**更新规则**：来了 $x_{k+1}$ 之后，

$$  
m_{k+1} = \max(m_k,\ x_{k+1}), \qquad  
s_{k+1} = s_k \cdot e^{m_k - m_{k+1}} + e^{x_{k+1} - m_{k+1}}  
$$

初值取 $m_0 = -\infty$、$s_0 = 0$。

**Correctness 证明**：$T$ 的更新就是普通的累加：

$$T_{k+1} = T_k + e^{x_{k+1}}$$

两边同时除以 $e^{m_{k+1}}$：

$$s_{k+1} = \frac{T_{k+1}}{e^{m_{k+1}}} = \frac{T_k}{e^{m_{k+1}}} + e^{x_{k+1} - m_{k+1}}$$

再代入 $T_k = s_k \cdot e^{m_k}$：

$$s_{k+1} = s_k \cdot e^{m_k - m_{k+1}} + e^{x_{k+1} - m_{k+1}} \qquad \blacksquare$$

**数值安全**：由于 $m_{k+1} \ge m_k$ 且 $m_{k+1} \ge x_{k+1}$，两个指数都 $\le 0$，exp 的结果落在 $(0, 1]$ 内，不会上溢。


## tile-wise的解释

当然，这个递推被我们扩展到tile-wise，tile内部做树形规约求tile内max，然后基于max求sum，每个tile内调用两轮规约kernel，tile外用推广的方法merge，即优化后，tile内还有依赖，只是tile间online合并，允许把reduce的维度切分给多个tile，在对attention dim=-1，即`seqlen_k`做更大粒度并行。

如果没有online softmax，我们在max和sum之间要全局规约，有online softmax之后可以免于全局规约，直接通过per tile的online update，从而免于全局规约，也允许了后续的fuse。

tile-wise online update的算法如下：
1. 每个tile各自max - sum
2. tile之间串行update自己的max和sum

note：
1. 也可以不串行，选择串行是因为其他维度的并行度已经足够，而串行能用最小的片上状态，并且方便和访存做流水
	- 片上只需要一份 $(m, l, O)$。树形合并需要同时保存多份 $O$，寄存器放不下。
	- 可以和 $K_j, V_j$ 的加载形成流水线。
	- 并行度已经由 $S_q$、batch 和 head 提供，KV 方向不需要再并行
2. splitk就是非串行reduce，有个reduce kernel


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
