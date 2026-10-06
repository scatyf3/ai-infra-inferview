---
title: Attention with Sinks
status: draft
tags: [attention, mask, sliding-window, leetgpu]
difficulty: 2
order: 4.6
related: [/leetgpu/sliding-window-self-attention, /leetgpu/mask, /inference/attention-variants]
stack: []
leetgpu: [112]
---

# Attention with Sinks

> [LeetGPU #112](https://leetgpu.com/challenges/attention-with-sinks) · 单头，因果 sliding window + 前 `num_sinks` 个 token 永远可见 · mask 的拼法见 [Attention mask 怎么拼](./mask)

query $i$ 能看 key $j$，当且仅当

$$
j \le i \quad \text{且} \quad \big(\ j < \texttt{num\_sinks} \ \ \text{或} \ \ i - j < \texttt{window\_size}\ \big)
$$

`num_sinks = 2, window_size = 3, M = 8` 时（■ 可见）：

```
     j: 0 1 2 3 4 5 6 7
i=0     ■ · · · · · · ·
i=1     ■ ■ · · · · · ·
i=2     ■ ■ ■ · · · · ·
i=3     ■ ■ ■ ■ · · · ·
i=4     ■ ■ ■ ■ ■ · · ·
i=5     ■ ■ · ■ ■ ■ · ·
i=6     ■ ■ · · ■ ■ ■ ·
i=7     ■ ■ · · · ■ ■ ■
```

## 题解

```python
import torch
import math


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d: int,
    num_sinks: int,
    window_size: int,
):
    attn = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d)    # [M, M]

    # 写「能看哪些」，最后一次取反
    i = torch.arange(M, device=Q.device)[:, None]                  # [M, 1] query
    j = torch.arange(M, device=Q.device)[None, :]                  # [1, M] key
    causal = j <= i                                                # 不看未来
    sink = j < num_sinks                                           # 前 num_sinks 列
    window = (i - j) < window_size                                 # 最近 window_size 个（含自己）
    allowed = causal & (sink | window)                             # [M, M] bool

    attn = attn.masked_fill(~allowed, float('-inf'))
    attn = torch.softmax(attn, dim=-1)                             # 沿 key 方向
    torch.matmul(attn, V, out=output)                              # [M, d]
```

和朴素逐 query 实现对拍过（200 组随机 `M / d / num_sinks / window_size`）。

## 怎么拼出来的

拆成两块，各自先和 `causal` 取交集，再取并集：

- **斜带** `causal & window`：$0 \le i - j < W$
- **sink 列** `causal & sink`：$j < S$ 且 $j \le i$

两块在前几行有重叠，bool 的 `|` 不会重复计算。

用 `tril` / `triu` 也能拼，但必须在 **bool** 矩阵上做：

```python
ones = torch.ones(M, M, dtype=torch.bool, device=Q.device)
causal = torch.tril(ones)                                         # j <= i
band = causal & torch.triu(ones, diagonal=-(window_size - 1))     # i - j <= W - 1
sinks = causal.clone()
sinks[:, num_sinks:] = False                                      # 只留前 S 列
allowed = band | sinks
```

## 为什么要 sink

StreamingLLM 观察到：attention 会把大量权重「倒」在最开头几个 token 上，不管它们内容是什么。softmax 的权重必须加起来等于 1，当前 query 没有特别想看的位置时，多出来的权重总得有个去处。纯 sliding window 把开头 token 滑出窗口后，这部分权重没地方放，分布被打乱，长文本困惑度会崩。保留前几个 token 常驻 KV cache（通常 4 个就够），就能用 $O(S + W)$ 的显存稳定地处理无限长输入。

## 这题的坑

1. **用 `-inf` 相加拼 sink 和 window**：加法 mask 只要有一个挡住就是 `-inf`，相当于取「允许」的**交集**；这题要并集，得用 bool 的 `|`。
2. **注释里写成 `row + col < S + W`**：窗口看的是离对角线多远，条件是 $i - j$，不是 $i + j$。
3. **window 没和 causal 取交集**：`i - j < W` 对 $j > i$ 也成立（$i - j$ 是负数），不加 `causal` 就看到了未来。
4. **window 的含义和 [#59](./sliding-window-self-attention) 不一样**：#59 是双向、单侧宽度（$\lvert i - j\rvert \le W$）；这题是因果、总宽度含自己（$i - j < W$）。
