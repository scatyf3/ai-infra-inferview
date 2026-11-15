---
title: Token Embedding Layer
status: draft
tags: [embedding, layernorm, leetgpu]
difficulty: 1
order: 6.9
related: [/handson/triton-fused-layernorm]
stack: []
leetgpu: [106]
---

# Token Embedding Layer

> [LeetGPU #106](https://leetgpu.com/challenges/token-embedding-layer) · `token_ids [B, T]`，`position_ids [T]`，词表 `[V, D]`，位置表 `[P, D]`，γ / β `[D]` · torch · fused 版见 [Triton fused LayerNorm](/handson/triton-fused-layernorm)

$$
s_{b,t} = E_{\text{tok}}[\text{token\_ids}_{b,t}] + E_{\text{pos}}[\text{position\_ids}_t] \in \mathbb{R}^D
$$

$$
\mu = \frac{1}{D}\sum_{d=1}^{D} s_d, \qquad
\sigma^2 = \frac{1}{D}\sum_{d=1}^{D} (s_d - \mu)^2, \qquad
y_d = \gamma_d \frac{s_d - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_d
$$

$\mu$、$\sigma^2$ 每个 token 各算一个，沿 $D$ 维 reduce；$\sigma^2$ 除以 $D$ 不是 $D-1$（有偏方差）。

## 思路

1. 查表：`token_embeddings[token_ids]` 得 `[B, T, D]`，`position_embeddings[position_ids]` 得 `[T, D]`，相加时自动广播到 `[B, T, D]`。
2. LayerNorm：沿最后一维求均值、方差，`keepdim=True` 保留成 `[B, T, 1]`，后面才能和 `[B, T, D]` 广播。
3. γ / β 是 `[D]`，逐通道，自动广播到最后一维。

例：某个 token 的 $s = [1, 2, 3, 6]$，$D = 4$。$\mu = 3$，减均值得 $[-2, -1, 0, 3]$，平方 $[4, 1, 0, 9]$，$\sigma^2 = 14/4 = 3.5$，$\sigma \approx 1.87$。

## 题解

```python
import torch


def solve(
    token_ids: torch.Tensor,            # (B, T)
    position_ids: torch.Tensor,         # (T)
    token_embeddings: torch.Tensor,     # (V, D)
    position_embeddings: torch.Tensor,  # (P, D)
    gamma: torch.Tensor,                # (D)
    beta: torch.Tensor,                 # (D)
    output: torch.Tensor,
    B: int, T: int, V: int, P: int, D: int,
    eps: float,
):
    pos_embedding = position_embeddings[position_ids]   # [T, D]
    tok_embedding = token_embeddings[token_ids]         # [B, T, D]
    s = pos_embedding + tok_embedding                   # 广播成 [B, T, D]
    mean = s.mean(dim=-1, keepdim=True)                 # [B, T, 1]
    var = ((s - mean) ** 2).mean(dim=-1, keepdim=True)  # [B, T, 1]，有偏
    y = gamma * (s - mean) / torch.sqrt(var + eps) + beta
    output.copy_(y)
```

对拍：`torch.nn.functional.layer_norm(s, (D,), gamma, beta, eps)`。

## 这题的坑

**会报错的**

1. **`token_embeddings[token_id]`**：参数名是 `token_ids`，`NameError`。

**不报错、结果静默错的**

2. **mean 和 var 用反**：写成 `(s - var) / sqrt(mean**2 + eps)`。分子减的是**均值**（让数据以 0 为中心），分母开根号的是**方差**（缩放到单位标准差）。`var` 本身已经是平方量，不要再平方；`mean**2` 是均值的平方，不是方差。
3. **方差用无偏**：`s.var(dim=-1)` 默认 `unbiased=True` 除以 $D-1$，和参考答案对不上。要写 `s.var(dim=-1, unbiased=False, keepdim=True)`，或像题解那样手算。
4. **忘了 `keepdim=True`**：mean 变成 `[B, T]`，和 `[B, T, D]` 相减时会按最后一维对齐，形状碰巧能广播（比如 $T = D$）时不报错、结果错。
5. **写 output**：`output.copy_(y)`，不要 `output = y`（只改了局部变量），见 [通用语法坑](./#通用语法坑)。

## 和 RMSNorm 的区别

| | LayerNorm（本题） | RMSNorm（[#50](./rms-normalization)） |
|---|---|---|
| 减均值 | 是 | 否 |
| 分母 | $\sqrt{\sigma^2 + \epsilon}$ | $\sqrt{\frac{1}{D}\sum s_d^2 + \epsilon}$ |
| β | 有 | 无 |
| 用在 | BERT、GPT-2、ViT | LLaMA、Qwen 等主流 LLM |

embedding 后面直接接 LayerNorm 是 BERT 的写法（`BertEmbeddings`：token + position + segment，再 LayerNorm）。GPT-2 / LLaMA 是 pre-norm，embedding 输出直接进第一个 block，norm 放在每个 block 的入口。
