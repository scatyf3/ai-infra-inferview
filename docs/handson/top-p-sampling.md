---
title: Top-p 采样
status: todo
tags: [sampling, handson]
difficulty: 2
order: 5
related: []
stack: [o-sampling]
leetgpu: [60]
---

# Top-p 采样

> 与 top-k / temperature 组合

## 一句话结论

采样的三个旋钮按顺序作用：temperature 把 logits 除以 T 改变分布的尖锐度，top-k 只保留概率最高的 k 个 token，top-p（nucleus）只保留累计概率刚超过 p 的最小集合，然后在剩下的 token 里按归一化后的概率采样。top-p 比 top-k 好在候选集大小随分布自适应：分布尖时只留几个，分布平时多留一些。

## 推导

- **顺序**：logits / T → softmax → 按概率降序 → top-k 截断 → 累计和超过 p 处截断（保留第一个超过 p 的 token）→ 重新归一化 → `multinomial`。
- **top-p 的实现**：`sort` + `cumsum`，mask 掉「前面已经够 p 了」的 token；排序是 O(V log V)，大 vocab 下是采样阶段的主要开销。
- **batch 化**：每个请求的 T、k、p 不同，推理框架把它们打包成 tensor 一次处理整个 batch；贪心（T = 0）单独走 argmax。
- **其他**：min-p、repetition penalty、logit bias 都在同一个 logits processor 流水线里，见 [logits processor](/stack/o-logits)。

## 面试追问

::: details Q：top-p 的 cumsum mask 为什么要先减掉自身概率再比较？
如果直接 mask 掉 cumsum 大于 p 的位置，当第一个 token 的概率就超过 p 时会把所有 token 都 mask 掉。正确做法是用 cumsum 减去自身概率（等于前缀和右移一位）再和 p 比较，保证「刚好把累计推过 p 的那个 token」被保留，候选集至少有一个元素。
:::

## 手撕

对应 [LeetGPU #60](https://leetgpu.com/challenges)。框架：

```python
def sample(logits, temperature=1.0, top_k=0, top_p=1.0):
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    sorted_p, idx = probs.sort(descending=True)
    # top-k：sorted_p[k:] = 0
    # top-p：cum = sorted_p.cumsum(-1); mask = (cum - sorted_p) > top_p; sorted_p[mask] = 0
    sorted_p /= sorted_p.sum(-1, keepdim=True)
    return idx.gather(-1, torch.multinomial(sorted_p, 1))
```
