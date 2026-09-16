---
title: 手写 MHA / GQA Forward
status: draft
tags: [attention, gqa, handson]
difficulty: 3
order: 1
related: [/inference/attention-variants, /handson/decode-step-kv-cache, /handson/stable-softmax]
---

# 手写 MHA / GQA Forward

> shape 与 mask 处理

## 一句话结论

四步：投影并 reshape 到 `[B, H, S, d_h]`、GQA 时把 K/V 从 $H_{kv}$ 扩到 $H$、算 score 并加 causal mask、softmax 后乘 V 再合并 head。全程盯住 shape，`transpose(1, 2)` 之后必须 `contiguous()` 才能 `view`。

## 交互

先用这个把每一步的 shape 过一遍再动手写。

<ShapeFlow variant="gqa" phase="prefill" />

## 手撕

::: code-group

```python [MHA / GQA forward]
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads      # 等于 n_heads 就是 MHA，1 就是 MQA
        assert n_heads % self.n_kv_heads == 0
        self.n_rep = n_heads // self.n_kv_heads       # 每组多少个 Q head
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """[B, H_kv, S, d_h] -> [B, H_kv * n_rep, S, d_h]"""
        if n_rep == 1:
            return x
        b, h_kv, s, d = x.shape
        # expand 不拷贝，contiguous 之前是 view；真实 kernel 里连这一步都不做
        return x[:, :, None, :, :].expand(b, h_kv, n_rep, s, d).reshape(b, h_kv * n_rep, s, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S, _ = x.shape

        # 1) 投影 + 拆 head：[B, S, d] -> [B, H, S, d_h]
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2) GQA：把 K/V 广播到 H 个 head
        k = self.repeat_kv(k, self.n_rep)
        v = self.repeat_kv(v, self.n_rep)

        # 3) score + causal mask：[B, H, S, S]
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is None:
            mask = torch.full((S, S), float("-inf"), device=x.device).triu(1)
        scores = scores + mask                         # 上三角（未来位置）变 -inf

        # 4) softmax @ V -> 合并 head
        probs = F.softmax(scores.float(), dim=-1).type_as(q)   # 累加用 fp32，避免溢出
        out = probs @ v                                        # [B, H, S, d_h]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)  # transpose 后必须 contiguous
        return self.wo(out)
```

```python [调用 SDPA 的版本]
# 面试时可以先写上面的手写版，再说生产里这样写：
# enable_gqa 让 SDPA 自己处理分组，内部走 FlashAttention，不物化 S×S
out = F.scaled_dot_product_attention(
    q, k, v, is_causal=True, enable_gqa=True
)  # q: [B, H, S, d_h], k/v: [B, H_kv, S, d_h]
```

:::

### 常见错误

1. **忘了 `contiguous()`**：`transpose` 只改 stride，`view` 会报 "view size is not compatible"。用 `reshape` 可以绕过但会隐式拷贝。
2. **mask 加错方向**：causal mask 是 `triu(1)`（严格上三角为 -inf），`triu(0)` 会把对角线也 mask 掉，token 看不到自己。
3. **softmax 在 fp16 下溢出**：score 最大值可能到几十，`exp` 之后超过 fp16 的 65504。转 fp32 再 softmax，或者依赖 kernel 内部的 max-subtraction。
4. **缩放用错**：除的是 $\sqrt{d_h}$（单 head 维度）不是 $\sqrt{d}$。
5. **GQA 的 repeat 方向**：必须是 `repeat_interleave` 的语义（同组的 Q head 相邻），不是 `repeat`（整体重复）。写成后者时 head 和 KV 的对应关系全错，但 loss 还能降，很难查。

## 面试追问

::: details Q：为什么要除以 √d_h？
$q \cdot k$ 是 $d_h$ 个独立项求和，如果各分量是零均值单位方差，点积的方差就是 $d_h$。不缩放的话 $d_h = 128$ 时 score 的标准差是 11，softmax 会饱和成接近 one-hot，梯度消失。除以 $\sqrt{d_h}$ 把方差拉回 1。
:::

::: details Q：这段代码在 decode 时要改什么？
Q 的 S 变成 1，K/V 要和 cache 拼接，mask 不需要了（当前 token 能看到所有历史）。见 [带 KV Cache 的 Decode Step](/handson/decode-step-kv-cache)。
:::

::: details Q：FlashAttention 和这段代码算的结果一样吗？
数学上一样，数值上有微小差异。FlashAttention 用 online softmax 分块累加，累加顺序和一次性 softmax 不同，浮点不满足结合律所以有 1e-3 量级的差异（bf16 下）。这也意味着开关 FlashAttention 会让生成结果不完全一致，做 A/B 对比时要注意。
:::

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Llama 参考实现的 repeat_kv](https://github.com/meta-llama/llama/blob/main/llama/model.py)
