---
title: Attention 变体：MHA / MQA / GQA / MLA
status: draft
tags: [attention, gqa, mla]
difficulty: 3
order: 4
related: [/inference/memory-accounting, /inference/kv-cache-paged-attention, /handson/mha-gqa-forward]
stack: [f-model]
---

# Attention 变体：MHA / MQA / GQA / MLA

> KV 大小与 decode 带宽影响

## 一句话结论

这四个变体解决的是同一个问题：**decode 时 KV cache 的访存量**。MHA 每层每 token 存 $2 H d_h$ 个元素，MQA 压到 $2 d_h$（质量掉太多），GQA 折中到 $2 H_{kv} d_h$（工业标准），MLA 换个思路，存一个 $d_c$ 维的压缩 latent 再在计算时展开，比 GQA 还小且质量更好。

## 推导

### 为什么只动 K 和 V

decode 时 Q 只有当前一个 token，算完就丢。K 和 V 必须留着，因为后续每个 token 都要 attend 到它们。所以 KV cache 的大小只由 K/V 的 head 数决定，跟 Q 的 head 数无关。这就给了一个免费的不对称性：**Q 保持 $H$ 个 head 不损失表达力，K/V 减到 $H_{kv}$ 个 head 直接省访存**。

$$
\text{KV/token/层} = 2 \cdot H_{kv} \cdot d_h \cdot b
$$

| 变体 | $H_{kv}$ | Llama-3-70B 规格下 KV/token | 相对 MHA |
|---|---|---|---|
| MHA | $H$ = 64 | 2.5 MiB | 100% |
| GQA | 8 | 320 KiB | 12.5% |
| MQA | 1 | 40 KiB | 1.6% |
| MLA | — | 68.6 KiB（DeepSeek-V3 规格） | — |

### GQA 的分组

$H$ 个 Q head 分成 $H_{kv}$ 组，每组 $H / H_{kv}$ 个 Q head 共享一对 K/V head。$H_{kv} = H$ 退化成 MHA，$H_{kv} = 1$ 退化成 MQA。实现上是 `repeat_interleave`，但**好的 kernel 不真的复制**：FlashAttention 让同组的 Q head 读同一块 K/V tile，省的是 HBM 流量而不只是显存。

从 MHA checkpoint 转 GQA 可以用 mean-pooling 每组的 K/V 投影再 uptrain 少量 token，这是 GQA 论文的做法，不用从头训。

### MLA 的两段式

MLA 把 K/V 投影拆成「下投影到 latent」和「上投影回 head」：

$$
c_{kv} = x W^{DKV} \in \mathbb{R}^{d_c}, \quad K = c_{kv} W^{UK}, \quad V = c_{kv} W^{UV}
$$

**只缓存 $c_{kv}$**（DeepSeek-V3 里 $d_c = 512$）。但这样有个问题：RoPE 是位置相关的，不能和 $W^{UK}$ 交换顺序。解法是 decoupled RoPE：另外拿一个 $d_r = 64$ 维的、所有 head 共享的 $k_{pe}$ 单独加 RoPE 并缓存。所以每 token 缓存 $d_c + d_r = 576$ 个元素。

算一下 DeepSeek-V3（$L = 61$ 层，bf16）每 token 的 KV：

$$
\text{KV/token} = L\,(d_c + d_r) \cdot 2 = 61 \times 576 \times 2 = 70{,}272\ \text{B} \approx 68.6\ \text{KiB}
$$

同样 61 层，换成 8 个 KV head、$d_h = 128$ 的 GQA 是 $2 \times 61 \times 8 \times 128 \times 2 = 244$ KiB，是 MLA 的 3.6 倍。latent 是所有 head 共用的一份，大小像 MQA；但每个 head 用自己的 $W^{UK}_i, W^{UV}_i$ 从它还原出不同的 K、V，所以不像 MQA 那样掉效果。

为什么 RoPE 不能直接加在上投影出来的 K 上：RoPE 给位置 $t$ 的向量乘旋转矩阵 $R_t$，并且 $R_m^\top R_n = R_{n-m}$。若 $k_n = R_n W^{UK} c_n$，位置 $m$ 的 query 是 $R_m q$，score 变成

$$
(R_m q)^\top R_n W^{UK} c_n = q^\top R_{n-m}\, W^{UK} c_n
$$

夹在中间的 $R_{n-m} W^{UK}$ 对每个历史位置 $n$ 都不一样，没法像下面那样提前并进 $q$，只能对每个 $n$ 展开 K，又回到存完整 K 的代价（DeepSeek-V2，[arXiv 2405.04434](https://arxiv.org/abs/2405.04434) §2.1.3）。所以 K 拆成两段：内容段 $W^{UK} c_n$ 不加 RoPE，照常吸收；位置段 $k_{pe}$ 只有 64 维、所有 head 共用、直接缓存。score 是两段点积之和，除以 $\sqrt{d_h + d_r} = \sqrt{192}$。

decode 时的关键技巧是 **weight absorption**：把 $W^{UK}$ 吸收进 Q 侧，直接在 latent 空间算 score，不需要把 K 展开：

$$
q^\top K^\top = q^\top (c_{kv} W^{UK})^\top = \underbrace{(q^\top W^{UK\top})}_{\text{预先算好}} c_{kv}^\top
$$

这样每 head 的 score 计算变成 $[1, d_c] \times [d_c, S_{kv}]$，读的还是那份共享的 $c_{kv}$。prefill 时反而直接展开走标准 attention 更快（compute-bound，展开的 FLOPs 无所谓）。

输出侧同理：$\sum_t p_t\, v_{t,i} = W^{UV}_i \big(\sum_t p_t\, c_t\big)$，先在 512 维上加权求和，最后乘一次 $W^{UV}_i$（还能并进 $W^O$）。于是 128 个 head 读同一份 576 维的 KV，形状上就是只有一个 KV head 的 MQA。

**代价**：MLA 的 $d_c = 512$ 大于单个 head 的 $d_h = 128$，所以计算量比 GQA 大。它是拿算力换带宽，正好符合 decode memory-bound 的处境。按每层每个历史 token 算（$H = 128$ 个 head，bf16）：

1. 吸收后：读 $576 \times 2 = 1152$ B；FLOPs $= 128 \times 2 \times (576 + 512) = 278{,}528$；AI ≈ **242** FLOP/B，接近 H100 的 ridge 295。
2. 展开成每 head 的 K（128 + 64 维）和 V（128 维）：读 $128 \times 320 \times 2$ B，FLOPs $= 128 \times 2 \times 320$，AI = **1**。

## 交互

切换变体和 phase，看 KV cache 的读写点落在哪一步、shape 怎么变。注意 decode 下 MLA 读的是 `[B, S_kv, d_c]` 而 GQA 读的是 `[B, H_kv, S_kv, h_d]`。

<ShapeFlow variant="gqa" phase="decode" />

## 面试追问

::: details Q：GQA 为什么不会明显掉点，MQA 会？
K/V 承载的是「被检索的内容」，多个 Q head 共享一组 K/V 相当于让它们在同一个子空间里检索，仍保留了 Q 侧的多样性。$H_{kv} = 8$ 时子空间数量仍然够用；$H_{kv} = 1$ 时所有 head 被迫在同一个子空间检索，表达力塌缩，且训练不稳定。经验值是 $H_{kv}$ 取 4–8，再少收益递减而损失陡增。
:::

::: details Q：GQA 和 TP 一起用有什么坑？
KV head 按 TP 切，$H_{kv} = 8$ 时 TP 最多切到 8。TP=16 就必须把 KV head 复制到两张卡上，每卡的 KV 显存不再是 $1/16$ 而是 $1/8$，而且算了两遍。所以大 TP 下 GQA 的显存收益会打折，这时 MLA 的优势更明显（latent 可以按别的维度切，或者干脆用 DP attention）。
:::

::: details Q：MLA 的 weight absorption 为什么 prefill 不用？
prefill 是 compute-bound，瓶颈在算力不在访存。吸收后每 head 都要和整个 $d_c = 512$ 维的 latent 做运算，FLOPs 比展开成 $d_h = 128$ 的标准 attention 更多。所以 prefill 展开、decode 吸收，两条路径分别优化，这也是 DeepSeek 官方实现和 vLLM/SGLang 里的做法。
:::

::: details Q：KV 量化到 fp8 和用 GQA，哪个更划算？
正交，可以叠加。GQA 是结构性的，必须在训练时就定，收益 8 倍；KV fp8 是推理时的，收益 2 倍，但会引入量化误差，长 context 下误差会累积（因为早期 token 的 KV 被反复读）。实践中先吃 GQA 的结构红利，KV 量化作为 context 特别长时的补充手段，且通常 per-head 或 per-token 量化而非 per-tensor。
:::

## 手撕

见 [手写 MHA / GQA Forward](/handson/mha-gqa-forward)。

## 参考

- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150)
- [DeepSeek-V2 / V3 技术报告（MLA）](https://arxiv.org/abs/2405.04434)
