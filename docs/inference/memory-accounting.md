---
title: 显存账：权重 / KV cache / 激活
status: draft
tags: [memory, kv-cache]
difficulty: 3
order: 2
related: [/inference/prefill-decode-roofline, /inference/attention-variants, /parallel/parallelism-overview]
stack: [ld-dtype, 4]
---

# 显存账：权重 / KV cache / 激活

> 能口算「70B bf16 + 8k context + batch 32 要多少卡」

## 一句话结论

推理显存只有四项：**权重 + KV cache + 激活 + 固定开销**。权重是常数，激活很小，唯一随 batch 和 context 线性涨的是 KV cache，所以「要几张卡」本质是「KV 要多少」。70B bf16 权重 132 GiB，KV 320 KiB/token，8k × 32 = 26 万 token 正好 80 GiB，加起来 220 GiB，H100 上要 4 张（TP=4）。

## 推导

### 权重

$$
W = P \cdot b_w
$$

$P$ 是参数量，$b_w$ 是每参数字节数（bf16 = 2，fp8/int8 = 1，int4 = 0.5）。**bf16 下 GiB 数约等于参数量 B 数的两倍**，这是最有用的口算：7B → 14 GiB，70B → 140 GiB（精确算 70.55B × 2 = 131.4 GiB，因为 1 GiB = 1.07e9）。

参数量本身可以从结构推：每层

$$
\underbrace{2d^2}_{W_q, W_o} + \underbrace{2 \cdot d \cdot H_{kv} \cdot d_h}_{W_k, W_v} + \underbrace{3 \cdot d \cdot d_{ff}}_{\text{SwiGLU}} + \underbrace{2d}_{\text{norms}}
$$

再加 embedding $V \cdot d$（untied 则 ×2）。注意 SwiGLU 是**三个**矩阵（gate、up、down），不是两个，这是常见的算错点。

### KV cache：唯一会爆的项

每个 token 每层要存 K 和 V 各一份，每份 $H_{kv} \cdot d_h$ 个元素：

$$
\text{KV/token} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b_{kv}
$$

Llama-3-70B：$2 \times 80 \times 8 \times 128 \times 2 = 327680$ B = **320 KiB/token**。

总量：

$$
\text{KV} = \text{KV/token} \cdot B \cdot S
$$

$8192 \times 32 = 262144$ token，$\times 320\ \text{KiB} = 80\ \text{GiB}$。

MLA 不一样，只缓存压缩后的 latent 加一个所有 head 共享的 RoPE key：

$$
\text{KV/token}_{\text{MLA}} = L \cdot (d_c + d_r) \cdot b_{kv}
$$

DeepSeek-V3：$61 \times (512 + 64) \times 2 = 70272$ B ≈ 68.6 KiB/token，比同规模 GQA 还小。

### 激活

推理时激活不是大头，因为不需要为反向保存中间结果，且 FlashAttention 不物化 $S \times S$ 的 score 矩阵。一层的活跃张量大约

$$
A \approx B \cdot S_{\text{chunk}} \cdot (4d + 2 d_{ff}) \cdot b
$$

$4d$ 来自 Q、K、V、attention 输出，$2d_{ff}$ 来自 gate 和 up。加上 logits $B \cdot V \cdot 4$（fp32，只算最后一个 token，但 vocab 大时不容忽视：batch 256 × 128k vocab × 4 B = 128 MiB）。

### 固定开销

CUDA context、NCCL buffer、CUDA graph 捕获的 memory pool、kernel workspace，实践中留 1–3 GiB。vLLM 的 `gpu_memory_utilization`（默认 0.9）就是把这部分连同碎片一起兜住，剩下的全给 KV。

### 几张卡

$$
N = \left\lceil \frac{W + \text{KV} + A + \text{overhead}}{\text{HBM} \times \text{util}} \right\rceil
$$

TP 下权重和 KV 都按 $1/\text{TP}$ 分，激活不分（每卡都要完整的 $[B, S, d]$）。**陷阱**：KV 按 head 切，$H_{kv} < \text{TP}$ 时 KV head 只能复制，每卡 KV 不再是 $1/\text{TP}$。Llama-3-70B 的 $H_{kv} = 8$，所以 TP=16 时 KV 在每两张卡上是重复的。

#### 代码演示

把上面几项串起来，回答两个问题：Llama-3-70B bf16 在 80 GiB 的 H100 上，TP = 2 / 4 / 8 时各能放多少 token 的 KV？同时服务 32 个 8k 上下文的请求最少几张卡？

口径：每张卡按 `gpu_memory_utilization = 0.9` 只用 72 GiB，再留 3 GiB 给激活峰值和 CUDA Graph 内存池，剩 69 GiB 给权重和 KV；权重按 TP 均分。改参数可以直接重跑。

```python
import math

GiB = 2**30

# Llama-3-70B, bf16
P, b_w = 70.6e9, 2                    # 参数量, 每参数字节
L, H_kv, d_h, b_kv = 80, 8, 128, 2    # 层数, KV head 数, head 维度, KV 每元素字节
kv_per_token = 2 * L * H_kv * d_h * b_kv     # K 和 V 各一份: 327,680 B = 320 KiB
W = P * b_w                                  # 权重: 131.5 GiB

HBM, util, act = 80 * GiB, 0.9, 3 * GiB
usable = HBM * util - act                    # 每卡 69 GiB 给权重和 KV

def kv_tokens(tp):
    """TP 张卡合起来能放多少 token 的 KV。"""
    free = tp * usable - W
    return max(0, int(free // kv_per_token))

for tp in (2, 4, 8):
    t = kv_tokens(tp)
    print(f"TP={tp}: KV {t * kv_per_token / GiB:5.1f} GiB = {t:>9,} token = {t // 8192:3d} 个 8k 请求")
# TP=2: KV   6.5 GiB =    21,290 token =   2 个 8k 请求
# TP=4: KV 144.5 GiB =   473,488 token =  57 个 8k 请求
# TP=8: KV 420.5 GiB = 1,377,885 token = 168 个 8k 请求

def min_tp(batch, ctx):
    """同时服务 batch 个 ctx 长的请求，最少几张卡（TP 要整除 H_kv）。"""
    need = W + batch * ctx * kv_per_token
    n = math.ceil(need / usable)
    tp = next((t for t in (1, 2, 4, 8) if t >= n), None)   # None: 一个节点放不下
    return round(need / GiB, 1), tp

print(min_tp(32, 8192))    # (211.5, 4)   权重 131.5 + KV 80 GiB
print(min_tp(64, 8192))    # (291.5, 8)   batch 翻倍，只多 80 GiB
print(min_tp(32, 32768))   # (451.5, 8)   上下文 ×4，KV ×4
```

怎么读这几个数：

1. **TP = 2 基本不可用**。两张卡 138 GiB，权重就占了 131.5 GiB，只剩 6.5 GiB 给 KV，两个 8k 请求就满了。
2. **TP 从 2 到 4，KV 空间涨 20 多倍**。权重是固定的一块，多出来的两张卡几乎全部变成 KV。这就是开头说的「要几张卡本质是 KV 要多少」。
3. **32 × 8k 要 4 张卡**：$131.5 + 80 = 211.5$ GiB，÷ 69 ≈ 3.07，向上取到 3 张；但 TP 要整除 $H_{kv} = 8$（也要整除 Q head 数 64），只能取 4。
4. **只有 KV 随并发和长度线性涨**。batch 翻倍或上下文 ×4，涨的全是 KV 那一项，权重和每卡的激活、固定开销都不变。所以算卡数时先算 KV，再加上常数项。
5. 这里按每个请求都满 8k 算的是最坏情况。PagedAttention 按实际 token 数分配，真正的约束是所有活跃序列的 token 总数 × KV/token ≤ KV 空间（见下面的面试追问）。

## 交互

改 batch、context、dtype，看哪一项先把显存吃满。把 KV dtype 调到 fp8 看 KV 减半，把模型换成 DeepSeek-V3 看 MLA 的 KV 有多小。

<MemoryCalculator />

## 面试追问

::: details Q：为什么 KV cache 是 2 倍，不是 1 倍或 3 倍？
K 和 V 各存一份，Q 不需要存（每步重新算，只有当前 token 的 Q）。所以是 2。MLA 是例外：它存的是一个能同时恢复 K 和 V 的 latent，加一个 decoupled RoPE key，所以公式里没有 2。
:::

::: details Q：batch 加倍，显存加倍吗？
不。权重不变，只有 KV 和激活加倍。70B + 8k 的例子里，batch 从 32 到 64，总显存从 220 GiB 到 300 GiB，涨了 36% 不是 100%。这也是为什么 continuous batching 值得做：权重的访存成本被更多请求摊薄，而显存代价是次线性的。
:::

::: details Q：weight-only 量化（W4A16）省的是显存还是带宽？
两者都省，而且省的是同一件事。decode 时权重从 HBM 读一遍才能算一个 token，int4 让这次读取变成 1/4。既然 decode 是 memory-bound，访存减 4 倍就意味着接近 4 倍的 TPOT 提升。激活保持 fp16 是因为 decode 时激活很小（batch × d），量化它没收益还掉精度。
:::

::: details Q：8k context、batch 32，但请求长度不一，怎么算？
上面算的是最坏情况（所有请求都满 8k）。实际用 PagedAttention 后按实际 token 数分配，所以真正的约束是「所有活跃序列的 token 总数 × KV/token ≤ 剩余显存」。调度器据此决定能同时跑多少请求，这个数就是 vLLM 启动时打印的 KV cache blocks。
:::

::: details Q：为什么不把 KV cache 放到 CPU 内存？
PCIe 4.0 是 32 GB/s，H100 的 HBM 是 3.35 TB/s，差 100 倍。decode 每步都要读整个 KV，放 CPU 意味着每步多花 100 倍的时间。只有抢占时的 swap 才会这么做，因为那是一次性的，不在每步的关键路径上。
:::

## 参考

- [vLLM: Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Transformer Inference Arithmetic — kipply](https://kipp.ly/transformer-inference-arithmetic/)
- [DeepSeek-V2: MLA](https://arxiv.org/abs/2405.04434)
