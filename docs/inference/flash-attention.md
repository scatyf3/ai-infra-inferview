---
title: FlashAttention v1 / v2 / v3
status: draft
tags: [flash-attention, kernel]
difficulty: 4
order: 5
related: [/handson/online-softmax, /handson/stable-softmax, /inference/attention-variants, /inference/prefill-decode-roofline, /inference/kv-cache-paged-attention, /gpu/tensor-core-gemm, /gpu/profiling]
stack: [k-attn]
---

# FlashAttention v1 / v2 / v3

> tiling + online softmax，为什么省的是 HBM 访问而不是 FLOPs；v2 改了什么切分；v3 怎么用 Hopper；decode 用的 FlashDecoding

## 一句话结论

标准 attention 要把 $S \times S$ 的 score 矩阵写到 HBM 再读回来做 softmax，FlashAttention 用 tiling 把 Q、K、V 分块搬进 SRAM，用 online softmax 在块之间增量更新 max 和分母，整个过程不把 score 矩阵写回 HBM。FLOPs 没少，省的是 HBM 读写：下面的例子里，标准实现的算术强度约 63 FLOP/B，是 memory-bound；FlashAttention 把它推到约 4000 FLOP/B，变成 compute-bound。之后的 v2、v3 不再省字节，而是想办法把 tensor core 用满。

## 推导

### 符号和约定

| 符号 | 含义 |
|---|---|
| $S$ | 序列长度（prefill 时 query 和 key 都是 $S$ 个） |
| $d_h$ | 每个 head 的维度，常见 64 / 128 |
| $H$ | Q head 数，$d = H d_h$ |
| $B_r, B_c$ | Q 块的行数、K/V 块的行数（tile 大小） |
| $b$ | 每个元素的字节数，bf16 = 2 |

单个 head 的 attention：$O = \text{softmax}(QK^\top / \sqrt{d_h})\,V$，其中 $Q, K, V, O \in \mathbb{R}^{S \times d_h}$，按行主序存放，每行是一个 token。下文的 FLOPs 先按乘加（MAC）数再 ×2，规则同 [Roofline 页](/inference/prefill-decode-roofline#符号和计数规则)。

GPU 有两级存储要分清。**HBM** 是显存，H100 SXM 80 GB、3.35 TB/s。**SRAM** 指每个 SM 上的 shared memory 和寄存器，H100 每个 SM 的 shared memory 最多 228 KB（[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)），132 个 SM 加起来也就 30 MB 左右，但带宽比 HBM 高一个数量级。FlashAttention 论文给的 A100 数字是 SRAM 约 19 TB/s、HBM 1.5–2.0 TB/s（[Dao et al., 2022, §2.1](https://arxiv.org/abs/2205.14135)）。

### 标准实现：score 矩阵来回搬

教科书写法是三个 kernel（[FlashAttention 论文 Algorithm 0](https://arxiv.org/abs/2205.14135)）：

1. 读 $Q, K$，算 $S_{\text{score}} = QK^\top$，**写** $S \times S$ 到 HBM。
2. **读** $S_{\text{score}}$，做 softmax，**写** $P$（还是 $S \times S$）。
3. **读** $P$ 和 $V$，算 $O = PV$，写 $O$。

$S \times S$ 的矩阵被搬了 4 遍。算一笔：$S = 8192$，$H = 64$，$d_h = 128$，bf16，一层，batch 1，按 MHA 数（GQA 只会让 K/V 更小，不影响结论）：

- 一份 score：$S^2 \cdot H \cdot b = 8192^2 \times 64 \times 2 \approx 8.6$ GB。搬 4 遍约 34 GB，按 3.35 TB/s 要 **10.3 ms**。
- FLOPs：$QK^\top$ 和 $PV$ 各 $S^2 d_h$ 次 MAC/head，乘 $H$ 得 $2S^2 d$ MAC，即 $4 S^2 d = 4 \times 8192^2 \times 8192 \approx 2.2$ TFLOP（不带 causal mask 的满额）。按 989 TFLOP/s 要 **2.2 ms**。
- 算术强度 $\text{AI} = 2.2\text{T} / 34\text{G} \approx 63$ FLOP/B，远低于 H100 的 ridge 295 → **memory-bound**，时间被 score 的搬运决定。

注意这还没算显存：8.6 GB 的 score 要真实存在 HBM 里，长序列直接 OOM。

### tiling + online softmax：score 不出 SRAM

难点在 softmax：每一行要先知道整行的最大值和分母才能归一化，看起来必须先算完整行 score。online softmax（[Milakov & Gimelshein, 2018](https://arxiv.org/abs/1805.02867)）解决了这个：一行分成若干块依次处理，维护三个量：

- $m$：目前见过的最大值
- $\ell$：目前的分母 $\sum e^{s - m}$
- $\text{acc}$：目前未归一化的输出 $\sum e^{s - m} v$

新来一块 score $s^{(j)}$ 时，最大值可能变大，旧的 $\ell$ 和 $\text{acc}$ 都是按旧 $m$ 算的，乘一个修正因子换算过来：

$$
m' = \max(m, \max s^{(j)}),\quad
\alpha = e^{m - m'},\quad
\ell' = \alpha \ell + \sum e^{s^{(j)} - m'},\quad
\text{acc}' = \alpha\, \text{acc} + e^{s^{(j)} - m'} V^{(j)}
$$

全部块处理完，$O = \text{acc} / \ell$。逐步推导和数字例子见 [Online Softmax](/handson/online-softmax)。

有了这个递推，一个 Q 块（$B_r$ 行）可以常驻 SRAM，让 K/V 一块一块流过，每块的 score 只是 $B_r \times B_c$ 的小 tile，算完就丢。前向伪代码（v2 的循环顺序，下节解释），单个 head：

```python
def flash_attn_fwd(Q, K, V, Br=128, Bc=128, causal=True):
    S, d = Q.shape                              # 行主序，每行一个 token
    O, LSE = zeros(S, d), zeros(S)
    for q0 in range(0, S, Br):                  # 每个 Q 块交给一个 thread block，块之间并行
        q = load(Q[q0:q0+Br])                   # HBM -> SRAM，只读一次
        m = full(Br, -inf); l = zeros(Br); acc = zeros(Br, d)   # 都在寄存器里
        k_end = q0 + Br if causal else S        # causal：右上方整块被 mask 的 tile 直接跳过
        for k0 in range(0, k_end, Bc):
            k, v = load(K[k0:k0+Bc]), load(V[k0:k0+Bc])
            s = q @ k.T / sqrt(d)               # [Br, Bc] tile，不写回 HBM
            if causal and k0 + Bc > q0:         # 只有对角 tile 需要逐元素 mask
                i = arange(q0, q0 + Br)[:, None]
                j = arange(k0, k0 + Bc)[None, :]
                allowed = j <= i
                s = where(allowed, s, -inf)
            m_new = maximum(m, s.max(axis=1))
            alpha = exp(m - m_new)              # 旧结果的修正因子
            p = exp(s - m_new[:, None])
            l = alpha * l + p.sum(axis=1)
            acc = alpha[:, None] * acc + p @ v
            m = m_new
        O[q0:q0+Br] = acc / l[:, None]          # 最后只除一次
        LSE[q0:q0+Br] = m + log(l)              # 存 logsumexp，反向用来重算 P
    return O, LSE
```

这段的 numpy 版本对 $S \in \{1, 7, 64, 130, 257\}$、causal 和非 causal 都和朴素实现逐元素对上（见下方手撕）。

### 访存账：省了多少，为什么 FLOPs 不降

还是上面那个例子。FlashAttention 理想情况下 Q、K、V 各读一遍、O 写一遍：$4 \cdot S \cdot d \cdot b = 4 \times 8192 \times 8192 \times 2 \approx 0.54$ GB，**0.16 ms**。FLOPs 还是 2.2 TFLOP，所以 $\text{AI} \approx 4096$ FLOP/B，远在 ridge 右边，时间由算力决定（理想 2.2 ms）。从 10.3 ms 的搬运主导变成了 2.2 ms 的计算主导。

两个要说清的细节：

1. **K/V 其实被读了很多遍。** 外层每个 Q 块都要把整个 K、V 扫一遍，共 $S / B_r$ 遍。论文的严格结论是 HBM 访问量 $\Theta(S^2 d_h^2 / M)$，$M$ 是 SRAM 能放的元素数；标准实现是 $\Theta(S d_h + S^2)$（[Dao et al., 2022, Theorem 2](https://arxiv.org/abs/2205.14135)）。$d_h = 128$、$M \approx 114$K 个 bf16 元素时 $d_h^2 / M \approx 0.14$，即 HBM 访问约是标准实现的 1/7，而不是 1/64。实际更好，是因为同一个 head 的不同 Q 块同时跑在不同 SM 上，它们读的是同一份 K/V（一个 head 是 $2 \times 8192 \times 128 \times 2 \approx 4$ MB），能在 50 MB 的 L2 里复用。所以「流量 O(S·d)」是 L2 命中良好时的近似，不是定理。这一条是按硬件参数推的估计，实测要用 ncu 看 DRAM bytes（见 [Profiling](/gpu/profiling)）。
2. **FLOPs 一点没少，训练时还多了。** 前向：矩阵乘还是 $QK^\top$ 和 $PV$，rescale 多出 $O(S^2)$ 次标量运算。训练还要做反向（backward，求梯度），求 $dV = P^\top dO$ 等梯度要用到 $P$。标准实现在前向把 $P$ 存在 HBM 里留给反向；FlashAttention 不存 $P$（存了就又是 $S \times S$），只存每行一个 LSE，反向时从 $Q, K$ 重算 $S_{\text{score}}$ 和 $P = e^{S_{\text{score}} - \text{LSE}}$，多一次 $QK^\top$。推理只有前向，没有这笔。用算力换访存之所以划算，正是因为标准实现是 memory-bound。

causal mask 下右上方整块被 mask 的 tile 直接跳过（伪代码里的 `k_end`），实际 FLOPs 约是满额的一半。

### v2：换循环顺序、按序列并行、warp 之间不通信

FlashAttention-2（[Dao, 2023](https://arxiv.org/abs/2307.08691)）的出发点：v1 已经不是 memory-bound 了，但只跑到 A100 理论峰值的 25–40%，GEMM 能到 80% 左右。剩下的问题是工作怎么切分。三处改动：

1. **循环顺序**：v1 外层遍历 K/V 块、内层遍历 Q 块，每处理一个 (K/V 块, Q 块) 对都要把这个 Q 块的 $O_i, m_i, \ell_i$ 从 HBM 读出来、更新、写回去，$O$ 被读写 $S / B_c$ 遍。v2 外层遍历 Q 块，$O_i, m_i, \ell_i$ 一直待在寄存器里，最后写一次（就是上面伪代码的结构）。
2. **少做非 matmul 运算**：$\text{acc}$ 全程不归一化，最后除一次 $\ell$；反向只存一个 LSE $= m + \log \ell$，而不是 $m$ 和 $\ell$ 两个。这很重要，因为 A100 上 fp16 matmul 是 312 TFLOP/s，非 matmul 的 fp32 只有 19.5 TFLOP/s，非 matmul 的一次运算约等于 16 次 matmul 运算（[Dao, 2023, §3.1](https://arxiv.org/abs/2307.08691)）。
3. **并行维度和 warp 分工**：
   - v1 只在 batch × head 上并行，batch 小、序列长时 thread block 不够填满 108 个 SM。v2 外层是 Q 块，Q 块之间互不依赖，于是再沿序列维并行，grid 是 batch × head × $(S / B_r)$。
   - thread block 内部，v1 把 K/V 切给 4 个 warp（「split-K」），每个 warp 算出部分结果后要经 shared memory 汇总。v2 改为把 Q 切给 warp，每个 warp 看到全部 K/V，各自输出不同的行，warp 之间不用通信。

作者自测：约比 v1 快 2 倍，前向在 A100 上达到理论峰值的 50–73%（[Dao, 2023](https://arxiv.org/abs/2307.08691)）。

### v3：Hopper 上的异步和 FP8

FlashAttention-2 在 H100 上只有约 35% 利用率（[Shah et al., 2024](https://arxiv.org/abs/2407.08608)），因为它没用上 Hopper 的新硬件：

- **TMA**（Tensor Memory Accelerator）：一个线程发起，硬件异步把一整块 tile 从 HBM 搬到 shared memory，搬的同时其他 warp 继续算。
- **wgmma**：一个 warpgroup（4 个 warp，128 线程）一起发起的异步矩阵乘，直接从 shared memory 读操作数，吞吐比 Ampere 风格的 `mma.sync` 高。

v3 的三个技巧都围绕「让搬数据、GEMM、softmax 同时进行」：

1. **warp specialization**：thread block 里分出 producer warp 只管用 TMA 搬 K/V，consumer warpgroup 只管算，两者用 shared memory 里的环形缓冲区和 barrier 同步。
2. **ping-pong 调度 + warpgroup 内流水**：softmax 里的 `exp` 很慢。H100 SXM 的 fp16 matmul 是 989 TFLOP/s，而 `exp` 这类特殊函数只有约 3.9 T 次/秒（16 次/SM/周期 × 132 SM × 1830 MHz，[Shah et al., 2024, §3.1](https://arxiv.org/abs/2407.08608)）。算一下比例：每个 score 元素要 1 次 `exp`，而 $QK^\top$ 和 $PV$ 在它身上花 $2 d_h$ 次 MAC $= 4 d_h = 512$ FLOP（$d_h = 128$）。于是 `exp` 耗时 ÷ matmul 耗时 $= (1 / 3.9\text{T}) \div (512 / 989\text{T}) \approx 0.5$，如果两者串行，总时间是 matmul 的 1.5 倍，tensor core 有三分之一的时间在等 `exp`。v3 让两个 warpgroup 交替：A 做 softmax 时 B 做 GEMM；同一个 warpgroup 内也把第 $j$ 块的 softmax 和第 $j+1$ 块的 $QK^\top$ 重叠。
3. **FP8**：FP8 的 wgmma 吞吐翻倍，但有两个麻烦。一是精度：用 block quantization（每个 tile 一个 scale），再用 incoherent processing，即 Q、K 都乘同一个随机正交矩阵 $M$（带随机符号的 Hadamard），因为 $(QM)(KM)^\top = QK^\top$，结果不变但 outlier 被摊开，更好量化。二是布局：FP8 wgmma 只接受 k-major 的操作数，$PV$ 这一步要求 $V$ 的 tile 沿序列维连续，所以 $V$ 要么预先在 HBM 里转置，要么载入 shared memory 后在 kernel 内转置（[Shah et al., 2024, §3.3](https://arxiv.org/abs/2407.08608)）。

作者自测：H100 上比 v2 快 1.5–2.0 倍，FP16 到 740 TFLOP/s（约 75% 利用率），FP8 接近 1.2 PFLOP/s，FP8 的数值误差比基线 FP8 attention 低 2.6 倍（[Shah et al., 2024](https://arxiv.org/abs/2407.08608)）。

### decode：FlashDecoding / split-KV

decode 时每个请求只有 1 个 query，score 是 $1 \times S_{ctx}$，本来就很小，FlashAttention 省 score 读写的那部分好处没了。瓶颈变成读 KV cache：Llama-3-70B（$H_{kv} = 8$，$d_h = 128$）、$S_{ctx} = 32$k，每层 KV 是 $2 \times 8 \times 128 \times 2 \times 32768 = 128$ MiB，按 3.35 TB/s 读满也要 40 µs。

问题在并行度。v2 按 batch × head × Q 块分 thread block，decode 时 Q 块只有 1 个。batch 1、GQA 下把同组 Q head 打包后只有 8 个 block，H100 有 132 个 SM，大部分闲着；单个 SM 拉不动整卡的 HBM 带宽，于是读 KV 的速度远低于 3.35 TB/s。FlashDecoding 博客报告，batch 1 时 FlashAttention 只用到不到 1% 的 GPU（[Dao et al., 2023](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)）。

**split-KV**（和 GEMM 的 split-K 同一个思路）：再沿 KV 长度切 $s$ 段，block 数变成 $B \times H_{kv} \times s$。

1. 每段各算局部输出 $o_j$（段内已归一化）和这一段的 $\text{lse}_j = m_j + \log \ell_j$，写回 HBM。只多写 $s$ 个 $d_h$ 维向量加 $s$ 个标量。
2. 一个小 kernel 合并，和 online softmax 的合并是同一个式子：

$$
\text{lse} = \log \sum_j e^{\text{lse}_j}, \qquad o = \sum_j e^{\text{lse}_j - \text{lse}}\; o_j
$$

$e^{\text{lse}_j - \text{lse}}$ 就是「第 $j$ 段的分母占总分母的比例」。上例切 $s = 16$ 段，block 数从 8 变成 128，接近 SM 数。

作者自测：CodeLlama-34B、batch 1、长序列下端到端 decode 最多快 8 倍，attention 算子本身最多快 50 倍，且到 32k 长度时 attention 时间几乎不随长度增加（[FlashDecoding 博客](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)）。$s$ 不是越大越好：段太短时每个 block 的工作量太小，合并 kernel 的开销占比上升。batch 大时 $B \times H_{kv}$ 本身就够填满 SM，就不用切了。后续的 FlashDecoding++（[Hong et al., 2023](https://arxiv.org/abs/2311.01282)）进一步用一个预设的统一最大值去掉段间同步，这里不展开。

PagedAttention 下 K/V 不连续，decode kernel 在 split-KV 之外还要按 block table 间接寻址，见 [KV Cache 与 PagedAttention](/inference/kv-cache-paged-attention)。

## 面试追问

::: details Q：FlashAttention 对 decode 有帮助吗？
直接的帮助有限。decode 时 score 只有 $1 \times S_{ctx}$，本来就不大，瓶颈是读整个 KV cache 的带宽，而 FlashAttention 的 KV 读取量和朴素实现一样多。真正的问题是并行度：只按 batch × head 分 block 时 SM 填不满。FlashDecoding 沿 KV 长度切分、多 block 并行再合并，让足够多的 SM 一起读 KV。
:::

::: details Q：FlashAttention 的 FLOPs 比标准实现多还是少？
前向基本一样，多了 rescale 的 $O(S^2)$ 次标量运算；causal 时跳过整块被 mask 的 tile，比「先算满再 mask」的朴素实现少一半。反向多一次 $QK^\top$ 重算。总体是「FLOPs 不降甚至略增，HBM 读写大降」。
:::

::: details Q：tile 大小 $B_r, B_c$ 受什么约束？
SRAM 要同时放下 Q 块 $B_r \times d_h$、K 块和 V 块各 $B_c \times d_h$、score tile $B_r \times B_c$（在寄存器里），还要给 TMA 的多级缓冲留位置。$d_h$ 越大 tile 越小，所以 $d_h = 256$ 的 kernel 比 $d_h = 128$ 难调。v1 论文取 $B_c = \lceil M / 4d \rceil$，$B_r = \min(\lceil M / 4d \rceil, d)$（[Dao et al., 2022, Algorithm 1](https://arxiv.org/abs/2205.14135)）；实际 kernel 按 head dim 和架构手调，常见 64 或 128。
:::

::: details Q：v2 为什么能沿序列维并行，v1 不行？
v1 外层是 K/V 块，不同 K/V 块要累加到同一个 $O_i$，有写冲突，只能在 batch × head 上并行。v2 外层是 Q 块，每个 Q 块独立产出自己的 $O_i$ 行，天然可以分给不同 thread block。反向则相反：v2 的反向外层遍历 K/V 块，$dK, dV$ 各自独立，$dQ$ 要跨块累加，用 atomic add。
:::

::: details Q：为什么 v3 要专门处理 exp？
因为 attention 已经是 compute-bound，tensor core 是关键资源。$d_h = 128$ 时每个 score 元素有 512 FLOP 的 matmul 和 1 次 exp，按 H100 的 989 T 对 3.9 T 算，exp 的耗时约是 matmul 的一半。不重叠的话 tensor core 有三分之一的时间在空等，所以要让 softmax 和另一个 warpgroup 的 GEMM 同时跑。
:::

## 手撕

常见题：

1. 写两块 K/V 的 online softmax 合并公式，说明为什么 $\text{acc}$ 也要乘修正因子。
2. 用 numpy / Triton 写一个简化的 FlashAttention 前向，并和朴素实现对比。概念铺垫见 [attention 变体](/inference/attention-variants)、[朴素 attention](/handson/naive-attention)。
3. 写 split-KV 的合并。

下面的 numpy 版本可以直接跑（mask 用 `allowed` 布尔矩阵，行索引 `[:, None]`、列索引 `[None, :]`）：

```python
import numpy as np

def flash_attn_fwd(Q, K, V, Br=64, Bc=64, causal=True):
    S, d = Q.shape
    O, LSE = np.zeros_like(Q), np.zeros(S)
    for q0 in range(0, S, Br):
        q = Q[q0:q0 + Br]; br = q.shape[0]
        m, l, acc = np.full(br, -np.inf), np.zeros(br), np.zeros((br, d))
        k_end = min(S, q0 + br) if causal else S
        for k0 in range(0, k_end, Bc):
            k, v = K[k0:k0 + Bc], V[k0:k0 + Bc]
            s = q @ k.T / np.sqrt(d)
            if causal and k0 + Bc > q0:
                i = np.arange(q0, q0 + br)[:, None]
                j = np.arange(k0, k0 + k.shape[0])[None, :]
                allowed = j <= i
                s = np.where(allowed, s, -np.inf)
            m_new = np.maximum(m, s.max(axis=1))
            alpha = np.exp(m - m_new)
            p = np.exp(s - m_new[:, None])
            l = l * alpha + p.sum(axis=1)
            acc = acc * alpha[:, None] + p @ v
            m = m_new
        O[q0:q0 + br] = acc / l[:, None]
        LSE[q0:q0 + br] = m + np.log(l)
    return O, LSE

def split_kv_decode(q, K, V, num_splits=4):
    """q: [d]，K, V: [S_ctx, d]。每段一个 block，最后合并。"""
    S, d = K.shape
    seg = -(-S // num_splits)
    outs, lses = [], []
    for s0 in range(0, S, seg):
        s = K[s0:s0 + seg] @ q / np.sqrt(d)
        m = s.max(); p = np.exp(s - m); l = p.sum()
        outs.append(p @ V[s0:s0 + seg] / l)
        lses.append(m + np.log(l))
    lses, outs = np.array(lses), np.stack(outs)
    lse = lses.max() + np.log(np.exp(lses - lses.max()).sum())
    return np.exp(lses - lse) @ outs

def ref_attn(Q, K, V, causal=True):
    S, d = Q.shape
    s = Q @ K.T / np.sqrt(d)
    if causal:
        i = np.arange(S)[:, None]; j = np.arange(S)[None, :]
        allowed = j <= i
        s = np.where(allowed, s, -np.inf)
    p = np.exp(s - s.max(axis=1, keepdims=True))
    return (p / p.sum(axis=1, keepdims=True)) @ V

rng = np.random.default_rng(0)
Q, K, V = (rng.standard_normal((257, 32)) for _ in range(3))
assert np.allclose(flash_attn_fwd(Q, K, V)[0], ref_attn(Q, K, V))
K, V, q = rng.standard_normal((1000, 32)), rng.standard_normal((1000, 32)), rng.standard_normal(32)
assert np.allclose(split_kv_decode(q, K, V, 7), ref_attn(q[None], K, V, causal=False)[0])
```

## 参考

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)（Dao et al., 2022）：Algorithm 0/1、Theorem 2 的 IO 复杂度
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)（Dao, 2023）
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)（Shah et al., 2024）
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)（Milakov & Gimelshein, 2018）
- [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682)（Rabe & Staats, 2021）：同期的分块 attention，侧重省显存
- [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)（Dao, Haziza, Massa, Sizov, 2023）
- [FlashDecoding++](https://arxiv.org/abs/2311.01282)（Hong et al., 2023）
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)：228 KB shared memory / SM、50 MB L2
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)：官方实现
