---
title: FlashAttention v1 / v2 / v3
status: draft
tags: [flash-attention, kernel]
difficulty: 4
order: 5
related: [/handson/online-softmax, /handson/stable-softmax, /inference/attention-variants, /inference/prefill-decode-roofline, /inference/kv-cache-paged-attention, /gpu/tensor-core-gemm, /gpu/profiling]
stack: [k-attn]
train: [tr-k-attn]
---

# FlashAttention v1 / v2 / v3


## naive attention

```python
def attn(q,k,v):
	d = q.size(-1)
	attn = torch.matmul(q,k.transpose(-1,-2))/ math.sqrt(d)
	attn = attn.masked_fill(~mask, float('-inf'))
	attn = torch.softmax(attn,dim=-1)
	out =  torch.matmul(attn,v)
	return out
```

这里从简单的kernel的角度，我们需要launch 3个kernel：
1. $QK^T$
2. softmax归一化attention score
3. $\text{attn} \times V$

>当然，我们还写了`/sqrt d`和+mask，但是在torch compile下，会被fuse进softmax，因为前两个是简单的pointwise，后面是reduction。这部分可以近似看作一个softmax kernel。

assume input是`[B,(H,)S,d]`：
1. 计算量：
	1. qk: $B \times S^2 \times d$个乘加
	2. softmax: 5BS² FLOPS
	   scale 1 次、mask select 1 次、求 max 1 次比较、减 max 1 次、exp 1 次、求 sum 1 次加法、归一化 1 次乘（乘倒数），合计约 5 次 FLOP 加 1 次 exp（exp 走 SFU）
	3. attnv: [B,S,S] x [B,S,d]: $B \times S^2 \times d$个乘加
2. memory traffic：
	1. qk: 读 2 $B\times S \times d$, 写$B\times S^2$
	2. softmax: 读写都是$B\times S^2$
		1. 看mask如何实现，如果meterialize mask,还要多读$B\times S^2$，若kernel根据index现算则没有这项overhead
	3. attnv: 读  $B\times S^2$+ $B\times S \times d$，写 $B\times S \times d$

anyway，这是一些细节，batch和dim在实际的计算里有限，我们考虑S大的量级，此时$S^2$ 支配复杂度，然后大家都batching所以B默认消掉，H同理，所以：
1. 计算：4d+5 O(S^2)
2. traffic：4xbf16 O(S^2)
所以算/存大概是d/2这个constant，从经验而论，每个head的d大概在128，但通常好的A100的ridge point在(A100)150～(H100)300。尤其是三个kernel还是串行，其中softmax最mem-bound。
3. QKᵀ / attn·V: intensity ≈ d
4. softmax: intensity ≈ 5 / 4 ≈ 1.25

## 怎么办

- recap：标准 attention 要把 $S \times S$ 的 score 矩阵写到 HBM 再读回来做 softmax，
- TLDR：FlashAttention 用 tiling 把 Q、K、V 分块搬进 SRAM，用 online softmax 在块之间增量更新 max 和分母，整个过程不把 score 矩阵写回 HBM。在这个过程中， FLOPs 没少，省的是 HBM 读写：下面的例子里，标准实现的算术强度约 63 FLOP/B，是 memory-bound；FlashAttention 把它推到约 4000 FLOP/B，变成 compute-bound。之后的 v2、v3 不再省字节，而是想办法把 tensor core 用满。


所以说这个naive attention最恶心的是softmax
1. 自己是个reduce，读进来的每个元素都用一次，严重memory bound
2. 在这个过程里引入非线性，阻碍进一步fuse，如果没有，
	1. 两个matmul可以fuse成一个gemm，which是一个非常传统的fuse trick $O = \sum_j (Q K_j^\top) V_j$
	2. 甚至还可以狂，用结合律换顺序 $(QK^\top)V = Q(K^\top V)$，把S换到规约维度，计算量从 4S²d 降到 **4Sd²**，这从计算流来论其实就是linear attention，只不过从算法角度，它有一些更好的数学性质...


softmax内部自己也有3轮 linear kernel
1. scan for max
2. scan for sum
3. per element compute


[[online-softmax]]是一篇老paper，fuse了max和sum两个scan




tiling方面的理解：

online softmax可以理解为attention的tiling吗，还是对reduce操作的并行化？
两种说法都不够准确。更贴切的理解是：online softmax 把 softmax 改写成了一个可以分块合并的 reduce。有了这个性质，FlashAttention 才能做 tiling；同样因为这个性质，这个 reduce 才能并行。所以它是这两件事的前提，本身既不是 tiling，也不是并行化。

如果没online softmax，我们如何尽我们最大努力tiling？
1. 早期fused softmax只切Q的seqlen维度，每个token的softmax互不相关，但是k不切分，所以score是[B,H,Sq,Sfull]，单个threadblock要放下 [Br, Sk]，这一项随 Sk 增长，所以 SRAM 放不下无法常驻sram，q读一遍
2. 另外一个做法是两次，pass1 求max，pass2 做标准 tiled matmul，这样每个k重复读了两遍

given online softmax这个特质，为啥softmax能和前后的gemm fuse呢？

**为什么能和前一个 GEMM（QKᵀ）融合**

S_j 这个 tile 算出来时就在寄存器 / SMEM 里。exp、rowmax、rowsum 都是 epilogue 级别的 elementwise 运算加行内 reduction，直接消费掉即可，不需要写回 HBM。

如果没有 online softmax，在发出 P 之前必须先拿到整行（长度 N）的 max 和 sum，就只能把整行 S 物化下来。N 一大，SRAM 放不下，融合就不成立了。

>这里就是可以分tile算完之后立刻用，不用写回，belike一个gemm和一个普通的elementwise也能这样fuse
>agent: **GEMM + elementwise + 一个可修正的 running row reduction**。正是"可修正"这一点让"算完立刻用"合法。


**为什么能和后一个 GEMM（PV）融合**

这是更关键的一点。PV 的 reduction 维**正好是 softmax 归一化的那一维**（KV 序列维）。一般来说，"依赖整个 reduction 维的非线性"是没法塞进 GEMM 的 K-loop 的。这里能塞进去，原因是：

1. 修正项 $e^{m-m'}$ 和 $1/\ell$ 都是**行标量**，也就是左乘一个对角矩阵：$\text{diag}(a)(PV) = (\text{diag}(a)P)V$。对角缩放和矩阵乘可交换，所以可以先乘后修，也可以先修后乘。
2. GEMM 对 P 是**线性**的，所以分块部分和可以先算，之后再统一缩放。
3. 归一化可以**推迟到最后**：FA2 就是这么做的，循环内不除 ℓ，只在最后除一次。

>维度match，把修正推迟到最后

**更抽象的说法**

$(m, \ell, \tilde o)$ 构成一个满足结合律的 monoid，merge 算子就是上面那组式子。所以沿 KV 维随便切、按任意顺序合并都对。Flash-Decoding 的 split-KV 和 FlashInfer 的 attention state merge 都是这个性质的直接应用。


**线性代数的formulation**

记号沿用 [Zihao Ye, *From Online Softmax to FlashAttention*](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)。只看 $Q$ 的第 $k$ 行，各行之间完全独立，多行时每行各维护一套 $(m, d', o')$ 即可。

| 符号 | 形状 | 含义 |
|---|---|---|
| $Q[k,:]$ | $1\times d$ | $Q$ 的第 $k$ 行 |
| $K^\top[:,i]$ | $d\times 1$ | $K^\top$ 的第 $i$ 列 |
| $V[i,:]$ | $1\times d$ | $V$ 的第 $i$ 行 |
| $x_i = Q[k,:]\,K^\top[:,i]$ | 标量 | 第 $i$ 个 score（$1/\sqrt d$ 并进 $Q$） |
| $N$ | 标量 | KV 长度 |
| $m_i$ | 标量 | 前 $i$ 个 score 的 max，$m_0=-\infty$ |
| $d'_i$ | 标量 | 前 $i$ 个的分母，以 $m_i$ 为基准，即伪代码里的 `l` |
| $o'_i$ | $1\times d$ | 前 $i$ 个的**已归一化**输出 |

目标：$O[k,:] = \sum_{i=1}^{N} \dfrac{e^{x_i}}{\sum_{j} e^{x_j}}\,V[i,:]$。约定 $e^{-\infty}=0$。

**引理 1（平移不变）** 对任意常数 $c$，

$$\frac{e^{x_i-c}}{\sum_j e^{x_j-c}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

证明：分子分母同乘 $e^c$。

取 $c = m_N$ 时所有指数 $\le 0$，不会溢出，这就是 safe softmax。代价是要先扫一遍拿到 $m_N$。

**引理 2（online 分母）** 定义 $d'_i := \sum_{j=1}^{i} e^{x_j - m_i}$，则

$$d'_i = d'_{i-1}\,e^{m_{i-1}-m_i} + e^{x_i-m_i},\qquad d'_0 = 0,\qquad d'_N = \sum_{j=1}^N e^{x_j-m_N}$$

证明见 [Online Softmax](/handson/online-softmax#逐元素递推)，思路和下面的定理 3 完全一样。$m$ 和 $d'$ 可以在同一遍里求出（online softmax）；但输出那一遍还要用 $m_N, d'_N$，所以整个 attention 仍然是 2-pass。

**定理 3（online 输出，FA1 形式）** 定义 $o'_i := \sum_{j=1}^{i} \dfrac{e^{x_j - m_i}}{d'_i}\,V[j,:]$，则

$$o'_i = o'_{i-1}\cdot\frac{d'_{i-1}\,e^{m_{i-1}-m_i}}{d'_i} + \frac{e^{x_i-m_i}}{d'_i}\,V[i,:],\qquad o'_0=0,\qquad o'_N = O[k,:]$$

证明分三步。

**第 1 步：从定义式里拆出最后一项。** 把定义里 $j=1..i$ 的求和拆成 $j=1..i-1$ 和 $j=i$ 两部分：

$$o'_i = \sum_{j=1}^{i-1} \frac{e^{x_j-m_i}}{d'_i}V[j,:] \;+\; \frac{e^{x_i-m_i}}{d'_i}V[i,:]$$

第二项就是递推式的第二项。第一项的系数用的是第 $i$ 步的 $m_i, d'_i$，所以它还不等于 $o'_{i-1}$。

**第 2 步：比较第一项的系数和 $o'_{i-1}$ 的系数。** $o'_{i-1}$ 中第 $j$ 项的系数是 $\alpha_j = \dfrac{e^{x_j-m_{i-1}}}{d'_{i-1}}$。两者相除：

$$\frac{e^{x_j-m_i}/d'_i}{e^{x_j-m_{i-1}}/d'_{i-1}} = \frac{d'_{i-1}\,e^{m_{i-1}-m_i}}{d'_i} =: c_i$$

$x_j$ 被约掉了，所以 **$c_i$ 与求和下标 $j$ 无关**：它只依赖第 $i$ 步前后的统计量 $(m_{i-1}, d'_{i-1}, m_i, d'_i)$，这几个都是已经在手上的标量，不需要回头看之前的 $x_j$ 或 $V[j,:]$。于是第一项的每个系数都可以写成 $\alpha_j \cdot c_i$。

**第 3 步：提出公因子。**

$$\sum_{j=1}^{i-1}\alpha_j\,c_i\,V[j,:] = c_i\sum_{j=1}^{i-1}\alpha_j\,V[j,:] = c_i\cdot o'_{i-1}$$

代回第 1 步，即得递推式。

这就是为什么 $o'_i$ 要定义成「只用前 $i$ 个元素、以当前的 $(m_i, d'_i)$ 为基准算出的输出」：这样前缀部分正好是上一步的 $o'_{i-1}$ 乘一个标量，递推能闭合。起作用的是「以 running 统计量为基准」，而不是「已归一化」：FA2 的 $\tilde o_i = d'_i\,o'_i$ 不归一化，递推同样闭合，只是系数变成 $e^{m_{i-1}-m_i}$（见推论 5）。

终点：$o'_N = \sum_j \dfrac{e^{x_j-m_N}}{d'_N}V[j,:]$。由引理 2，$d'_N = \sum_j e^{x_j-m_N}$；再由引理 1，这些系数就是真正的 softmax 权重，所以 $o'_N = O[k,:]$。即只要分tile 迭代，最后的结果就是最后的partial O

$c_i$ 的直觉：$o'_{i-1}$ 是按旧标准 $(m_{i-1}, d'_{i-1})$ 归一化的。换到新标准分三步：① 乘 $d'_{i-1}$，撤销旧的归一化；② 乘 $e^{m_{i-1}-m_i}$，换 max；③ 除以 $d'_i$，按新分母重新归一化。

到这里，所有量都只依赖下标 $\le i$ 的元素，一遍就能扫完，这就是 1-pass 的 FlashAttention。

::: details 数值例子：$x = (1, 3)$，$V$ 取标量 $v_1, v_2$
- $i=1$：$m_1 = 1$，$d'_1 = 1$，$o'_1 = v_1$
- $i=2$：$m_2 = 3$，$d'_2 = 1\cdot e^{1-3} + e^0 = e^{-2}+1$

$$o'_2 = v_1\cdot\frac{1\cdot e^{-2}}{e^{-2}+1} + \frac{1}{e^{-2}+1}\,v_2$$

直接计算 softmax 权重：$\dfrac{e^1}{e^1+e^3} = \dfrac{e^{-2}}{e^{-2}+1}$，$\dfrac{e^3}{e^1+e^3} = \dfrac{1}{e^{-2}+1}$，两者一致。
:::

**定理 4（tiling）** 把 KV 切成 $T = N/b$ 块，第 $t$ 块覆盖位置 $(t-1)b+1, \dots, tb$。记 $\mathbf{x}_t \in \mathbb{R}^{1\times b}$ 为这一块的 score，$V_t \in \mathbb{R}^{b\times d}$ 为对应的 $V$ 行：

$$
\begin{aligned}
\mathbf{x}_t &= Q[k,:]\,K^\top[:,(t-1)b+1:tb] \\
m_t &= \max\big(m_{t-1},\ \max_j \mathbf{x}_t[j]\big) \\
d'_t &= d'_{t-1}\,e^{m_{t-1}-m_t} + \textstyle\sum_{j=1}^{b} e^{\mathbf{x}_t[j]-m_t} \\
o'_t &= o'_{t-1}\cdot\dfrac{d'_{t-1}\,e^{m_{t-1}-m_t}}{d'_t} + \dfrac{e^{\mathbf{x}_t-m_t}}{d'_t}\,V_t
\end{aligned}
$$

则 $o'_T = O[k,:]$。

证明：记 $\mathcal{P}_t$ 为前 $t$ 块覆盖的位置集合，对 $t$ 归纳证明以下不变量：

$$m_t = \max_{p\in\mathcal{P}_t} x_p,\qquad d'_t = \sum_{p\in\mathcal{P}_t} e^{x_p-m_t},\qquad o'_t = \sum_{p\in\mathcal{P}_t}\frac{e^{x_p-m_t}}{d'_t}V[p,:]$$

- $m$：并集上的 max 等于两部分各自 max 的 max。
- $d'$ 和 $o'$：证明和引理 2、定理 3 完全相同，只是把「最后一个元素」换成「最后一块」。把 $\mathcal{P}_t$ 上的和拆成 $\mathcal{P}_{t-1}$ 上的部分和第 $t$ 块上的部分，前一部分乘上同一个与 $p$ 无关的 $c_t$。

$t=T$ 时 $\mathcal{P}_T$ 就是全部位置，再用引理 1 收尾。

块内的两个求和就是两个小 GEMM：$\mathbf{x}_t$ 是 $QK^\top$ 的一个 tile，$e^{\mathbf{x}_t - m_t}V_t$ 是一个 $(1\times b)(b\times d)$ 的 $PV$ tile。两个 GEMM 落在同一个循环里，这就是 fuse。

**推论 5（FA2：循环内不归一化）** 令 $\tilde o_t := d'_t\,o'_t$（未归一化的输出），把定理 4 中 $o'$ 的递推两边同乘 $d'_t$：

$$\tilde o_t = \tilde o_{t-1}\,e^{m_{t-1}-m_t} + e^{\mathbf{x}_t-m_t}\,V_t,\qquad O[k,:] = \tilde o_T / d'_T$$

$d'$ 从递推式里消失了，循环内不再有除法，只在最后除一次。与下文伪代码一一对应：`alpha` $= e^{m_{t-1}-m_t}$，`l` $= d'_t$，`acc` $= \tilde o_t$。


有了这个递推，一个 Q 块（$B_r$ 行）可以常驻 SRAM，让 K/V 一块一块流过，每块的 score 只是 $B_r \times B_c$ 的小 tile，算完就丢。前向伪代码（v2 的循环顺序，下节解释），单个 head：

```python
# fused attention, directly compute O based on QKV
def flash_attn_fwd(Q, K, V, Br=128, Bc=128, causal=True):
    # batch and head is parallelized internally
    # Q: [S,d]
    S, d = Q.shape 
    # allocate output and lse for backprop
    # lse：softmax 分母取对数                            
    O, LSE = torch.empty_like(Q), Q.new_empty(S)
    # tiled Q: [Br,d]
    for q0 in range(0, S, Br):
        # load current q tile from HBM to SRAM
        q = load(Q[q0:q0+Br])
        # m, d',o'
        m = Q.new_full((Br,), -torch.inf)
        l = Q.new_zeros(Br)
        acc = Q.new_zeros(Br, d)
        # compute [0,k_end] if causal(skip upper),
        # else full attention without mask
        # 这里是可视化里面的决定k的右end
        k_end = q0 + Br if causal else S
        # iterate kv for q, compute o
        for k0 in range(0, k_end, Bc):
            k, v = load(K[k0:k0+Bc]), load(V[k0:k0+Bc])
            s = q @ k.mT * d**-0.5              # [Br, Bc] tile，不写回 HBM
            if causal and k0 + Bc > q0:         # 只有对角 tile 需要逐元素 mask
                i = torch.arange(q0, q0 + Br)[:, None]
                j = torch.arange(k0, k0 + Bc)[None, :]
                allowed = j <= i
                s = torch.where(allowed, s, -torch.inf)
            m_new = torch.maximum(m, s.amax(dim=-1)) # update max
            alpha = torch.exp(m - m_new) # update d using this
            # partial attention score with new m
            p = torch.exp(s - m_new[:, None])
            # d' for final reduce
            l = alpha * l + p.sum(dim=-1)
            # O'
            acc = alpha[:, None] * acc + p @ v
            m = m_new
        O[q0:q0+Br] = acc / l[:, None]          # 最后只除一次
        LSE[q0:q0+Br] = m + torch.log(l)        # 存 logsumexp，反向用来重算 P
    return O, LSE
```

`k_end` 的由来：causal 下第 $i$ 个 query 只能看 $j \le i$ 的 key。当前 Q 块最靠后的一行是 `q0+Br-1`，所以 `j ≥ q0+Br` 的 key 对块内所有行都被 mask，整块跳过，不加载也不计算。以 $S = 512$、$B_r = B_c = 128$ 为例，共 4×4 个 tile：

```
              k块0   k块1   k块2   k块3
             0-127 128-255 256-383 384-511
q块0  0-127   [◣]    ✗      ✗      ✗      k_end = 128 → 算 1 块
q块1 128-255  [■]   [◣]     ✗      ✗      k_end = 256 → 算 2 块
q块2 256-383  [■]   [■]    [◣]     ✗      k_end = 384 → 算 3 块
q块3 384-511  [■]   [■]    [■]    [◣]     k_end = 512 → 算 4 块

■ 整块可见：直接算，不用 mask
◣ 对角块：一半可见，逐元素 mask（伪代码里 k0 + Bc > q0 那段）
✗ 整块不可见：k_end 直接跳过
```

16 个 tile 只算 10 个；一般地是 $\frac{n(n+1)/2}{n^2}$（$n = S/B_r$），$S$ 越大越接近一半。这里默认 $B_r = B_c$，对角区域正好落在一个 K 块上；不相等时对角区域会横跨多个 K 块，`k0 + Bc > q0` 的判断会把它们都加上 mask，结果仍然正确。

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

伪代码：

```python
# split-KV (Flash-Decoding): 在 FA2 的基础上，再沿 KV 维切成 num_splits 段
# kernel 1: grid = (Q 块) x (split)，每个 (q0, sp) 一个 thread block，互相独立
# kernel 2: 把各段的 (O_part, LSE_part) 合并成最终的 O
def flash_attn_fwd_splitkv(Q, K, V, Br=128, Bc=128, num_splits=4, causal=True):
    # Q: [S_q, d], K/V: [S_k, d]；causal 时假设 S_q == S_k（prefill）
    S_q, d = Q.shape
    S_k = K.shape[0]
    # 每段的长度，向上取整到 Bc 的倍数
    seg = ceil(ceil(S_k / num_splits) / Bc) * Bc

    # 中间结果：每个 split 一份，要写回 HBM（比 FA2 多出来的开销）
    # O_part 每段内部已经归一化，LSE_part 是每段自己的 log 分母
    O_part   = torch.zeros(num_splits, S_q, d, dtype=torch.float32)
    LSE_part = torch.full((num_splits, S_q), -torch.inf)

    # ---------------- kernel 1 ----------------
    # 下面两层循环在 GPU 上是 grid 的两个维度，并行执行
    for q0 in range(0, S_q, Br):
        for sp in range(num_splits):
            # 这一段负责的 KV 区间 [kv_start, kv_end)
            kv_start = sp * seg
            kv_end   = min(kv_start + seg, S_k)
            if causal:
                kv_end = min(kv_end, q0 + Br)    # 对角线右边的部分跳过；可能整段为空

            q = load(Q[q0:q0+Br])
            # 和 FA2 完全一样的 running state：m, d', õ
            m   = torch.full((Br,), -torch.inf)
            l   = torch.zeros(Br)
            acc = torch.zeros(Br, d)

            # 只扫自己这一段 KV，而不是 [0, k_end)
            for k0 in range(kv_start, kv_end, Bc):
                k1 = min(k0 + Bc, kv_end)
                k, v = load(K[k0:k1]), load(V[k0:k1])
                s = q @ k.mT * d**-0.5
                if causal and k1 > q0:            # 对角 tile 逐元素 mask
                    i = torch.arange(q0, q0 + Br)[:, None]
                    j = torch.arange(k0, k1)[None, :]
                    s = torch.where(j <= i, s, -torch.inf)
                m_new = torch.maximum(m, s.amax(dim=-1))
                alpha = torch.exp(m - m_new)      # 实际 kernel 要特判 m_new == -inf
                p = torch.exp(s - m_new[:, None])
                l = alpha * l + p.sum(dim=-1)
                acc = alpha[:, None] * acc + p @ v
                m = m_new

            # 和 FA2 的区别：写回的是“这一段”的结果，不是最终结果
            # 某些行可能在这一段里一个 key 都看不到（l == 0），要置成 O=0、LSE=-inf
            valid = l > 0
            O_part[sp, q0:q0+Br]   = torch.where(valid[:, None], acc / l[:, None], 0)
            LSE_part[sp, q0:q0+Br] = torch.where(valid, m + torch.log(l), -torch.inf)

    # ---------------- kernel 2: combine ----------------
    # 总分母取 log：LSE = log Σ_sp exp(LSE_sp)，内部同样要先减 max 才安全
    LSE = torch.logsumexp(LSE_part, dim=0)               # [S_q]
    # 第 sp 段的权重 = 它的分母占总分母的比例
    #   e^{LSE_sp - LSE} = e^{m_sp - m} * l_sp / l  （换 max 和归一化两个修正合成了一个）
    w = torch.exp(LSE_part - LSE[None, :])                # [num_splits, S_q]，各段之和为 1
    O = (w[:, :, None] * O_part).sum(dim=0)               # [S_q, d]
    return O, LSE
```

为啥non-splitk不需要这个reduce kernel？
1. non-splitk在循环里处理kv的全部tile，这样边循环边update o‘，最终结果就是o
2. splitk里，kv的切片被划分到了不同的thread block里，他们之间本身要同步，唯一的办法就是barrier，这隐式通过启动一个kernel来做到

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
