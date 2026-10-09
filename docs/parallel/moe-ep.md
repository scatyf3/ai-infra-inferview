---
title: MoE 与 Expert Parallel
status: draft
tags: [moe, ep, all-to-all]
difficulty: 4
order: 5
related: [/parallel/parallelism-overview, /parallel/collective-comm, /parallel/comm-overlap, /inference/prefill-decode-roofline]
stack: [d-intra, f-model]
---

# MoE 与 Expert Parallel

> 路由、all-to-all、专家负载不均、EP 与 TP 混用

## 一句话结论

MoE 把一个大 FFN 换成 $E$ 个小 FFN（专家），每个 token 只走其中 top-$k$ 个：**显存按总参数算，FLOPs 按激活参数算**。代价是每个专家只分到 $B k / E$ 个 token，decode 时算术强度被稀释 $E/k$ 倍，所以 MoE 推理要把很多卡的 token 凑到一起喂专家。Expert Parallel（EP）就是干这个的：专家分散到各卡，token 用 all-to-all 发过去（dispatch）、算完再发回来（combine）。三个核心麻烦：all-to-all 通信、专家负载不均、变长分组的 GEMM。

## 推导

### 第 0 步：MoE 层长什么样

Transformer 的 FFN（SwiGLU）是 $\text{FFN}(x) = W_2\big(\text{SiLU}(W_1 x) \odot W_3 x\big)$。MoE 层把它换成：

$$
y = \sum_{j=1}^{N_s} \text{FFN}^{s}_j(x) \;+\; \sum_{i \in \text{TopK}(s(x),\,k)} g_i(x)\,\text{FFN}_i(x)
$$

- **专家（expert）**：就是一个普通 FFN，形状和 dense FFN 一样或更窄。$E$ 个路由专家各有一套 $W_1, W_2, W_3$。
- **路由器 / gate**：一个 $d \times E$ 的线性层，算出每个 token 对每个专家的分数 $s(x) = \sigma(x W_g)$。经典做法用 softmax（[Switch](https://arxiv.org/abs/2101.03961)、[Mixtral](https://arxiv.org/abs/2401.04088)），[DeepSeek-V3](https://arxiv.org/abs/2412.19437) 用 sigmoid 再在 top-$k$ 内归一化。
- **top-$k$**：每个 token 只选分数最高的 $k$ 个专家，输出按 $g_i$ 加权求和。Switch 用 $k=1$，Mixtral 用 $k=2$，DeepSeek-V3 用 $k=8$。
- **共享专家（shared expert）**：每个 token 都必经的专家，不参与路由，负责学通用知识，让路由专家更专门化（[DeepSeekMoE](https://arxiv.org/abs/2401.06066)）。

MoE 最早在 [GShard](https://arxiv.org/abs/2006.16668) 里和 EP 一起大规模用起来：专家天然按卡切分，token 用 all-to-all 搬运。

### 第 1 步：总参数 vs 激活参数

| 模型 | 路由专家 $E$ | top-$k$ | 共享专家 | 总参数 | 激活参数 |
|---|---|---|---|---|---|
| [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) | 8 | 2 | 0 | 47B | 13B |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 256 | 8 | 1 | 671B | 37B |

自己算一遍 DeepSeek-V3（数据来自技术报告）：$d = 7168$，专家中间维度 2048，61 层里前 3 层是 dense，其余 58 层是 MoE。

- 一个专家：$3 \times 7168 \times 2048 \approx 44\text{M}$ 参数，FP8 下 44 MB。
- 路由专家总量：$58 \times 256 \times 44\text{M} \approx 654\text{B}$，占了 671B 的绝大部分。
- 每 token 激活：$58 \times (8 + 1) \times 44\text{M} \approx 23\text{B}$，加上 attention、dense 层、embedding 等，到 37B。

Mixtral 为什么是 47B 不是 $8 \times 7 = 56$B：只有 FFN 复制了 8 份，attention 和 embedding 是共享的。

**含义**：计算量像一个 37B 模型，显存像一个 671B 模型。单机 8×80 GB 放不下 FP8 的 671B 权重再加 KV，必须多机，必须分专家。

### 第 2 步：为什么 MoE decode 要很大的 batch

回顾 [Roofline](/inference/prefill-decode-roofline)：decode 时 GEMM 的 $m$ 维就是 token 数，权重读一次只算 $m$ 个 token，bf16 下 $\text{AI} \approx m$，FP8 权重下 $\text{AI} \approx 2m$。H100 的 ridge 约 295（bf16），所以 dense 模型 batch 到 ~300 才开始 compute-bound。

MoE 里一个 step 有 $T$ 个 token，每个选 $k$ 个专家，均匀路由时**每个专家分到**：

$$
m_{\text{expert}} = \frac{T \cdot k}{E}
$$

专家 GEMM 的算术强度也按 $m_{\text{expert}}$ 算，比 dense 低 $E/k$ 倍。DeepSeek-V3 的 $E/k = 32$。

**更糟的是：小 batch 下几乎所有专家都会被点到。** 每个 token 选中某专家的概率是 $k/E$，$T$ 个 token 都没选它的概率是 $(1 - k/E)^T$。被激活的专家数期望：

$$
E_{\text{active}} = E\left[1 - (1 - k/E)^T\right]
$$

算一笔（DeepSeek-V3，$E = 256$，$k = 8$）：

| 场景 | $T$ | 每专家 token 数 $Tk/E$ | 被激活专家占比 | bf16 AI（相对 ridge 295） |
|---|---|---|---|---|
| 单机，batch 64 | 64 | 2 | $1 - 0.969^{64} \approx 87\%$ | 2，低 ~150 倍 |
| 单机，batch 512 | 512 | 16 | ≈ 100% | 16，低 ~18 倍 |
| EP128，每卡 128 请求 | $128 \times 128 = 16384$ | 512 | 100% | 512，越过 ridge |

第一行的意思是：为了 64 个 token，要从 HBM 读近 87% 的专家权重（几百 GB 量级），却只做 64 个 token 的计算。dense 模型 batch 64 时 AI 是 64，MoE 只有 2。

**结论**：$m_{\text{expert}}$ 要达到几百，总 token 数 $T$ 要达到 $300 \cdot E / k \approx 10^4$。单卡 KV 显存撑不住这么大的 batch，只能**让很多卡各自跑一部分请求（DP attention），再把 token 汇聚到专家所在的卡**。这就是大 EP 的根本动机。第三行的配置来自 DeepSeek 公开的 [profile-data](https://github.com/deepseek-ai/profile-data)（decode：EP128，每卡 128 请求）。

一般化：$N$ 张卡、每卡贡献 $b$ 个 token、每卡放 $E/N$ 个专家，则

$$
m_{\text{expert}} = \frac{N b k}{E}
$$

卡数 $N$ 越大，每个专家能聚到的 token 越多，而每卡要读的专家权重是 $1/N$。这是 EP 相对单机部署的双重收益。

### 第 3 步：EP 的 dispatch / combine

DP attention + EP 的布局见 [并行总览](/parallel/parallelism-overview) 的 DP attention 一节（含与 AF 分离的对比表），这里只看 MoE 层内部：

1. **router**：本卡的 $b$ 个 token 各选 $k$ 个专家。
2. **permute**：把 $(token, 专家)$ 对按目标专家排序，同一目标卡的行连续。
3. **dispatch（all-to-all）**：每卡把属于别卡专家的 token 发过去。
4. **专家计算**：每卡收到的 token 按本地专家分组，做 grouped GEMM。
5. **combine（all-to-all）**：结果按原路发回。
6. **unpermute + 加权求和**：每个 token 把 $k$ 份结果乘 $g_i$ 加起来，再加共享专家输出。

**通信量**（每卡每层，上界，不去重）：

$$
V_{\text{dispatch}} = b \cdot k \cdot d \cdot \text{bytes}_{\text{disp}}, \quad V_{\text{combine}} = b \cdot k \cdot d \cdot \text{bytes}_{\text{comb}}
$$

其中大约 $(N-1)/N$ 要出本卡。和 TP 比：TP 每层 all-reduce 约 $2 b d \cdot \text{bytes}$ 且与 $k$ 无关，EP 正比于 $k$。

算一笔：DeepSeek-V3 decode，每卡 $b = 128$，$k = 8$，$d = 7168$，dispatch FP8（1 B）、combine BF16（2 B）：

- dispatch：$128 \times 8 \times 7168 \approx 7.3$ MB，combine 约 14.7 MB。
- 走 IB，按每卡约 50 GB/s 算：dispatch ≈ 0.15 ms，combine ≈ 0.29 ms，一层约 0.44 ms。
- 58 个 MoE 层：约 25 ms/step。不藏起来的话，光通信就把 TPOT 吃满了。

对照：[DeepEP](https://github.com/deepseek-ai/DeepEP/tree/v1.2.1) 低延迟 kernel 在 H800 + 400 Gb/s IB、每批 128 token、top-8 下，EP256 测得 dispatch 194 µs、combine 360 µs（作者自测），和上面的估算同量级。

**为什么对拓扑敏感**：

- all-to-all 是**每对卡之间**都有流量，速度取决于最慢的那段链路。节点内 NVLink、节点间 IB，DeepSeek-V3 报告的 H800 上是 160 GB/s vs 50 GB/s（[技术报告](https://arxiv.org/abs/2412.19437)），差 3.2 倍。
- 消息大小取决于路由结果，运行时才知道；某个专家热门时，流向它的链路成为热点。
- 对策一：**node-limited routing**。DeepSeek-V3 限制每个 token 最多发往 $M = 4$ 个节点，先选节点再在节点内选专家，跨节点流量从 $k d$ 降到最多 $M d$。
- 对策二：**分层转发**。同一 token 去同一节点的多个专家时，IB 上只发一份，到对端节点再走 NVLink 分发。DeepEP 的 normal kernel 就是 NVLink + RDMA 转发（[README](https://github.com/deepseek-ai/DeepEP/tree/v1.2.1)）。

### 第 4 步：负载不均与对策

step 时间由**最忙的那张卡**决定。定义不均衡度 $\text{max load} / \text{mean load}$，它是 1.5 就意味着平均有 1/3 的时间别的卡在空等。

| 手段 | 阶段 | 做法 | 代价 |
|---|---|---|---|
| 辅助 loss | 训练 | [Switch](https://arxiv.org/abs/2101.03961)：$\mathcal{L}_{aux} = \alpha E \sum_i f_i P_i$，$f_i$ 是实际分到专家 $i$ 的 token 比例，$P_i$ 是路由概率均值，$\alpha = 10^{-2}$ | 和主 loss 抢梯度，$\alpha$ 大了伤效果 |
| 无辅助 loss 的 bias 均衡 | 训练 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437)：给每个专家一个 bias $b_i$，**只加在选 top-$k$ 的分数上，不影响 $g_i$**；每步结束后过载专家 $b_i -= \gamma$，欠载 $b_i += \gamma$（$\gamma = 0.001$），另配 $\alpha = 10^{-4}$ 的序列级小 loss 防极端情况 | 基本不伤主任务梯度 |
| capacity factor + 丢 token | 训练/推理 | 每专家最多收 $C = \text{CF} \cdot T k / E$ 个 token，超出的跳过专家、只走残差（[Switch](https://arxiv.org/abs/2101.03961)，作者报告通常 <1% 被丢） | 推理时结果依赖同 batch 的其他请求，影响质量；DeepSeek-V3 训练和推理都不丢 token |
| 冗余专家（EPLB） | 推理 | 按历史负载统计复制热门专家，重新摆放到各卡（[EPLB](https://github.com/deepseek-ai/EPLB)）：节点数整除专家组数时用分层策略（先分组到节点，再节点内复制，适合 prefill 小 EP），否则用全局策略（适合 decode 大 EP） | 额外显存；负载分布漂移后要重排 |

DeepSeek-V3 的推理部署（技术报告）：prefill 最小单元 4 节点 32 卡、MoE 用 EP32，额外 32 个冗余专家（每卡 8 + 1），约每 10 分钟按统计重新确定；decode 最小单元 40 节点 320 卡、EP320，每卡 1 个专家，另 64 张卡放冗余专家和共享专家。SGLang 复现时 EPLB 带来 prefill 1.49×、decode 2.54× 加速（[LMSYS 博客](https://lmsys.org/blog/2025-05-05-large-scale-ep/)，作者自测）。

注意：bias 均衡、辅助 loss 改变的是**训练出来的路由分布**；EPLB 改变的是**部署时专家到卡的映射**。推理时短时间内的局部热点只能靠后者。

### 第 5 步：grouped GEMM

每张卡上有若干本地专家，每个专家收到的 token 数不同且运行时才知道。朴素做法是 for 循环逐专家调 GEMM，每个都很小，launch 开销和 SM 空闲很严重。**grouped GEMM** 把多个形状 $(m_e, K) \times (K, N)$、$m_e$ 不同的 GEMM 合成一个 kernel，所有专家共享 $N, K$，只按 $m$ 维分组。

[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 提供两种布局：

| 布局 | 用于 | 做法 |
|---|---|---|
| contiguous | 训练、prefill | 所有专家的 token 拼成一个张量，每段按 block 大小对齐，形状动态 |
| masked | decode | 每专家预留固定容量，用 mask 表示有效行数，CPU 不需要知道每专家的 token 数，可以进 CUDA graph |

### 第 6 步：DeepEP 两类 kernel 与 two-batch overlap

[DeepEP](https://github.com/deepseek-ai/DeepEP/tree/v1.2.1)（v1.2.1 README）把 dispatch/combine 分成两套：

| | normal kernel | low-latency kernel |
|---|---|---|
| 用于 | 训练、prefill | decode |
| 目标 | 吞吐 | 延迟 |
| 传输 | 节点内 NVLink，跨节点 RDMA → NVLink 转发 | 纯 RDMA（后来也尽量用 NVLink） |
| 形状 | 动态，需 CPU 同步拿到接收 token 数 | 固定容量 buffer，兼容 CUDA graph |
| 与计算 overlap | 可限制占用的 SM 数 | receive hook：RDMA 在后台收，不占 SM |
| 实测（作者自测，H800） | EP8 节点内约 153 GB/s；EP32 跨节点约 58 GB/s | 128 token/批：EP8 dispatch 77 µs，EP256 194 µs |

SGLang 的说法是：同一个通信组里不能两种模式混用，所以要配合 PD 分离，prefill 集群用 normal + contiguous grouped GEMM，decode 集群用 low-latency + masked grouped GEMM（[LMSYS 博客](https://lmsys.org/blog/2025-05-05-large-scale-ep/)）。

**two-batch overlap（TBO）**：第 3 步算出来通信约 25 ms/step，必须藏。把一个 batch 切成两个 micro-batch A、B，A 做 attention / 专家计算时 B 做 dispatch / combine，交替进行。DeepSeek 在 [profile-data](https://github.com/deepseek-ai/profile-data) 里公开了训练、prefill（EP32，每卡 16K token）、decode（EP128，每卡 128 请求）三种场景的双 micro-batch 时间线；decode 里 all-to-all 不占 SM，RDMA 发出后 SM 就释放。SGLang 报告 TBO 在 prefill 提升 27%~35%；decode 每卡 token 少于约 64~128 时反而变慢（32 token/卡时 −27%），256 token/卡时 +25.5%（作者自测）。原因很简单：batch 太小时计算本身很短，藏不住通信，切成两半还让每半的 GEMM 更小。

### 第 7 步：专家用 EP 还是 TP

| | 专家 TP（每卡持有每个专家的 $1/N$ 片） | 专家 EP（每卡持有完整的 $E/N$ 个专家） |
|---|---|---|
| 通信 | all-reduce / reduce-scatter，规则，与路由无关 | all-to-all，大小由路由决定 |
| 负载均衡 | 天然均衡，每卡算每个专家的同等一片 | 热门专家所在卡成瓶颈，需要 EPLB |
| GEMM 形状 | 中间维被切细（2048 / 8 = 256），GEMM 很瘦 | 专家完整，GEMM 形状不变 |
| 扩展范围 | 受 NVLink 域限制（≤ 8） | 可跨节点到几百卡 |
| 每卡权重 | 所有专家的 $1/N$ | $E/N$ 个专家 |
| 适合 | 专家少而大（如 Mixtral 8 专家）、单机部署 | 专家多而细（DeepSeek-V3 256 专家）、多机大 batch |

两者的 $m_{\text{expert}}$ 都等于「共享这组专家的 token 总数 × $k/E$」，切法本身不改变 AI。EP 的优势是**能把共享的卡数做到几百**，从而把 $T$ 做大；TP 被 NVLink 域卡死。实际系统常混合：attention 用 DP（或小 TP），专家用大 EP；DeepSeek-V3 prefill 是 attention TP4 + SP + DP8、MoE EP32（[技术报告](https://arxiv.org/abs/2412.19437)）。

## 面试追问

::: details Q：MoE 推理为什么说 batch 要很大才划算？
每个专家只分到 $T k / E$ 个 token，算术强度比同 batch 的 dense 模型低 $E/k$ 倍（DeepSeek-V3 是 32 倍）。同时小 batch 下被激活的专家占比 $1 - (1-k/E)^T$ 就已接近 1（$T=64$ 时约 87%），权重几乎全读一遍，只算几个 token。要让每专家 token 数到 ridge（~300），总 token 数要到 $300 E / k \approx 10^4$，单卡 KV 撑不住，只能多卡 DP attention 汇聚 token 到大 EP 上。
:::

::: details Q：估算一次 EP dispatch 的通信量。
每卡 $b \cdot k \cdot d \cdot \text{bytes}$，约 $(N-1)/N$ 出卡。DeepSeek-V3 decode $b=128, k=8, d=7168$、FP8：约 7.3 MB；combine 用 BF16 再翻倍。按 50 GB/s IB 一层合计约 0.44 ms，58 层约 25 ms，所以要 node-limited routing 减少跨节点份数，并用 two-batch overlap 藏起来。
:::

::: details Q：aux loss 和 DeepSeek-V3 的 aux-loss-free 有什么区别？
aux loss 把均衡目标加进 loss，通过梯度改路由器，系数大了会干扰主任务。aux-loss-free 给每个专家一个不参与梯度的 bias，只用于选 top-$k$，不改变加权系数 $g_i$；每步按负载用固定步长 $\gamma$ 调 bias。均衡由外部控制回路完成，主任务梯度基本不受影响。V3 仍保留一个极小（$\alpha=10^{-4}$）的序列级 loss 兜底。
:::

::: details Q：推理时为什么不爱用 capacity factor 丢 token？
丢不丢取决于同一个 batch 里其他请求怎么路由，同一个请求在不同 batch 里输出不同，质量也受损。推理更倾向不丢 token，用 EPLB 冗余热门专家来压低最大负载；DeepSeek-V3 训练和推理都不丢 token。
:::

::: details Q：decode 为什么要单独的 low-latency all-to-all？
decode 每卡 token 少，通信是延迟主导而非带宽主导；还要进 CUDA graph 去掉 CPU launch 开销，就不能有「CPU 等接收数量」这种同步。low-latency kernel 用固定容量 buffer + 纯 RDMA，配 masked grouped GEMM，形状静态；receive hook 不占 SM，计算和通信能重叠。
:::

::: details Q：共享专家有什么用？对系统有什么影响？
算法上：吸收所有 token 都要的通用知识，让路由专家更专门化（DeepSeekMoE）。系统上：它不需要 all-to-all，可以在本卡和 dispatch 并行计算；但它对每个 token 都激活，相当于一个小 dense FFN，DeepSeek-V3 decode 把它当成「必选的路由专家」，每 token 选 9 个专家统一调度。
:::

## 手撕

常见题：写 top-$k$ gating（含 Switch aux loss）；写 EP 下的 MoE forward；估算 all-to-all 通信量和每专家 token 数。

```python
def moe_forward_ep(x, W_g, bias, W1, W3, W2, shared_ffn, k, E, group):
    # x: [T, d] 本卡的 token（DP attention 的输出）
    # W1/W3/W2: 本卡的 E_loc 个专家权重，[E_loc, d, f] / [E_loc, f, d]
    R = group.size(); E_loc = E // R          # 专家 e 放在 rank e // E_loc
    T, d = x.shape

    # 1. router
    s = torch.sigmoid(x @ W_g)                # [T, E]（Mixtral 用 softmax）
    idx = torch.topk(s + bias, k, dim=-1).indices   # bias 只用于选专家
    g = s.gather(-1, idx); g = g / g.sum(-1, keepdim=True)   # [T, k]

    # 2. permute：(token, 专家) 对按专家 id 排序
    flat_e = idx.flatten()                    # [T*k]
    flat_t = torch.arange(T).repeat_interleave(k)
    order = torch.argsort(flat_e, stable=True)
    x_perm = x[flat_t[order]]                 # [T*k, d]
    send = torch.bincount(flat_e[order] // E_loc, minlength=R)   # 发给每个 rank 的行数

    # 3. dispatch all-to-all：先换计数，再换数据和专家 id
    recv = all_to_all(send, group)
    x_in = all_to_all_v(x_perm, send, recv, group)          # [M, d]
    e_in = all_to_all_v(flat_e[order], send, recv, group) % E_loc

    # 4. 本地按专家分组 + grouped GEMM
    o2 = torch.argsort(e_in, stable=True)
    cnt = torch.bincount(e_in, minlength=E_loc)             # 每个本地专家的 token 数
    h = F.silu(grouped_gemm(x_in[o2], W1, cnt)) * grouped_gemm(x_in[o2], W3, cnt)
    y_sorted = grouped_gemm(h, W2, cnt)
    y_in = torch.empty_like(y_sorted); y_in[o2] = y_sorted  # 恢复接收顺序

    # 5. combine all-to-all：计数反过来
    y_perm = all_to_all_v(y_in, recv, send, group)          # [T*k, d]

    # 6. unpermute + 加权求和 + 共享专家（可与 dispatch 重叠）
    out = torch.zeros(T, d)
    out.index_add_(0, flat_t[order], y_perm * g.flatten()[order, None])
    return out + shared_ffn(x)
```

Switch aux loss（$k=1$ 时）：

```python
probs = softmax(x @ W_g, -1)                 # [T, E]
f = bincount(probs.argmax(-1), minlength=E) / T   # 实际分配比例，不可导
P = probs.mean(0)                            # 路由概率均值，可导
aux = alpha * E * (f * P).sum()              # 均匀时 = alpha
```

## 参考

- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)：top-1 路由、aux loss、capacity factor
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)：细粒度专家、共享专家
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：aux-loss-free 均衡、node-limited routing、推理部署
- [DeepEP](https://github.com/deepseek-ai/DeepEP/tree/v1.2.1)：normal / low-latency all-to-all kernel（当前 main 分支 API 已改版，本文按 v1.2.1）
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)：contiguous / masked grouped GEMM
- [EPLB](https://github.com/deepseek-ai/EPLB)：冗余专家与放置
- [DeepSeek profile-data](https://github.com/deepseek-ai/profile-data)：双 micro-batch overlap 时间线
- [SGLang：96 张 H100 上的 PD 分离 + 大规模 EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
