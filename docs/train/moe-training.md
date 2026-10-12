---
title: MoE 训练：路由怎么稳、负载怎么均、为什么 MFU 更低
status: draft
tags: [moe, load-balancing, router, z-loss, mfu]
difficulty: 4
order: 8.5
related: [/parallel/moe-ep, /parallel/comm-overlap, /train/compute-mfu, /train/pipeline-schedules, /train/parallel-config, /train/stability, /train/fp8-training, /posttrain/training-memory]
stack: []
---

# MoE 训练：路由怎么稳、负载怎么均、为什么 MFU 更低

> 训练视角的 MoE：算力账和显存账分开算，负载不均直接变成墙钟时间，router 是全模型最容易数值出事的地方

## 一句话结论

MoE 训练的账要分两本：**算力按激活参数**（DeepSeek-V3 每 token $6 \times 37\text{B}$ FLOPs），**显存和梯度同步按总参数**（671B × 16 字节 ≈ 10.7 TB 的模型状态）。负载不均在训练里直接变成时间：EP 组里最忙的那张卡决定这一层多久，一个专家拿到 2 倍平均 token，就能让整层慢 25%。早期做法（Switch / GShard）给每个专家定容量 $C = \text{CF}\cdot Tk/E$，超出就丢 token；DeepSeek-V3 改成 aux-loss-free 的 bias 调节（每步 $\pm\gamma$，$\gamma = 0.001$）加 node-limited routing（每 token 最多去 4 个节点），训练和推理都不丢 token。路由稳定性靠 router 用 fp32 和 router z-loss（ST-MoE，系数 $10^{-3}$）。MFU 低的主因是跨节点 all-to-all（V3 报告计算 : 通信约 1 : 1）和负载不均；按 V3 公开的 GPU 小时反推，每卡约 343 TFLOP/s（只算 $6N$），约为 H800 bf16 dense 峰值的 35%、fp8 峰值的 17%（本页推算）。

MoE 层的结构、EP 的 dispatch / combine、grouped GEMM、DeepEP 都在 [MoE 与 EP](/parallel/moe-ep)，本页不重复。

## 推导

### 0. 记号

| 符号 | 含义 | 例子（DeepSeek-V3） |
|---|---|---|
| $E$ | 每层路由专家数 | 256 |
| $k$ | 每个 token 选的路由专家数（top-$k$） | 8 |
| $N_{\text{total}}$ | 总参数 | 671 B |
| $N_{\text{active}}$ | 每个 token 激活的参数 | 37 B |
| $T$ | 一次路由涉及的 token 数（一个 batch 或一个 group） | 4096 |
| $c_i$ | 这批 token 里分给专家 $i$ 的 (token, 专家) 对数 | 平均 $Tk/E = 128$ |
| CF | capacity factor，每个专家容量相对平均值的倍数 | Switch 常用 1.0–1.25 |
| $C$ | 每个专家的容量（最多收多少 token） | $\text{CF}\cdot Tk/E$ |
| $s_{i,t}$ | token $t$ 对专家 $i$ 的亲和度分数 | V3 用 sigmoid |
| $b_i$ | 专家 $i$ 的路由 bias（aux-loss-free 用） | 初值 0 |
| $\gamma$ | bias 每步的更新步长 | 0.001 |
| $\text{EP}$ | expert parallel 度，一组专家分到多少张卡 | 64 |
| $M$ | 每个 token 最多发往的节点数 | 4 |

单位：1 B = $10^9$，GB = $10^9$ 字节，TB = $10^{12}$ 字节。V3 的数字都来自 [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)：61 层（前 3 层 dense），$d = 7168$，每个专家中间维 2048，1 个共享专家，预训练 14.8T token，序列长度 4K。

### 1. 训练的两本账：算力按激活，显存与同步按总量

**算力**。每 token 训练 FLOPs $\approx 6N$（forward 2、backward 4：backward 对输入和对权重各一次同样大的矩阵乘，见 [训练算力账](/train/compute-mfu)）。每个 token 只经过被选中的 $k$ 个专家，没被选中的专家权重不参与这个 token 的矩阵乘，所以 $N$ 取 $N_{\text{active}}$：

$$
C \approx 6 N_{\text{active}} D
$$

V3：$6 \times 3.7\times10^{10} = 2.22\times10^{11}$ FLOPs / token，14.8T token 共 $3.29\times10^{24}$ FLOPs。

**显存**。优化器要给**每个**参数存主权重和 Adam 状态，不管这一步它有没有被用到。按混合精度 Adam 每参数 16 字节（bf16 参数 2 + bf16 梯度 2 + fp32 主权重 4 + $m$ 4 + $v$ 4，见 [训练显存账](/posttrain/training-memory)）：

| | 参数 | 模型状态（× 16 字节） | 至少要几张 80 GB 卡才放得下 |
|---|---|---|---|
| V3 总参数 | 671 B | 10.7 TB | 135 |
| 同算力的 dense（37 B） | 37 B | 0.59 TB | 8 |

差 18 倍。V3 实际把 Adam 的 $m, v$ 降到 bf16、主权重和梯度累加仍是 fp32（[FP8 训练](/train/fp8-training)），能省一些，但量级不变。

**同步**。DP 的梯度同步、ZeRO 的分片也都按参数个数算，所以每步要同步的梯度总量是 671B 规模，不是 37B。再加上 dense 模型没有的 all-to-all（每 token 每层 dispatch、combine 各发 $k$ 份 hidden，见 [MoE 与 EP](/parallel/moe-ep#第-3-步-ep-的-dispatch-combine)）。

所以 MoE 的卖点是「用 37B 的算力训一个 671B 容量的模型」，代价全落在显存、通信和下面的负载均衡上。

### 2. 负载不均的代价：最慢的 EP rank 决定时间

一个 MoE 层的专家计算，EP 组内每张卡并行跑自己的专家，下一次 all-to-all 要等所有卡算完。所以这一层的耗时由**负载最重的那张卡**决定：

$$
\text{专家计算效率} = \frac{\text{平均每卡负载}}{\text{最大每卡负载}}
$$

算一笔：$E = 256$，EP 64，每卡 4 个专家；$T = 4096$，$k = 8$，平均每个专家 $Tk/E = 128$ 对。

1. 均匀时每卡 $4 \times 128 = 512$。
2. 某个专家拿到 2 倍平均，256 对，它所在的卡是 $256 + 3 \times 128 = 640$。
3. 其余 63 张卡算完 512 后空等，这一层专家计算效率 $512 / 640 = 80\%$。这张卡的 all-to-all 接收量也多 25%。

**容量与丢 token**（[Switch](https://arxiv.org/abs/2101.03961) 式 (3)，[GShard](https://arxiv.org/abs/2006.16668) 同一思路）：为了让张量形状静态、让最忙的卡不至于太慢，每个专家最多收 $C$ 个，超出的 token 跳过这个专家，只走残差连接传到下一层。top-$k$ 时按 (token, 专家) 对计数（Megatron-Core 的 `get_capacity` 就是用 $T \cdot k$ 去算）：

$$
C = \left\lceil \text{CF}\cdot\frac{Tk}{E} \right\rceil, \qquad
\text{drop 率} = \frac{\sum_i \max(0,\ c_i - C)}{Tk}
$$

接着上面的例子，$\text{CF} = 1.25$：$C = 1.25 \times 128 = 160$。热门专家收 256 对，丢 96 对，drop 率 $96 / 32768 = 0.29\%$。代价两头都有：

1. CF 越大，丢得越少，但每个专家按 $C$ 预留的 buffer 越大，空槽也在算、也在传（Switch 图 3）。
2. 被丢的 token 在这一层没过专家，等于少训了一部分。Switch 报告用够大的 aux loss 系数时 drop 率通常 <1%（作者自测）。

**为什么现在主流不丢**。DeepSeek-V2 还在用设备级丢 token（每卡容量相当于 CF = 1.0，丢亲和度最低的 token，约 10% 的序列永远不丢，[arXiv 2405.04434](https://arxiv.org/abs/2405.04434) §2.2.4）。V3 报告原话是 V3 训练时不丢任何 token（does not drop any tokens），推理也不丢。前提是均衡做得足够好（下一节的 bias 调节加 node-limited routing），这样不丢 token 也不会出现某张卡长期过载。不丢的好处是每个 token 都完整训练，训练和推理的计算路径一致。

### 3. 均衡手段的演进（训练侧）

**3.1 aux loss 与系数取舍**。公式见 [MoE 与 EP](/parallel/moe-ep#第-4-步-负载不均与对策)：$\mathcal{L}_{\text{aux}} = \alpha E \sum_i f_i P_i$，$f_i$ 是实际分到专家 $i$ 的比例（不可导），$P_i$ 是路由概率均值（可导），均匀时等于 $\alpha$。难点在 $\alpha$：

1. Switch 扫了 $10^{-1}$ 到 $10^{-5}$，取 $\alpha = 10^{-2}$（作者自测）。
2. [Loss-Free Balancing](https://arxiv.org/abs/2408.15664)（DeepSeek）图 2：$\alpha$ 取 0 或 $10^{-4}$ 时负载严重不均；取 $10^{-2}$ 时均衡了，但困惑度明显变差（作者自测）。原因是 aux loss 的梯度和语言模型的梯度一起改 router，均衡和效果在抢同一组参数。

**3.2 device-limited / node-limited routing**。专家细了以后 $k$ 变大（V3 是 8），一个 token 的 8 个专家可能散在 8 个节点上，跨节点 IB 流量随之涨。做法：先选节点，再在选中的节点里选专家。

1. DeepSeek-V2：每层路由专家放在 8 张卡上，每 token 最多发往 $M = 3$ 张卡（§3.1.2）。
2. DeepSeek-V3：每 token 最多发往 $M = 4$ 个节点。每个节点的得分 = 该节点上专家的亲和度里最高的 $k/M = 2$ 个之和，取得分最高的 4 个节点，再在这 4 个节点的专家里选 top-8（§2.1.2）。
3. 效果：每个 token 跨 IB 最多发 4 份，到了对端节点再走 NVLink 分发给同节点的多个专家。

**3.3 aux-loss-free：用 bias 做外部控制回路**（V3 §2.1.2，最早见 [Loss-Free Balancing](https://arxiv.org/abs/2408.15664)）：

$$
g'_{i,t} =
\begin{cases}
s_{i,t}, & s_{i,t} + b_i \in \text{TopK}\left(\{s_{j,t} + b_j\},\ k\right) \\
0, & \text{否则}
\end{cases}
\qquad
b_i \leftarrow b_i + \gamma \cdot \text{sign}\left(\bar{c} - c_i\right)
$$

1. $b_i$ **只用来选专家**，门控权重仍用原始 $s_{i,t}$（再在选中的 $k$ 个里归一化）。所以 bias 不改变专家输出的加权方式，也没有梯度流过它。
2. 每步结束后数一遍负载：过载（$c_i > \bar{c}$）的专家 $b_i$ 减 $\gamma$，欠载的加 $\gamma$。$\bar{c}$ 是平均负载。
3. V3 取 $\gamma = 0.001$，前 14.3T token 用它，最后 500B token 设为 0。
4. Loss-Free Balancing 论文 1B 模型：aux loss 方法验证困惑度 9.56、$\text{MaxVio}_{\text{global}} = 0.72$；loss-free 9.50、0.04（作者自测）。$\text{MaxVio} = \max_i c_i / \bar{c} - 1$，就是上一节「最大 / 平均」减 1。

算一笔：4 个专家，这一步负载 $(700, 500, 450, 350)$，平均 500，$\gamma = 0.001$，bias 从 0 变成 $(-0.001, 0, +0.001, +0.001)$。负载正好等于平均时 sign 为 0，不动。

**3.4 序列级小 aux loss 兜底**。bias 只管整个 batch 的统计，单条序列内部仍可能极端集中。V3 另加一个序列级 balance loss，$T$ 换成一条序列的长度：

$$
\mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{E} f_i P_i,\qquad
f_i = \frac{E}{kT}\sum_{t=1}^{T} \mathbb{1}\left(i \in \text{TopK}_t\right),\qquad
P_i = \frac{1}{T}\sum_{t=1}^{T} \frac{s_{i,t}}{\sum_j s_{j,t}}
$$

$f_i$ 归一化成均匀时等于 1，所以均匀时 $\mathcal{L}_{\text{Bal}} = \alpha$。V3 取 $\alpha = 10^{-4}$，报告说只是为了避免单条序列里的极端不均。比 Switch 的 $10^{-2}$ 小两个数量级，对主任务梯度的干扰可以忽略。

### 4. 路由稳定性

**4.1 expert collapse**。路由是自我强化的：被选得多的专家训得多、变得更好，于是被选得更多；没被选的专家拿不到梯度，越来越没用。[DeepSeekMoE](https://arxiv.org/abs/2401.06066) 把它叫 routing collapse：模型总是只选少数几个专家，其他专家得不到充分训练。训练时要按层监控每个专家的负载比例 $c_i / (Tk)$ 和 MaxVio，出现长期接近 0 的专家（dead expert）就是信号。上一节的均衡手段首先就是防这个。

**4.2 router 用 fp32**。router 的输出要过 exp（softmax）或 sigmoid，再取 top-$k$ 这种离散决策，舍入误差会直接换掉选中的专家。bf16 尾数 7 位，在 $[128, 256)$ 区间相邻两数间隔是 $2^{7-7} = 1$：logit 128.5 会被舍入到 128 或 129，误差 0.5 对应概率差 $e^{0.5} \approx 1.65$ 倍（ST-MoE 用的就是这个例子）。所以：

1. Switch 的 selective precision：只把 router 的输入和计算 cast 到 fp32，算完再 cast 回 bf16，不额外传 fp32 张量（Switch 表 2）。
2. Megatron-Core 的 score function 也是先把 logits 转 fp32 再算 softmax / sigmoid。
3. V3 的 FP8 训练里 MoE gating 保留原精度，见 [FP8 训练](/train/fp8-training)。

**4.3 router z-loss**（[ST-MoE](https://arxiv.org/abs/2202.08906) 式 (5)(6)）。fp32 在最大规模上仍不够，ST-MoE 加了一项罚 router logits 的整体大小：

$$
\mathcal{L}_z = \frac{1}{B}\sum_{i=1}^{B}\left(\log\sum_{j=1}^{E} e^{x_j^{(i)}}\right)^2,\qquad
\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{CE}} + c_B \mathcal{L}_B + c_z \mathcal{L}_z
$$

1. $B$ 是 token 数，$x^{(i)} \in \mathbb{R}^E$ 是第 $i$ 个 token 进 router 的 logits，$\mathcal{L}_B$ 是 aux loss。
2. $c_z = 10^{-3}$，ST-MoE 按预训练后的模型质量扫出来的；$c_B = 10^{-2}$。
3. 作用：把 logits 拉回 0 附近，进 exp 之前的数小，舍入误差就小。ST-MoE 表 4：基线 6 次里 4 次稳定，加 z-loss 3 次里 3 次稳定，质量还略好（作者自测）。

算一笔：$E = 8$，一个 token 的 logits 全是 10，$\log\sum e^{x} = 10 + \ln 8 = 12.08$，$\mathcal{L}_z = 145.9$，乘 $c_z$ 是 0.146；logits 全是 0 时是 $(\ln 8)^2 \times 10^{-3} = 0.0043$。和输出层的 z-loss（PaLM，系数 $10^{-4}$）是同一个形式，只是用在 router 上，见 [训练稳定性](/train/stability)。

**4.4 sigmoid gating**。Switch、Mixtral 用 softmax；V3 用 sigmoid 算每个专家的亲和度 $s_{i,t} = \sigma(u_t^\top e_i)$，再在选中的 $k$ 个里归一化得到门控权重。sigmoid 下每个专家的分数各自独立，不会因为别的专家变大而被压低；归一化保证 $k$ 个门控权重加起来是 1，输出尺度和 $E$ 无关。bias 加在这种独立分数上，调一个专家不会牵动其他专家的分数。

### 5. 为什么 MoE 的 MFU 更低

MFU 的分子按 $6N_{\text{active}}$ 算（定义见 [训练算力账](/train/compute-mfu)），下面这些都只花时间、不进分子：

1. **all-to-all 通信**。V3 报告跨节点 EP 带来的计算 : 通信约 1 : 1，不藏的话一半时间在等网络。V3 用 [DualPipe](/train/pipeline-schedules) 把一个 forward chunk 和一个 backward chunk 配对，让一个的 all-to-all 和另一个的计算重叠；all-to-all kernel 占了 H800 132 个 SM 里的 20 个（[计算通信 Overlap](/parallel/comm-overlap)）。这 20 个 SM 不能做 GEMM，相当于峰值打了 85 折。
2. **负载不均**。第 2 节：最大 / 平均 = 1.25 时专家计算只有 80% 效率。
3. **专家 GEMM 变窄、变碎**。EP 度为 $\text{EP}$、均匀路由时，每卡有 $E/\text{EP}$ 个专家，每卡贡献 $T_{\text{gpu}}$ 个 token，每个专家收到

$$
m = \frac{T_{\text{gpu}} \cdot k \cdot \text{EP}}{E}
$$

   V3（$k = 8$，EP 64，$E = 256$）是 $m = 2T_{\text{gpu}}$。GEMM 形状 $[m, 7168] \times [7168, 2048]$：$N$ 维只有 2048，而 V3 dense 层的 FFN 中间维是 18432。micro-batch 越小 $m$ 越小，tile 越难填满 SM。

4. **grouped GEMM 的 padding**。每个专家的 token 段要对齐到 GEMM 在 $M$ 方向的 block 大小（[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) contiguous layout 的要求）。block 取 128 行时，每个专家平均多算约 64 行空行：$m = 1024$ 时浪费约 6%，$m = 8192$ 时不到 1%。

**用 V3 公开的成本反推等效 MFU**（本页推算）。报告表 1：预训练 2664K H800 GPU 小时，上下文扩展 119K，后训练 5K，共 2.788M。只看预训练：

1. 总 FLOPs（只算 $6N$）：$6 \times 3.7\times10^{10} \times 1.48\times10^{13} = 3.29\times10^{24}$。
2. GPU 秒数：$2.664\times10^{6} \times 3600 = 9.59\times10^{9}$。
3. 每卡吞吐：$3.29\times10^{24} / 9.59\times10^{9} \approx 343$ TFLOP/s。
4. H800 SXM 的计算单元和 H100 SXM 相同，按 bf16 dense 989、fp8 dense 1979 TFLOP/s 算：相对 bf16 峰值 35%，相对 fp8 峰值 17%。
5. 加上 attention（MLA 每头 QK 维 192、V 维 128，128 头，$s = 4096$，按 PaLM 的不带 causal mask 口径 $3 \times 2s \cdot 128 \cdot (192+128)$ 每层，61 层共 $6.1\times10^{10}$ / token，是 $6N$ 的 28%）：约 437 TFLOP/s，bf16 峰值的 44%，fp8 峰值的 22%。

没算 MTP 模块和 recompute（recompute 本来就不进 MFU）。V3 的主要 GEMM 用 fp8 算，所以更公平的分母是 fp8 峰值：17–22%。对照 dense 的 Llama 3 405B 在 H100 上 bf16 MFU 38–43%（作者自测，见 [训练算力账](/train/compute-mfu)）。MoE 换来的是同样 GPU 小时能训一个大得多的模型，而不是更高的 MFU。

### 6. 共享专家与细粒度专家对训练效率的影响

[DeepSeekMoE](https://arxiv.org/abs/2401.06066) 的两个改动，系统侧的后果：

1. **细粒度专家**：每个专家切成 $m$ 份更小的，激活数也乘 $m$，总参数和每 token 计算不变。组合数从 $\binom{16}{2} = 120$ 变成 $\binom{64}{8} = 4{,}426{,}165{,}368$（论文的例子），专家能分得更专。代价是 $k$ 变大：dispatch / combine 的通信量和 $k$ 成正比，所以 V3 的 $k = 8$ 才需要 node-limited routing；专家 GEMM 的 $N$ 维变窄（2048），单个 GEMM 效率更低。
2. **共享专家**：每个 token 都走，不参与路由，所以不需要 all-to-all，可以在本卡上和 dispatch 同时算，正好填进等通信的空档；它的负载天然均衡，不受第 2 节的影响。

## 面试追问

::: details Q：MoE 训练时 6ND 的 N 用总参数还是激活参数？显存呢？
算力用激活参数：没被选中的专家不参与这个 token 的矩阵乘。显存用总参数：优化器要给每个参数存主权重和 Adam 状态，不管这一步有没有用到。V3 是 $6 \times 37\text{B}$ FLOPs / token，但模型状态 $671\text{B} \times 16 = 10.7$ TB，至少 135 张 80 GB 卡才放得下。
:::

::: details Q：capacity factor 设大设小各有什么代价？
设小：热门专家溢出，丢 token，$\text{drop} = \sum_i \max(0, c_i - C) / (Tk)$，被丢的 token 这层没训。设大：每个专家按 $C = \text{CF}\cdot Tk/E$ 预留 buffer，空槽也占显存、也参与计算和通信（Switch 图 3）。ST-MoE 推荐训练 CF 1.25。DeepSeek-V3 干脆不设上限、不丢 token，靠 bias 调节把负载压平。
:::

::: details Q：aux-loss-free 的 bias 为什么只加在选专家上，不加在门控权重上？
加在门控上就改变了专家输出的加权方式，等于改了模型函数，bias 又不是学出来的，会干扰训练。只用于 top-$k$ 选择时，bias 只决定「去哪个专家」，选中后仍按原始分数 $s_{i,t}$ 加权，没有梯度经过 bias，也就没有和主任务抢梯度的问题。这是它相比 aux loss 的核心好处。
:::

::: details Q：已经有 bias 调节了，V3 为什么还要序列级 aux loss？
bias 按整个 batch 的负载调，只保证 batch 级均衡；一条序列内部的 token 可能高度集中到少数专家。V3 加一个 $\alpha = 10^{-4}$ 的序列级 balance loss，只为防这种极端情况，系数小到对主任务梯度的影响可以忽略。
:::

::: details Q：router z-loss 和输出层的 z-loss 有什么区别？
形式一样，都是 $(\log\sum e^{x})^2$，都把 logits 的整体大小拉回 0 附近。输出层 z-loss 作用在词表 logits 上，PaLM 系数 $10^{-4}$；router z-loss 作用在 $E$ 维的 router logits 上，ST-MoE 系数 $10^{-3}$。router 更需要它：exp 之后再 top-$k$ 是离散决策，舍入误差会直接换掉选中的专家。
:::

::: details Q：怎么发现 expert collapse？
按层画每个专家的负载比例 $c_i / (Tk)$ 和 MaxVio（$\max_i c_i / \bar{c} - 1$）。正常训练 MaxVio 应该稳定在小值；有专家负载长期接近 0、或者少数专家的比例持续上涨，就是 collapse 的前兆。aux-loss-free 下也可以直接看 bias：某个专家的 $b_i$ 一直单调上涨，说明它怎么调都没人选。
:::

::: details Q：MoE 训练的 MFU 为什么比 dense 低？给个量级。
四个来源：跨节点 all-to-all（V3 报告计算 : 通信约 1 : 1，靠 DualPipe 重叠，通信还占了 20 个 SM）；负载不均（最大 / 平均 1.25 就损失 20%）；细粒度专家的 GEMM 窄（$N$ = 2048）且每专家 token 数随 micro-batch 变小；grouped GEMM 的对齐 padding。量级：V3 预训练 2664K GPU 小时、14.8T token，只算 $6N$ 每卡约 343 TFLOP/s，约 fp8 峰值的 17%（本页推算）。
:::

::: details Q：训练侧的均衡手段和推理侧的 EPLB 是什么关系？
训练侧（aux loss、bias、node-limited routing）改的是**模型学出来的路由分布**；推理侧的 EPLB 改的是**专家到卡的映射**，复制热门专家来压低最大负载。训练均衡得再好，推理时具体某个 batch 仍可能有局部热点，所以两者都要。EPLB 见 [MoE 与 EP](/parallel/moe-ep#第-4-步-负载不均与对策)。
:::

## 手撕

top-$k$ sigmoid router + aux-loss-free bias 更新 + 容量 / drop 统计。router 权重固定不训练，token 带一个公共偏移，让某些专家天然更热门；只靠 bias 调节，看负载随步数变均衡：

```python
import numpy as np

rng = np.random.default_rng(0)
T, d, E, k = 4096, 64, 16, 2           # 每步 token 数、hidden、专家数、top-k
CF, gamma = 1.25, 1e-3                 # capacity factor、bias 更新步长
W_g = rng.normal(0, 1 / np.sqrt(d), (d, E))   # router 权重（这里固定不训练）
mu = rng.normal(0, 0.5, d)             # 所有 token 共享的偏移 -> 某些专家天然更热门

def route(x, bias):
    s = 1 / (1 + np.exp(-(x @ W_g)))                 # sigmoid 亲和度 [T, E]
    idx = np.argsort(-(s + bias), axis=1)[:, :k]     # bias 只参与选专家
    g = np.take_along_axis(s, idx, 1)
    g = g / g.sum(1, keepdims=True)                  # 门控权重用原始 s，在 top-k 内归一化
    return idx, g

def stats(idx):
    load = np.bincount(idx.ravel(), minlength=E)      # 每个专家拿到的 token 数 c_i
    C = int(np.ceil(CF * T * k / E))                  # 容量 = CF * T * k / E
    dropped = np.maximum(load - C, 0).sum()           # 超出容量的 (token, 专家) 对
    return load, load.max() / load.mean(), dropped / (T * k)

bias = np.zeros(E)
for step in range(301):
    x = rng.normal(0, 1, (T, d)) + mu
    idx, g = route(x, bias)
    load, imb, drop = stats(idx)
    if step % 50 == 0:
        print(f"step {step:3d}  max/mean={imb:.2f}  drop={drop:6.2%}  "
              f"min/max load={load.min()}/{load.max()}")
    bias += gamma * np.sign(load.mean() - load)       # 欠载 +gamma，过载 -gamma

# step   0  max/mean=3.06  drop=27.67%  min/max load=116/1566
# step  50  max/mean=2.23  drop=10.64%  min/max load=175/1143
# step 100  max/mean=1.55  drop= 1.86%  min/max load=366/792
# step 150  max/mean=1.10  drop= 0.00%  min/max load=464/561
# step 200  max/mean=1.08  drop= 0.00%  min/max load=481/553
# step 250  max/mean=1.08  drop= 0.00%  min/max load=474/553
# step 300  max/mean=1.07  drop= 0.00%  min/max load=481/548
```

读输出：一开始最热的专家是平均的 3 倍，按 CF = 1.25 会丢 28% 的 (token, 专家) 对；bias 每步只动 0.001，约 150 步后最大 / 平均降到 1.1，已经不需要丢 token。这就是 V3 敢不丢 token 的原因。$\gamma$ 太小前期迟迟均衡不了，太大则训练后期的均衡反而变差（Loss-Free Balancing 图 4，作者自测）。

常见变体：把 `stats` 换成按 EP rank 汇总（每卡 $E/\text{EP}$ 个专家的负载相加）算专家计算效率；加 Switch aux loss 或 router z-loss 的数值；把 top-$k$ 改成先选 $M$ 个节点再选专家。

## 参考

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：§2.1.2 aux-loss-free 均衡、序列级 balance loss、node-limited routing、不丢 token；§3.2 DualPipe、20 SM 通信；§4.2 超参（$\gamma$、$\alpha$、$M$）；表 1 训练成本
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)：§2.2 device-limited routing、三种 balance loss、设备级丢 token
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)：bias 更新算法、MaxVio
- [Switch Transformers](https://arxiv.org/abs/2101.03961)：expert capacity 式 (3)、aux loss、selective precision
- [ST-MoE](https://arxiv.org/abs/2202.08906)：router z-loss、capacity factor 的取舍
- [GShard](https://arxiv.org/abs/2006.16668)：top-2 路由、expert capacity
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)：细粒度专家、共享专家、routing collapse
- [Megatron-Core MoE](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py)：`get_capacity`、`z_loss_func`、fp32 score function
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)：contiguous layout 的对齐要求
