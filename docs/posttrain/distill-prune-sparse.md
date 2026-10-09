---
title: 蒸馏、剪枝、稀疏（含 KV Pruning）
status: draft
tags: [distillation, pruning, kv-pruning]
difficulty: 3
order: 6
related: [/posttrain/sft, /posttrain/rl-infra, /inference/quantization-gptq, /inference/speculative-decoding, /inference/prefill-decode-roofline, /inference/memory-accounting, /inference/kv-cache-paged-attention, /leetgpu/attention-with-sinks]
stack: [4]
---

# 蒸馏、剪枝、稀疏（含 KV Pruning）

> 每种方法省的是什么（字节、FLOPs、KV），以及砍掉之后精度掉在哪

## 一句话结论

三条路都是「用更少的计算或字节逼近原模型」，区别在省哪一项：

- **蒸馏**：训练一个更小的 student 去模仿 teacher。省的是**一切**（参数、FLOPs、KV 都按小模型算），代价是要重新训练。loss 有三种：对齐 teacher 的 logits（forward KL，[Hinton et al. 2015](https://arxiv.org/abs/1503.02531)）、直接在 teacher 生成的数据上做 SFT（sequence-level KD），以及让 student 自己生成、teacher 逐 token 打分的 **on-policy 蒸馏**（reverse KL，[MiniLLM](https://arxiv.org/abs/2306.08543)、[GKD](https://arxiv.org/abs/2306.13649)）。
- **剪枝**：直接删权重。**非结构化**（magnitude、[Wanda](https://arxiv.org/abs/2306.11695)、[SparseGPT](https://arxiv.org/abs/2301.00774)）在 LLM 上能一次性剪到 50% 而困惑度只小幅上升，但在 GPU 上**几乎不提速**；**结构化**（删 head、删 FFN 通道、删层）直接缩小矩阵，FLOPs 和字节同比例下降，但掉点多，需要继续训练（常配蒸馏）。
- **2:4 半结构化稀疏**：每 4 个权重恰好 2 个为零，Ampere 起的 Sparse Tensor Core 峰值算力翻倍（[Mishra et al. 2021](https://arxiv.org/abs/2104.08378)），权重字节降到 56%。实测线性层约 1.6 倍、端到端约 1.24 倍（Wanda 论文，A6000，作者自测），而一次性剪成 2:4 的 LLaMA-7B 困惑度从 5.68 涨到 11 左右。
- **KV pruning**：推理时丢掉不重要的 token 的 K/V（H2O、SnapKV、StreamingLLM），省 KV 显存和 decode 读 KV 的带宽，代价是长程检索类任务。

## 推导

### 0. 先算清楚「推理成本」由什么决定

记号：$\Psi$ = 参数量，$L$ = 层数，$h$ = hidden size，$b$ = batch，$s$ = 上下文长度。

1. **decode（memory-bound）**：每步要从 HBM 读一遍全部权重（bf16 下 $2\Psi$ 字节）和 batch 里所有序列的 KV。时间 ≈ 字节 / 带宽。
2. **prefill（compute-bound）**：每 token 约 $2\Psi$ FLOPs。时间 ≈ FLOPs / 峰值算力。
3. **KV 显存**：每 token $2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot 2$ 字节，决定能并发多少请求。Llama-2-7B（MHA，$n_{kv} d_{head} = 4096$）是 $2 \times 32 \times 4096 \times 2 = 512$ KB/token，16K 上下文一条序列 8.6 GB。

推导见 [Roofline](/inference/prefill-decode-roofline) 和 [显存账](/inference/memory-accounting)。下面每种方法都回到这三项看它动了哪个。

### 1. 蒸馏：logit KD 与温度

teacher 和 student 在同一个位置给出 logits $z^T, z^S \in \mathbb{R}^V$（$V$ = 词表大小）。温度 $\tau$ 下的分布：

$$
p_\tau = \text{softmax}(z^T / \tau), \qquad q_\tau = \text{softmax}(z^S / \tau)
$$

[Hinton et al. 2015](https://arxiv.org/abs/1503.02531) 的 loss 是：

$$
\mathcal{L} = \lambda \cdot \text{CE}(y, q_1) + (1 - \lambda) \cdot \tau^2 \cdot \text{KL}(p_\tau \,\|\, q_\tau)
$$

- 为什么要温度：$\tau > 1$ 把分布拉平，teacher 对「错误答案的相对排序」这类信息（暗知识）才能传过去；$\tau = 1$ 时这些概率太小，几乎没有梯度。
- 为什么乘 $\tau^2$：对 student logit 的梯度是 $(q_{\tau,i} - p_{\tau,i}) / \tau$，高温下 $q - p$ 本身又约按 $1/\tau$ 缩小，总共约 $1/\tau^2$；乘回 $\tau^2$ 让改温度时不用重调 $\lambda$。

LLM 上是**每个位置**算一次，再按 SFT 的 loss mask 求和（见 [SFT](/posttrain/sft)）。前提：teacher 和 student **词表相同**，否则 $p$ 和 $q$ 不在同一个空间上。

**成本要算一下。** 两种做法：
- 在线：训练时每个 batch 跑一次 teacher forward。teacher 70B、student 8B 时，每 token teacher forward $2 \times 70\text{B} = 140$ GFLOPs，student 训练 $6 \times 8\text{B} = 48$ GFLOPs，teacher 反而是大头，约 3 倍。
- 离线：先把 teacher 的 logits 存盘。全词表存不下：$V = 128{,}256$（Llama 3 词表）× 2 字节 = 256 KB/token，$10^9$ token 就是 256 TB。只存 top-64：64 × (2 字节值 + 4 字节 id) = 384 B/token，$10^9$ token 384 GB，可行，但 top-k 之外的概率质量被丢掉。

### 2. Forward KL vs Reverse KL

$$
\text{Forward: } \text{KL}(p \,\|\, q) = \sum_v p(v) \log \frac{p(v)}{q(v)}, \qquad
\text{Reverse: } \text{KL}(q \,\|\, p) = \sum_v q(v) \log \frac{q(v)}{p(v)}
$$

（$p$ = teacher，$q$ = student。）看哪里的惩罚会爆：

- **Forward KL**：若 teacher 认为 $v$ 可能（$p(v) > 0$）而 student 给 $q(v) \to 0$，$\log(p/q) \to \infty$。所以 student 必须**覆盖** teacher 的所有模式（mode-covering）。容量不够的小模型只好把概率摊开，结果在 teacher 认为不可能的地方也放了概率，生成时会采到这些「胡话」。
- **Reverse KL**：若 student 在 $p(v) \approx 0$ 的地方放了概率，$\log(q/p) \to \infty$；student 漏掉 teacher 的某个模式则不受罚。所以 student 会**集中**在 teacher 的主要模式上（mode-seeking），宁可少说，不说错。

MiniLLM 的论点就是生成任务应该用 reverse KL，避免 student 高估 teacher 的低概率区域（作者自测效果更好）。

还有一个更关键的区别：forward KL 的期望在 $p$ 下，可以在**固定数据**上算（teacher-forcing）；reverse KL 的期望在 $q$ 下，**必须用 student 自己生成的序列**，这就引出 on-policy 蒸馏。

### 3. On-policy 蒸馏

off-policy（logit KD、数据蒸馏）的问题叫 exposure bias：训练时 student 总是在「teacher 或数据给的前缀」上预测下一个 token，推理时却要在**自己生成的前缀**上继续，一旦早期走偏，后面就进入没见过的状态，错误越滚越大。

GKD（[Agarwal et al. 2023](https://arxiv.org/abs/2306.13649)）和 MiniLLM 的做法：让 student 采样，teacher 在 student 的轨迹上逐 token 给出分布，最小化两者的散度。

```python
for prompts in loader:
    with torch.no_grad():
        y = student.generate(prompts)                     # rollout：student 自己写
    logits_s = student(prompts, y)                          # [B, T, V]，有梯度
    with torch.no_grad():
        logits_t = teacher(prompts, y)                      # teacher 在 student 的前缀上打分
    logq = logits_s.log_softmax(-1)
    logp = logits_t.log_softmax(-1)
    # 每个位置的全词表 reverse KL：Σ_v q(v) (log q(v) − log p(v))
    kl = (logq.exp() * (logq - logp)).sum(-1)               # [B, T]
    loss = (kl * resp_mask).sum() / resp_mask.sum()         # 只在回答 token 上
    loss.backward(); opt.step()
```

全词表版本需要 teacher 和 student 同时有完整 logits。另一种是只用**采样到的那个 token** 估计：每个 token 的 reward 记为 $-(\log q(y_t) - \log p(y_t))$，再用策略梯度更新（[Thinking Machines: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) 用的就是这种 per-token reverse KL）。

和 RL 比：RL 每条回答只有末尾一个标量 reward；on-policy 蒸馏**每个 token 都有监督信号**，所以样本效率高得多。Qwen3 的小模型用 strong-to-weak 蒸馏，在 Qwen3-8B 上 on-policy 蒸馏比 RL 效果更好且只用约 1/10 的 GPU 小时（[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)，作者自测）。

系统上它就是一个 RL 流水线：rollout 引擎生成、teacher 作为打分服务、trainer 更新 student 并同步权重，见 [RL Infra](/posttrain/rl-infra)。

**数据蒸馏（sequence-level KD）**：[Kim & Rush 2016](https://arxiv.org/abs/1606.07947) 提出直接在 teacher 生成的序列上训练 student，等价于用 teacher 输出做 SFT。不需要词表一致、不需要 teacher logits，所以最常用：DeepSeek-R1 用 R1 生成的约 80 万条样本直接 SFT 了 Qwen 和 Llama 系列小模型（[DeepSeek-R1](https://arxiv.org/abs/2501.12948)，作者自测）。投机解码的 draft 模型也常这样对齐 target 模型（[DistillSpec](https://arxiv.org/abs/2310.08461)），见 [投机解码](/inference/speculative-decoding)。

### 4. 非结构化剪枝：magnitude、Wanda、SparseGPT

目标：给线性层 $W \in \mathbb{R}^{d_{out} \times d_{in}}$ 选一个 mask，把 50% 的元素置零，让输出 $WX$ 变化最小。$X \in \mathbb{R}^{d_{in} \times n}$ 是 $n$ 个校准 token 的输入（两篇论文都用 C4 的 128 条序列）。

**Magnitude**（[Han et al. 2015](https://arxiv.org/abs/1506.02626)）：删 $|W_{ij}|$ 最小的。问题：LLM 的激活里有少数特征维度的幅值特别大（outlier，见 [量化](/inference/quantization)），一个很小的权重如果连着 outlier 通道，对输出的影响反而很大。

**Wanda**：分数加上输入的幅值，并且**在每个输出行内部**比较：

$$
S_{ij} = |W_{ij}| \cdot \|X_{j,:}\|_2
$$

每一行删掉分数最低的 50%。不更新剩下的权重，几秒钟剪完一个 65B 模型（作者自测）。

**SparseGPT**：把每层当成重建问题 $\min_{\hat W} \|WX - \hat W X\|_F^2$，在置零一个权重 $w_q$ 后，用二阶信息（$H = XX^\top$）**调整同一行其余权重**来补偿误差：

$$
\delta_{\text{row}} = -\frac{w_q}{[H^{-1}]_{qq}} \, H^{-1}_{:,q}
$$

这就是 OBS（Optimal Brain Surgeon）更新，和 GPTQ 的量化补偿是同一套机器（GPTQ 把「置零」换成「舍入到格点」），见 [GPTQ](/inference/quantization-gptq)。OPT-175B 4.5 小时内剪完（作者自测）。

效果（Wanda 论文 Table 3，WikiText 困惑度，越低越好，作者自测）：

| LLaMA | dense | 50% magnitude | 50% SparseGPT | 50% Wanda | 2:4 SparseGPT | 2:4 Wanda |
|---|---|---|---|---|---|---|
| 7B | 5.68 | 17.29 | 7.22 | 7.26 | 11.00 | 11.53 |
| 65B | 3.56 | 5.90 | 4.57 | 4.57 | 6.28 | 6.25 |

读法：(1) 大模型对剪枝更鲁棒；(2) 同样 50%，2:4 比非结构化掉得多，因为它强制每 4 个里删 2 个，不能把预算挪到不重要的区域。

**为什么非结构化 50% 在 GPU 上不提速**：

1. **计算**：Tensor Core 按稠密 tile 做矩阵乘，tile 里散落的零照样参与乘加，见 [Tensor Core GEMM](/gpu/tensor-core-gemm)。
2. **存储**：要省字节就得换稀疏格式，但索引本身有开销。CSR 每个非零元存 2 字节值 + 2 字节列号（int16），50% 稀疏时平均每个原元素 $0.5 \times 4 = 2$ 字节，**和稠密 bf16 一样大**。bitmask 格式（每元素 1 bit + 非零值）是 $0.5 \times 16 + 1 = 9$ bit，56%，但解码不规则。
3. **kernel**：通用稀疏 GEMM 访存不规则、用不上 Tensor Core，稀疏度不够高时比稠密还慢。

所以非结构化稀疏在 GPU 上主要是「研究信号」（说明有多少冗余），落地要么转成 2:4，要么在 CPU / 专用硬件上用。

### 5. 2:4 稀疏：真实能快多少

**格式**：沿矩阵乘的归约维（$K$，即 $d_{in}$）每连续 4 个元素恰有 2 个是零。压缩存储只留 2 个非零值，外加每个值 2 bit 的位置索引（metadata）。

| | 每 4 个元素的位数 | 相对稠密 |
|---|---|---|
| bf16 稠密 | $4 \times 16 = 64$ | 100% |
| bf16 2:4 | $2 \times 16 + 2 \times 2 = 36$ | 56.25% |
| int8 2:4 | $2 \times 8 + 2 \times 2 = 20$ | 62.5%（相对 int8 稠密） |

**算力**：Sparse Tensor Core 用 metadata 只取出 $B$ 中需要的那一半元素和非零值相乘，峰值数学吞吐是稠密的 2 倍（[Mishra et al. 2021](https://arxiv.org/abs/2104.08378)）。H100 规格表上标 "with sparsity" 的数字就是这个 2 倍，平时比较要用 dense 值。

**用 roofline 估上限**，Llama-2-7B、bs = 1 decode、H100（3.35 TB/s）：

- 稠密：线性层权重 $6.48\text{B} \times 2 = 12.96$ GB，加 lm_head 0.26 GB，共 13.2 GB，$13.2 / 3.35 \approx 3.9$ ms/token。
- 2:4：线性层 $12.96 \times 0.5625 = 7.29$ GB，共 7.55 GB，约 2.25 ms/token，上限 1.75 倍（memory-bound，收益来自少读字节，而不是 2 倍算力）。
- prefill（compute-bound）：上限是 2 倍算力。

**实测远低于上限**：Wanda 论文在 A6000 上用 CUTLASS 测 LLaMA-65B 的线性层，2:4 比稠密快 1.54–1.63 倍；LLaMA-7B 端到端只快 1.24 倍（312 ms → 251 ms），因为 attention、norm、采样等不受益（作者自测）。

**精度怎么补回来**：NVIDIA 给的流程是稠密训练 → 按幅值剪成 2:4 → 用原来的训练计划再训一遍（Mishra et al.，在视觉和 NLP 模型上作者自测无损）。LLM 上重训太贵，常见做法是剪完后做 LoRA 微调或蒸馏：Wanda 论文里 2:4 的 LLaMA-7B 经 LoRA 微调后困惑度从 11.53 降到 8.24（作者自测），仍明显高于稠密的 5.68。

### 6. 结构化剪枝：真正缩小矩阵

删的单位是整行 / 整列 / 整块：attention head、FFN 中间通道、整层、甚至 hidden 维。剪完就是一个更小的**稠密**模型，不需要特殊 kernel。

每种删法动的成本项不同：

| 删什么 | 权重字节 / FLOPs | KV cache |
|---|---|---|
| FFN 中间通道（$i$ 变小） | 按比例降 | 不变 |
| attention head（$n_{kv}$ 变小） | 降 attention 部分 | 按比例降 |
| 整层（$L$ 变小） | 按比例降 | 按比例降，延迟也按层数降 |
| hidden 维（$h$ 变小） | 几乎所有矩阵都降 | 按比例降 |

代表工作（均为作者自测）：
- **ShortGPT**（[Men et al. 2024](https://arxiv.org/abs/2403.03853)）：用 Block Influence $\text{BI}_\ell = 1 - \mathbb{E}[\cos(x_\ell^{in}, x_\ell^{out})]$ 衡量每层改变了多少隐状态，直接删 BI 最小的层。说明很多中间层高度冗余。
- **Sheared LLaMA**（[Xia et al. 2023](https://arxiv.org/abs/2310.06694)）：把 LLaMA2-7B 定向剪到 1.3B / 2.7B 的目标形状，再继续预训练，计算量只有从头训的 3%。
- **Minitron**（[Muralidharan et al. 2024](https://arxiv.org/abs/2407.14679)）：从 Nemotron-4 15B 剪出 8B / 4B，用蒸馏做恢复训练，每个模型少用最多 40 倍训练 token。

结构化剪枝 + 蒸馏恢复，是目前「从大模型造小模型」的主流做法：剪枝提供好的初始化，蒸馏提供密集的监督信号。

### 7. KV pruning：推理时丢 token

和前面不同，这里不改模型权重，而是在推理时**只保留一部分 token 的 K/V**。省的是 KV 显存（能开更大 batch）和 decode 每步读 KV 的带宽。Llama-2-7B、16K 上下文一条序列 KV 8.6 GB，只留 20% 就是 1.7 GB。

- **StreamingLLM**（[Xiao et al. 2023](https://arxiv.org/abs/2309.17453)）：保留开头几个 attention sink token + 最近 $W$ 个 token。mask 写法和原因见 [Attention with Sinks](/leetgpu/attention-with-sinks)。
- **H2O**（[Zhang et al. 2023](https://arxiv.org/abs/2306.14048)）：观察到少数 token（heavy hitters）累计拿走了大部分 attention 权重。保留「最近窗口 + 累计 attention 分数最高的 token」，超预算时淘汰分数最低的。保留 20% heavy hitters 时，OPT-6.7B / 30B 上吞吐比 DeepSpeed Zero-Inference、HF Accelerate 高最多 29 倍，比 FlexGen 高最多 3 倍（作者自测）。
- **SnapKV**（[Li et al. 2024](https://arxiv.org/abs/2404.14469)）：prefill 结束后用 prompt 末尾一个观察窗口的 query 对前文的 attention，**每个 head 各自**挑出重要位置（做一次池化让选中的位置成簇），一次性压缩 prompt 的 KV。16K 输入下 decode 快 3.6 倍、显存效率 8.2 倍（作者自测）。

**落地时的系统问题**（这是分析，不是论文结论）：
1. H2O 需要每步的 attention 权重，而 FlashAttention / PagedAttention kernel 不把 softmax 概率写回显存，要改 kernel 额外输出列和。
2. 每个 head 留的 token 不同，和 [PagedAttention](/inference/kv-cache-paged-attention) 「一个序列一张 block table、所有 head 共用」的布局冲突；淘汰后 block 里留下空洞，要么容忍碎片，要么做压缩搬移。
3. 被丢掉的 token 永远找不回来，后续问题恰好问到它时就答错，所以「大海捞针」类长程检索最敏感。prefix caching 共享的 block 也不能被某个请求单独剪掉。

### 8. 怎么准备 motivation 和 ablation

面试讲自己做过的压缩工作，按这个顺序：

1. **motivation 落到成本项**：瓶颈是 decode 带宽、prefill 算力还是 KV 显存？用第 0 节的公式算出当前数字，说明方法动的正是那一项。
2. **每个改动单独开关**：比如「结构化剪枝」「蒸馏恢复」「2:4」分开测，再测组合。
3. **三条线一起报**：困惑度（灵敏但不代表任务）、下游任务（含长上下文检索和推理类，剪枝最容易在这里掉）、**实际延迟 / 吞吐**（同一引擎、同一 batch 和长度设置）。只报 FLOPs 或理论稀疏度不够，非结构化 50% 就是反例。

## 面试追问

::: details Q：非结构化剪枝到 50% 为什么在 GPU 上几乎不提速？
三个原因：Tensor Core 按稠密 tile 算，散落的零照样参与乘加；稀疏格式的索引有开销，CSR 在 50% 稀疏时（2 字节值 + 2 字节 int16 列号）和稠密 bf16 一样大；通用稀疏 kernel 访存不规则，稀疏度不够高时比稠密还慢。要加速得用硬件支持的 2:4 模式（每 4 个里 2 个零），峰值算力 2 倍、权重字节降到 56%，实测线性层约 1.6 倍（Wanda 论文，作者自测）。
:::

::: details Q：2:4 稀疏为什么 decode 时到不了 2 倍？
decode 是 memory-bound，时间由读权重的字节数决定，不由算力决定。2:4 的 bf16 权重连 metadata 是稠密的 56.25%，所以 decode 线性层的上限是 $1/0.5625 \approx 1.78$ 倍；attention、norm、采样不受益，端到端更低（Wanda 论文 LLaMA-7B 端到端 1.24 倍，作者自测）。2 倍算力只在 compute-bound 的 prefill 才有意义。
:::

::: details Q：forward KL 和 reverse KL 蒸馏有什么区别？为什么 LLM 蒸馏偏向 reverse KL？
forward KL $\text{KL}(p\|q)$ 在 teacher 有概率而 student 没有的地方惩罚爆炸，逼 student 覆盖所有模式，小模型只能把概率摊开，结果在 teacher 认为不可能的地方也放概率，生成时采出胡话。reverse KL $\text{KL}(q\|p)$ 惩罚的是 student 在 teacher 认为不可能的地方放概率，student 会集中在主要模式上。另外 reverse KL 的期望在 student 分布下，必须用 student 自己的生成序列，天然是 on-policy 的，顺带解决 exposure bias（MiniLLM、GKD）。
:::

::: details Q：on-policy 蒸馏和 RL 有什么关系？
系统形态几乎一样：student 采样、外部打分、更新 student。区别在信号：RL 每条回答只有一个末尾标量 reward；on-policy 蒸馏里 teacher 对**每个 token** 给出分布（或 log 概率），每个位置都有监督，样本效率高得多。Qwen3 报告 8B 上 on-policy 蒸馏比 RL 效果好且只用约 1/10 GPU 小时（作者自测）。前提是有一个更强的 teacher，且 logit 版本要求词表一致。
:::

::: details Q：teacher 和 student 词表不一样怎么蒸馏？
logit 级（forward / reverse KL）要求两个分布在同一个词表上，不一致时不能直接用。最常用的退路是数据蒸馏：teacher 生成回答，student 在上面做 SFT，只需要文本。DeepSeek-R1 蒸馏 Qwen / Llama 就是这样（作者自测 80 万条样本）。
:::

::: details Q：结构化剪枝里删层、删 head、删 FFN 通道，对推理成本的影响有什么不同？
删 FFN 通道只减权重和 FLOPs，KV cache 不变；删 KV head 减 attention 权重，同时 KV 按比例减；删层同时减权重、FLOPs、KV，而且 decode 延迟按层数线性降，因为层是串行的。所以 KV 显存是瓶颈（长上下文、大 batch）时，删层或删 KV head 比缩 FFN 更值。
:::

::: details Q：KV pruning 和 KV 量化怎么选？
KV 量化（FP8）对所有 token 一视同仁地省一半，精度损失均匀且小；KV pruning 能省更多（只留 20% 就是 5 倍），但被丢的 token 永远找不回，长程检索类任务掉得厉害，而且和 PagedAttention、prefix caching、FlashAttention kernel 的配合都要改。生产上一般先上 KV 量化，pruning 用在超长上下文且任务对远处细节不敏感的场景，两者可以叠加。
:::

## 手撕

**logit 蒸馏 loss（带温度）**：

```python
import torch.nn.functional as F

def kd_loss(logits_s, logits_t, labels, tau=2.0, lam=0.5):
    """logits: [B, T, V]；labels: [B, T]，非回答位置为 -100"""
    mask = labels != -100
    ce = F.cross_entropy(logits_s[mask], labels[mask])            # 硬标签
    logq = F.log_softmax(logits_s[mask] / tau, -1)
    logp = F.log_softmax(logits_t[mask] / tau, -1)
    kl = (logp.exp() * (logp - logq)).sum(-1).mean()               # forward KL(p || q)
    return lam * ce + (1 - lam) * tau ** 2 * kl
```

**Wanda 剪枝（50% 非结构化 / 2:4）**：

```python
@torch.no_grad()
def wanda_prune(W, X, mode="unstructured"):
    """W: [d_out, d_in]；X: [n_tokens, d_in] 校准输入"""
    S = W.abs() * X.norm(dim=0)[None, :]               # 每列乘该输入通道的 L2 范数
    if mode == "unstructured":
        k = W.shape[1] // 2
        idx = S.topk(k, dim=1, largest=False).indices  # 每个输出行内删最小的一半
        mask = torch.ones_like(W, dtype=torch.bool).scatter_(1, idx, False)
    else:                                              # 2:4：沿 d_in 每 4 个留 2 个
        g = S.view(W.shape[0], -1, 4)
        keep = g.topk(2, dim=-1).indices
        mask = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, keep, True).view_as(W)
    return W * mask
```

**H2O 式 KV 淘汰（decode 一步）**：

```python
def h2o_step(K, V, score, q, k_new, v_new, budget, recent):
    """K, V: [n, d] 当前保留的 KV；score: [n] 每个 token 的累计 attention"""
    K = torch.cat([K, k_new[None]]); V = torch.cat([V, v_new[None]])
    score = torch.cat([score, score.new_zeros(1)])
    p = (q @ K.T / K.shape[1] ** 0.5).softmax(-1)      # [n+1]，这一步的 attention
    out = p @ V
    score += p                                         # 累计每个 token 被看的总量
    if K.shape[0] > budget:
        n = K.shape[0]
        cand = score[: n - recent]                     # 最近 recent 个 token 不参与淘汰
        drop = cand.argmin()
        keep = torch.arange(n) != drop
        K, V, score = K[keep], V[keep], score[keep]
    return out, K, V, score
```

常见变体：把 KD loss 改成 reverse KL；2:4 的 mask 沿哪个维度（归约维 $d_{in}$）；H2O 加 attention sink（前几个 token 也不参与淘汰）。

## 参考

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)
- [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD)](https://arxiv.org/abs/2306.13649)
- [Thinking Machines: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)、[DeepSeek-R1](https://arxiv.org/abs/2501.12948)、[DistillSpec](https://arxiv.org/abs/2310.08461)
- [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)
- [SparseGPT](https://arxiv.org/abs/2301.00774)、[Wanda](https://arxiv.org/abs/2306.11695)
- [Accelerating Sparse Deep Neural Networks（2:4 Sparse Tensor Core）](https://arxiv.org/abs/2104.08378)
- [ShortGPT](https://arxiv.org/abs/2403.03853)、[Sheared LLaMA](https://arxiv.org/abs/2310.06694)、[Minitron](https://arxiv.org/abs/2407.14679)
- [StreamingLLM](https://arxiv.org/abs/2309.17453)、[H2O](https://arxiv.org/abs/2306.14048)、[SnapKV](https://arxiv.org/abs/2404.14469)
