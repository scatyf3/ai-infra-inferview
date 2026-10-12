---
title: RL 训练提速：一步的时间拆开看，每一段怎么压
status: draft
tags: [rl-infra, verl, throughput, dynamic-batch, sequence-balancing, colocate]
difficulty: 3
order: 11
related: [/posttrain/rl-infra, /posttrain/rl-async-rollout, /posttrain/rl-weight-sync, /posttrain/rl-train-infer-mismatch, /posttrain/rl-monitoring, /inference/prefill-decode-roofline, /inference/kv-cache-paged-attention, /posttrain/training-memory, /parallel/zero-fsdp]
stack: []
train: [tr-r-rlsys, tr-m-offload]
---

# RL 训练提速：一步的时间拆开看，每一段怎么压

> 同步 colocate 的 GRPO 一步分成 generate、log-prob、ref、update、权重同步几段。每段用一个公式估时间，再看哪个旋钮能把它压下去

## 一句话结论

一个 RL step 的时间是各段之和。generate 是 decode，受**显存带宽**限制，每个 decode 步都要把权重读一遍，步数由**最长**那条回答决定；其余各段是训练引擎上的 forward / backward，受**算力**限制，时间和总 token 数 $T$ 成正比：old log-prob 和 ref 各 $2NT$ FLOPs，actor update $6NT$。所以提速的顺序是：

1. 先看 verl 的 `timing_s/*` 哪段最大。
2. generate 大：让每张卡同时 decode 的序列够多（KV 显存给够），同一 prompt 的 $n$ 条共享前缀，处理长尾。
3. log-prob / ref 大：能省的 forward 就省（LoRA 的 ref 复用 actor、不用 KL 就不要 ref），省不掉就按 token 数组 batch。
4. update 大：去掉 padding、按 token 数而不是样本数组 micro-batch、让各 DP rank 的 token 数一样多。
5. colocate 下训练状态和推理引擎抢同一块显存，用 sleep / offload 腾挪，切换本身只要秒级。

架构、异步和权重同步各有专页（[RL Infra](/posttrain/rl-infra)、[异步 Rollout](/posttrain/rl-async-rollout)、[权重同步](/posttrain/rl-weight-sync)），本页只讲同步 colocate 下一步之内怎么提速。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $B$ | 一步的 prompt 数（verl `data.train_batch_size`） | 128 |
| $n$ | 每个 prompt 采样的回答数（`rollout.n`） | 8 |
| $\ell_p$ | prompt 长度（token） | 512 |
| $\bar\ell$ | 回答的平均长度 | 4000 |
| $\ell_{\max}$ | 回答的最大长度（`data.max_response_length`） | 16000 |
| $N$ | 模型参数量 | $7.6\times10^9$（Qwen2.5-7B） |
| $k_v$ | 每 token 的 KV cache 字节数：$2 \cdot L \cdot H_{kv} \cdot d_h \cdot 2$ | $2\times28\times4\times128\times2 = 57344$ B |
| $G$ | GPU 数，训练和推理共用（colocate） | 8 张 H100 80 GB |
| $b$ | 每张卡同时 decode 的序列数，$b = Bn/G$（推理每卡一个 TP=1 实例） | 128 |
| $T$ | 训练侧要过的总 token 数，$T = Bn(\ell_p + \bar\ell)$ | $4.62\times10^6$ |
| BW | HBM 带宽 | 3.35 TB/s（H100 SXM） |
| peak | bf16 dense 峰值算力 | 989 TFLOP/s（H100 SXM） |
| MFU | 训练侧实际算力 / peak | 假设 40% |
| $M$ | 单卡显存 | 80 GB |

单位：GB = $10^9$ 字节；$k_v$ 里 $L$ 是层数，$H_{kv}$ 是 KV head 数，$d_h$ 是 head 维度，开头的 2 是 K、V 两份，结尾的 2 是 bf16 字节数。

### 1. 一步拆成几段

verl 的同步训练循环（[ray_trainer.py](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py)）按顺序给每段计时，面板上的名字是 `timing_s/<段名>`：

| 段（`timing_s/…`） | 做什么 | 时间估算 | 例子（秒） |
|---|---|---|---|
| `gen` | 推理引擎采 $Bn$ 条回答 | 见下式 | 94.6 |
| `reward` | 规则打分 / reward model | 规则打分可忽略 | ≈ 0 |
| `old_log_prob` | actor 重算 $\pi_{\text{old}}$ 的 logprob | $2NT / (G \cdot \text{peak} \cdot \text{MFU})$ | 22.2 |
| `ref` | reference 模型的 logprob | 同上 | 22.2 |
| `values` | critic forward（只有 PPO） | 同上 | GRPO 无 |
| `adv` | 算 advantage（GRPO 组内归一化） | CPU 上几乎免费 | ≈ 0 |
| `update_critic` | critic 训练（只有 PPO） | $6N_cT/(\dots)$ | GRPO 无 |
| `update_actor` | actor 训练 | $6NT/(G \cdot \text{peak} \cdot \text{MFU})$ | 66.6 |
| `update_weights` | 新权重推给推理引擎 | 见 [权重同步](/posttrain/rl-weight-sync) | 假设 5 |
| `step` | 整步 | 以上之和 | 210.6 |

另有 `timing_per_token_ms/<段名>`：`gen` 除以回答 token 数，其余除以 prompt + 回答的总 token 数（[metric_utils.py](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/metric_utils.py)），换 batch 大小后也能直接比。

**训练侧各段**：一次 forward 每 token 每参数做一次乘加，即 $2N$ FLOPs；backward 要对输入和对权重各求一次梯度，各是一次同样大小的矩阵乘，$4N$；合起来 update 是 $6N$（推导见 [训练显存账](/posttrain/training-memory)）。例子：$2NT = 2 \times 7.6\times10^9 \times 4.62\times10^6 = 7.0\times10^{16}$ FLOPs，除以 $8 \times 989\times10^{12} \times 0.4$，得 22.2 s；update 是它的 3 倍，66.6 s。verl 默认开 gradient checkpointing，backward 前多一次 forward，实际是 $8N$，这里按 $6N$ 算下界。

**generate**：decode 是 memory-bound（[Roofline](/inference/prefill-decode-roofline)），每个 decode 步的耗时 ≈ 这一步读的字节数 ÷ BW。读的是两样：

1. 全部权重 $2N$ 字节，每步一次，和 batch 里有几条无关。同步 RL 要等最长的那条写完，步数至少 $\ell_{\max}$。
2. 每条序列自己的 KV。一条回答写到第 $t$ 个 token 时要读 $(\ell_p + t)k_v$ 字节，写完 $\ell$ 个 token 共读 $\sum_{t=1}^{\ell}(\ell_p+t)k_v \approx \ell(\ell_p + \ell/2)k_v$。

加起来：

$$
t_{\text{gen}} \gtrsim \frac{\ell_{\max} \cdot 2N + b\,\bar\ell\,(\ell_p + \bar\ell/2)\,k_v}{\text{BW}}
$$

代入：权重项 $16000 \times 15.2\ \text{GB} / 3.35\ \text{TB/s} = 72.6$ s，KV 项 $128 \times 4000 \times (512 + 2000) \times 57344 / 3.35\times10^{12} = 22.0$ s，合计 94.6 s。检查 memory-bound 的前提：$b = 128$ 时每步算力时间 $2N b/\text{peak} = 2.0$ ms，读权重 $15.2\ \text{GB}/\text{BW} = 4.5$ ms，确实是读更慢。

**为什么 generate 往往占大头**。这个例子里它占 45%，但下界里藏着三件事：

1. 权重项由 $\ell_{\max}$ 决定。如果所有回答都是 4000 token，权重项只有 18.1 s；多出来的 54 s 都在等最长的几条。$\ell_{\max}$ 放到 32k（平均不变），generate 涨到 167 s，占 59%。训练侧各段只看总 token 数 $T$，不受长尾影响。
2. KV 项和 $\bar\ell^2$ 成正比，训练侧只和 $\bar\ell$ 成正比。回答越长，generate 涨得越快。
3. 下界假设带宽跑满、$b$ 不受 KV 显存限制（第 2 节会看到 $b$ 常常被卡住）。实测里 rollout 占 63%–87%（Seer 的分解，作者自测，见 [训练监控](/posttrain/rl-monitoring)）。

### 2. 压 generate

**(a) 每张卡同时 decode 的序列要够多**。一步的耗时是 $(2N + b \cdot \text{ctx} \cdot k_v)/\text{BW}$，ctx 是序列当前的上下文长度。权重那 $2N$ 字节被 $b$ 条序列分摊，$b$ 越大，每个 token 分到的读权重开销越小；$b$ 大到 KV 字节超过权重以后，再加 $b$ 就不再摊薄（见 [Roofline](/inference/prefill-decode-roofline) 的「batch 再大，AI 也有上限」）。$b$ 受三样东西限制：每卡分到的请求数 $Bn/G$、`rollout.max_num_seqs`（verl 默认 256）、KV 显存。

**(b) KV 显存：`gpu_memory_utilization`**。vLLM 0.7 起，这个值是整个 vLLM 实例占**总**显存的比例（[verl 性能调优文档](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)），权重和激活之外剩下的给 KV：

$$
\text{KV 容量（token）} \approx \frac{u \cdot M - 2N - \text{激活与 CUDA graph}}{k_v}
$$

$u$ 是 `gpu_memory_utilization`。忽略最后一项算上界：

1. verl 默认 $u = 0.5$：$(40 - 15.2)\ \text{GB} / 57344\ \text{B} = 4.3\times10^5$ token。每条写到平均长度时上下文 $\ell_p + \bar\ell = 4512$，只装得下 **95** 条，不到需要的 128 条。装不下的请求要排队，或者被抢占（preempt）后重算（见 [KV Cache 与 PagedAttention](/inference/kv-cache-paged-attention)），$b$ 实际变小、步数变多。
2. $u = 0.7$：$(56 - 15.2)/57344 = 7.1\times10^5$ token，装得下 157 条，够了。

$u$ 能调多高受 colocate 的显存账限制，见第 5 节。verl 文档建议 0.5–0.7。推理 TP 也影响 KV 容量：TP 越小，每卡要放的权重越多（TP=1 时一整份），留给 KV 的越少，换来的是没有 TP 通信、实例更多。

**(c) 同一 prompt 的 $n$ 条共享前缀**。$n$ 条回答的 prompt 完全相同。开 prefix caching（verl `rollout.enable_prefix_caching` 默认 True），第一条算完 prompt 的 KV，其余 $n-1$ 条直接命中，KV block 按引用计数共用一份。

例子：prefill token 数从 $Bn\ell_p = 524{,}288$ 降到 $B\ell_p = 65{,}536$，按 $2N$ / token、8 卡、MFU 40% 算，从 2.5 s 降到 0.3 s。$\ell_p$ 只有 512 时这点时间不起眼；prompt 是长文档、或者是多轮 agent（每轮的输入是之前所有轮次，见 [Agentic RL](/posttrain/rl-agentic)）时才是大头。

**(d) CUDA graph**。$b$ 很小的时候（长尾阶段只剩几条），每步的 GPU 时间只有几毫秒，CPU 逐个发 kernel 的开销就露出来了。CUDA graph 把一步的所有 kernel 录成一张图一次发出（[CUDA Graph](/framework/cuda-graph)）。verl 默认 `rollout.enforce_eager=False`，即开着。代价是 graph 占的显存在训练时不释放（vLLM 默认 sleep 时 graph 显存留在卡上），显存紧时用 `cudagraph_capture_sizes` 只录小 batch。

**(e) 长尾**。上面的权重项 $\ell_{\max} \cdot 2N/\text{BW}$ 是 generate 里最大的一块，它只认最长的那条。直接的旋钮是 `max_response_length`；不砍长度的办法（partial rollout、异步、过采样）见 [异步 Rollout](/posttrain/rl-async-rollout)。

### 3. 压 log-prob 与 ref

**(a) old log-prob：一次 forward，$2NT$**。能不能直接用推理引擎返回的 logprob（verl `rollout.calculate_log_probs` 默认 True，本来就会返回）？verl 的 `algorithm.rollout_correction.bypass_mode` 就是这么做的：`old_log_probs` 直接取 rollout 的 logprob，整段 `old_log_prob` 跳过。例子里省 22.2 s，占一步的 11%。

代价：推理引擎和训练引擎对同一 token 给的概率不一样，ratio 的分母换成推理侧的值，第一步更新前 ratio 就不等于 1，PPO clip 会把这些差当成「更新过头」截掉。默认做法（decoupled）是重算一遍，训推差距用 TIS 等单独修，见 [训推不一致](/posttrain/rl-train-infer-mismatch)。

**(b) ref：又一次 forward，$2NT$**。三种情况：

1. **不加 KL 就不要 ref**。verl 只在 `use_kl_in_reward` 或 `actor.use_kl_loss` 为真时才建 ref 模型（`need_reference_policy`）；DAPO 这类配方去掉了 KL 项，整段 `ref` 不存在。
2. **LoRA 训练：ref = 关掉 adapter 的 actor**。LoRA 冻结底座，底座本身就是 SFT 模型，关掉 adapter 算出来的就是 $\pi_{\text{ref}}$。verl 在 `model.lora_rank > 0` 时设 `ref_in_actor`，不单独建 ref worker，省一份权重和一组进程；forward 照样要算，时间不省。
3. **全参训练：ref 参数 offload**。ref 只做 forward，没有梯度和优化器状态，只存参数。按 bf16 算，FSDP 切 8 份后每卡 $2N/G = 1.9$ GB（verl 当前生成的配置里 ref 的 `model_dtype` 写的是 fp32，那就是 3.8 GB）。`ref.fsdp_config.param_offload` 打开后它平时待在 CPU，用时搬回来：按 PCIe Gen5 x16 单方向 64 GB/s 算 0.03–0.06 s，相对 22 s 的 forward 可以忽略。

**(c) GRPO 去掉 critic 省了什么**。PPO 的 critic 设成和 actor 一样大（$N_c = N$），训练侧多两段：`values` 一次 forward $2NT$，`update_critic` $6NT$。

| | old log-prob | ref | values | update_critic | update_actor | 合计 |
|---|---|---|---|---|---|---|
| PPO | $2NT$ | $2NT$ | $2NT$ | $6NT$ | $6NT$ | $18NT$ |
| GRPO | $2NT$ | $2NT$ | — | — | $6NT$ | $10NT$ |

训练侧计算少 $8/18 = 44\%$；例子里一步从 299 s 降到 211 s。显存上还省掉 critic 的整套训练状态（每参数 16 字节）。前提是两者的回答总数 $Bn$ 相同；GRPO 靠增大 $n$ 换来组内基线，generate 那段并不省。

### 4. 压 update

**(a) 去掉 padding（remove padding）**。一个 batch 里的序列长短不一，按最长的补齐后，补的位置也要做矩阵乘。去掉 padding：把所有序列首尾拼成一条一维长序列，用 `cu_seqlens`（每条序列的起止位置）告诉 varlen attention kernel 在哪里切开，计算量只和真实 token 数有关。verl 的开关是 `model.use_remove_padding`（当前默认 True）。

例子：一张卡上 8 条序列，prompt 都是 0.5k，回答 $[1, 1, 2, 2, 3, 4, 8, 16]$k，真实 token 共 41k。全部补到 16.5k 是 132k，**69% 的计算花在 padding 上**。

**(b) 按 token 数组 micro-batch（dynamic batch size）**。固定样本数的 micro-batch，大小只能按最坏情况定：每条都可能是 16.5k，显存只允许一次过 33k token 时，micro-batch 只能取 2 条，8 条要分 4 次。按 token 数组：verl 开 `actor.use_dynamic_bsz=True`，每个 micro-batch 的 token 数不超过 `actor.ppo_max_token_len_per_gpu`，micro-batch 数是

$$
\left\lceil \frac{\text{本卡总 token 数}}{\texttt{ppo\_max\_token\_len\_per\_gpu}} \right\rceil
$$

例子：上限取 verl 文档建议的下限 $2 \times (512 + 16000) = 33024$，$\lceil 41000/33024 \rceil = 2$ 次，比 4 次少一半。每个 micro-batch 是一次 forward + backward：次数少，矩阵更大、kernel 更满；FSDP 每次 forward / backward 都要 all-gather 一遍参数，次数少通信也少。forward-only 的段（log-prob、ref）用 `log_prob_use_dynamic_bsz` 和 `log_prob_max_token_len_per_gpu`，默认继承 actor 的设置；不存激活，上限可以设得比训练更大。

verl 切 micro-batch 时也做均衡（`rearrange_micro_batches`）：先按上式定个数，再把序列分进各 micro-batch 使工作量接近，各 DP rank 的 micro-batch 个数取最大值对齐。

**(c) DP rank 之间的序列均衡**。数据并行下，每个 rank 处理自己那份数据，然后在梯度同步处互相等待。所以 update 的时间由**总工作量最大**的 rank 决定：

$$
t_{\text{update}} \propto \max_r W_r, \qquad \text{效率} = \frac{\text{mean}_r W_r}{\max_r W_r}
$$

$W_r$ 是第 $r$ 个 rank 的工作量。GRPO 下不均衡特别严重：同一 prompt 的 $n$ 条回答在 batch 里挨着，难题的 $n$ 条都长、简单题的都短，按顺序切给各 rank，长的会扎堆。

例子：4 个 rank，4 道题 × $n = 4$，回答长度（千 token）依次是 $[1,1,2,1]$、$[2,3,2,3]$、$[4,6,5,5]$、$[12,16,9,14]$。按顺序切，四个 rank 的和是 $[5, 10, 20, 51]$，均值 21.5，最慢的是均值的 **2.37 倍**，效率 42%。均衡之后是 $[22, 22, 21, 21]$，最慢的是均值的 1.02 倍。update 段时间几乎减半。

verl 的做法（[seqlen_balancing.py](https://github.com/volcengine/verl/blob/main/verl/utils/seqlen_balancing.py)）：`trainer.balance_batch=True`（默认）时，在 generate 之后把整个 batch 重排，用 Karmarkar–Karp 差分法（largest differencing method）分成 DP 份、每份条数相同。思路是每次取出「最大和最小之差」最大的两个部分划分，大的配小的合并，直到只剩一个。权重不是纯 token 数，而是 `calculate_workload` 给的 $24576\,s + s^2$（$s$ 是序列长度，注释说常数按 hidden 4096 的 7B 标定，$s^2$ 项代表 attention）。面板上 `global_seqlen/max`、`global_seqlen/balanced_max`、`global_seqlen/mean` 分别是均衡前后最大 rank 的 token 数和平均值。

### 5. colocate 下的显存切换

训练和推理在同一组卡上轮流跑，一张卡的显存要同时满足两边：

1. **训练状态**：FSDP 切 $G$ 份后每卡约 $16N/G$ 字节（参数、梯度、Adam 状态，见 [训练显存账](/posttrain/training-memory) 和 [ZeRO / FSDP](/parallel/zero-fsdp)）。verl 默认 `fsdp_config.param_offload=False`、`optimizer_offload=False`，这部分在 generate 期间**留在卡上**。
2. **推理引擎**：$u \cdot M$，含一整份 bf16 权重、KV、CUDA graph。

不 offload 时要满足 $u \cdot M + 16N/G \le M$。例子：$16 \times 7.6\times10^9 / 8 = 15.2$ GB，$u \le (80 - 15.2)/80 = 0.81$；第 2 节想要的 $u = 0.7$ 放得下（还要给 ref 分片、通信 buffer 留余量）。更大的模型放不下时，打开 offload：`optimizer_offload` 把 Adam 的 $m$、$v$（每参数 8 字节）挪到 CPU，`param_offload` 把参数也挪走，腾出来的显存给 KV。

**推理侧怎么让出显存**：vLLM 的 sleep mode（[文档](https://docs.vllm.ai/en/latest/features/sleep_mode/)）

| 级别 | 权重 | KV cache | 醒来要做什么 |
|---|---|---|---|
| level 1 | 卸到 CPU 内存 | 丢掉 | 从 CPU 拷回权重 |
| level 2 | 丢掉 | 丢掉 | 重新灌权重 |

全参 RL 用 level 2：醒来本来就要换新权重，没必要备份旧的。LoRA 以 adapter 形式加载进 vLLM 时，只同步 adapter、底座不变，要用 level 1 把底座备份在 CPU；verl 的 `_resolve_sleep_level` 在这种情况（以及开了 MTP 投机解码时）选 level 1，其余选 level 2。醒来的顺序是先 `wake_up(tags=["weights"])`、灌完新权重再 `wake_up(tags=["kv_cache"])`，避免两者同时占满显存（细节见 [权重同步](/posttrain/rl-weight-sync)）。

**切换开销估算**：搬运量 ÷ PCIe 带宽（单方向 64 GB/s，要用 pinned memory 才跑得到）。

1. level 1 的权重 $2N = 15.2$ GB：卸下、装回各 0.24 s。level 2 不搬。
2. 训练状态全部 offload：每卡 15.2 GB，每个方向 0.24 s；只 offload 优化器状态是一半。

合计不到 1 s，相对 210 s 的一步可以忽略。只有模型小、回答短、一步只有十几秒时，切换才值得单独优化。

### 6. 现象 → 先动哪个旋钮

| 现象 | 先看 | 先动的旋钮 |
|---|---|---|
| `timing_s/gen` 占大头，vLLM 日志里 KV cache 用满、有抢占 | KV 用量、运行中的请求数 | 调大 `gpu_memory_utilization`；不够就打开训练侧 offload 腾显存 |
| gen 慢，但 KV 没满、同时运行的请求少 | 每卡请求数 $Bn/G$ | 调大 `max_num_seqs` 或 `train_batch_size`；确认 `enforce_eager=False` |
| gen 慢，`response_length/max` 远大于 mean | 长度分布 | `max_response_length`；partial rollout / 异步（[异步 Rollout](/posttrain/rl-async-rollout)） |
| `old_log_prob`、`ref` 合起来接近 `update_actor` | 每段 `timing_per_token_ms` | 开 `log_prob_use_dynamic_bsz` 并调大 `log_prob_max_token_len_per_gpu`；不用 KL 就去掉 ref |
| `update_actor` OOM | 报错时的 micro-batch token 数 | 调小 `ppo_max_token_len_per_gpu`；确认 gradient checkpointing 开着 |
| `update_actor` 慢、`perf/mfu/actor` 低 | padding 比例、micro-batch 个数 | `use_remove_padding=True`，`use_dynamic_bsz=True`，调大 `ppo_max_token_len_per_gpu` |
| 各 rank 互相等，`global_seqlen/max` 远大于 mean | `global_seqlen/balanced_max` | `balance_batch=True` |
| 训练和推理切换时 OOM | 醒来时的显存峰值 | 先醒权重后醒 KV；`cudagraph_capture_sizes` 只录小 batch；调小 `gpu_memory_utilization` |

## 面试追问

::: details Q：RL 训练很慢，怎么定位慢在哪？
看 `timing_s/*` 各段的占比，再看 `timing_per_token_ms/*` 和自己的估算比：generate 和 $\ell_{\max} \cdot 2N/\text{BW}$ 加 KV 项比，训练侧和 $2NT$、$6NT$ 除以 $G \cdot \text{peak} \cdot \text{MFU}$ 比。哪段离估算最远，问题就在哪段。generate 远超估算，查 KV 是否装得下、是否在抢占；训练侧远超估算，查 padding、micro-batch 个数和 rank 间的不均衡（`global_seqlen/*`）。
:::

::: details Q：GRPO 去掉 critic，训练侧省了多少计算？
设 critic 和 actor 一样大。PPO 训练侧是 old log-prob $2NT$ + ref $2NT$ + values $2NT$ + critic update $6NT$ + actor update $6NT$ = $18NT$；GRPO 去掉中间两项剩 $10NT$，少 44%。显存上还少一整套 critic 训练状态（每参数 16 字节）。generate 不省，GRPO 每题要采 $n$ 条。
:::

::: details Q：一批数据只更新一次（ppo_epochs = 1，只有一个 mini-batch），old log-prob 还要单独算吗？
数学上不用：更新前 $\pi_\theta = \pi_{\text{old}}$，update 第一次 forward 算出的 logprob 取 detach 就是 old log-prob，ratio 恒为 1，clip 不起作用。有多个 mini-batch 或多个 epoch 时，第二个 mini-batch 开始参数已经变了，必须在更新前单独算一遍。至于用推理引擎的 logprob 代替，是另一个问题，代价是训推不一致。
:::

::: details Q：`use_dynamic_bsz` 和 `use_remove_padding` 有什么区别？
remove padding 改的是**一个 micro-batch 内部**：序列拼成一条、用 `cu_seqlens` 切开，计算量只算真实 token。dynamic batch size 改的是**怎么分 micro-batch**：按 token 上限装，而不是按固定条数。前者去掉无效计算，后者让每个 micro-batch 都接近显存上限、减少 forward / backward 次数。两者一起开。
:::

::: details Q：为什么 GRPO 下 DP 不均衡比 SFT 严重？
GRPO 同一题的 $n$ 条回答在 batch 里挨着，长度高度相关：难题的 $n$ 条都长，简单题的都短。按顺序切给各 rank，长的扎堆到一两个 rank 上。update 时间由最慢的 rank 决定，效率是 mean / max。verl 的 `balance_batch` 在 generate 后用 Karmarkar–Karp 重排，让各 rank 的工作量接近。重排不影响 advantage（按 uid 分组算的），只改变 mini-batch 的组成。
:::

::: details Q：`gpu_memory_utilization` 为什么不直接调到 0.9？
colocate 时训练状态默认留在卡上，每卡 $16N/G$。7.6B、8 卡是 15.2 GB，$u$ 最多约 0.81，还要给 ref 分片和通信 buffer 留余量。想再大就打开 optimizer / param offload，代价是每步多搬几个 GB 过 PCIe（秒级以内）。另外 CUDA graph 的显存在训练时默认不释放，也要算进去。
:::

## 手撕

**(1) step time 估算**：照第 1 节的式子，同步、colocate、推理每卡一个 TP=1 实例。

```python
def step_time(N, B, n, lp, lbar, lmax, kv, G, bw=3.35e12, peak=989e12, mfu=0.4,
              critic=False, ppo_epochs=1, t_sync=5.0):
    b = B * n / G                                   # 每卡同时 decode 的序列数
    T = B * n * (lp + lbar)                         # 训练侧要过的总 token 数
    fwd = 2 * N * T / (G * peak * mfu)              # 一次 forward：每 token 2N FLOPs
    t = {
        "gen": (lmax * 2 * N + b * lbar * (lp + lbar / 2) * kv) / bw,
        "old_log_prob": fwd,
        "ref": fwd,
        "values": fwd if critic else 0.0,
        "update_critic": 3 * fwd * ppo_epochs if critic else 0.0,
        "update_actor": 3 * fwd * ppo_epochs,       # 6N = 3 次 forward 的量
        "update_weights": t_sync,
    }
    t["step"] = sum(t.values())
    return t

t = step_time(N=7.6e9, B=128, n=8, lp=512, lbar=4000, lmax=16000, kv=57344, G=8)
print({k: round(v, 1) for k, v in t.items()})
# {'gen': 94.6, 'old_log_prob': 22.2, 'ref': 22.2, 'values': 0.0, 'update_critic': 0.0,
#  'update_actor': 66.6, 'update_weights': 5.0, 'step': 210.6}
print(round(t["gen"] / t["step"], 2))              # 0.45
```

**(2) 把序列分给 $k$ 个 DP rank，使最大总长最小**，每个 rank 条数相同。贪心：从长到短，每条放进当前总长最小、条数未满的 rank。

```python
def sequential_split(lens, k):
    m = len(lens) // k
    return [list(range(r * m, (r + 1) * m)) for r in range(k)]

def greedy_balance(lens, k):
    m = len(lens) // k
    parts, sums = [[] for _ in range(k)], [0] * k
    for i in sorted(range(len(lens)), key=lambda i: -lens[i]):
        r = min((r for r in range(k) if len(parts[r]) < m), key=lambda r: sums[r])
        parts[r].append(i)
        sums[r] += lens[i]
    return parts

def report(name, lens, parts):
    sums = [sum(lens[i] for i in p) for p in parts]
    mean = sum(sums) / len(sums)
    print(f"{name}: {sums}, max/mean = {max(sums) / mean:.2f}")

lens = [1, 1, 2, 1, 2, 3, 2, 3, 4, 6, 5, 5, 12, 16, 9, 14]   # 4 题 × n=4，千 token
report("sequential", lens, sequential_split(lens, 4))  # sequential: [5, 10, 20, 51], max/mean = 2.37
report("greedy", lens, greedy_balance(lens, 4))        # greedy: [22, 22, 21, 21], max/mean = 1.02
```

verl 用的是 Karmarkar–Karp，同样是近似最优的启发式，代码见 [seqlen_balancing.py](https://github.com/volcengine/verl/blob/main/verl/utils/seqlen_balancing.py)；里面也保留了一个 `greedy_partition`。

常见题：给定模型和 batch 配置，估算一步里各段的时间并指出瓶颈；解释 dynamic batch size 和 remove padding；写序列均衡分桶；GRPO 和 PPO 的计算量对比；colocate 下 `gpu_memory_utilization` 的上限。

## 参考

- [verl ray_trainer.py：各段计时](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py) · [metric_utils.py：timing / throughput 指标](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/metric_utils.py)
- [verl seqlen_balancing.py：Karmarkar–Karp、dynamic batch](https://github.com/volcengine/verl/blob/main/verl/utils/seqlen_balancing.py)
- [verl 性能调优指南](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)
- [HybridFlow（verl）](https://arxiv.org/abs/2409.19256)
- [vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
- [Largest differencing method（Karmarkar–Karp）](https://en.wikipedia.org/wiki/Largest_differencing_method)
- [Seer：rollout 时间分解](https://arxiv.org/abs/2511.14617)
- [DeepSeekMath（GRPO）](https://arxiv.org/abs/2402.03300)
