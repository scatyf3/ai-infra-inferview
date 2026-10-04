// 八股闪卡：推理系统、并行通信、GPU、框架、Post-train、系统设计的高频问答。
// 内容和数字取自站内对应文章（ref），改文章里的结论时记得同步这里。
//
// id 是复习记录的 key（src/data/flashcard-progress.json），改题面可以，改 id 会丢掉这张卡的记录。
// a 支持行内 `code` 和 **加粗**，\n 换行；code 是答案下面的代码块；ref 是出处（站内链接，不含 base）。

import type { Card } from '@lib/flashcards'

const ROOF = '/inference/prefill-decode-roofline'
const MEM = '/inference/memory-accounting'
const KV = '/inference/kv-cache-paged-attention'
const ATTN = '/inference/attention-variants'
const FA = '/inference/flash-attention'
const SCHED = '/inference/batching-scheduling'
const SPEC = '/inference/speculative-decoding'
const QUANT = '/inference/quantization'
const METRIC = '/inference/metrics-benchmark'
const TP = '/parallel/megatron-tp'
const PAR = '/parallel/parallelism-overview'
const COMM = '/parallel/collective-comm'
const MOE = '/parallel/moe-ep'
const ZERO = '/parallel/zero-fsdp'
const GPU = '/gpu/gpu-architecture'
const GEMM = '/gpu/tensor-core-gemm'
const GRAPH = '/gpu/cuda-graph-fusion'
const PROF = '/gpu/profiling'
const LIFE = '/framework/request-lifecycle'
const V1 = '/framework/vllm-v1-architecture'
const COMPILE = '/framework/torch-compile'
const PTI = '/framework/pytorch-internals'
const ADD = '/framework/add-model-vllm-sglang'
const TMEM = '/posttrain/training-memory'
const LORA = '/posttrain/lora-qlora'
const RLHF = '/posttrain/rlhf-ppo-dpo-grpo'
const RLI = '/posttrain/rl-infra'
const SFT = '/posttrain/sft'
const SD = '/basics/system-design-llm-serving'
const PY = '/basics/python'

const b = (c: Omit<Card, 'deck'>): Card => ({ deck: 'bagu', ...c })

export const baguCards: Card[] = [
  // ---------------- roofline ----------------
  b({
    id: 'bagu-ridge-point',
    topic: 'roofline',
    q: 'H100 的 ridge point 是多少？怎么用它判断一个算子的瓶颈？',
    a: '峰值算力 / 显存带宽 = 989 TFLOP/s ÷ 3.35 TB/s ≈ **295 FLOP/B**（bf16 dense）。\n算子的算术强度 AI = FLOPs / 访存字节：低于 295 是 **memory-bound**（算力在等数据），高于 295 是 **compute-bound**（带宽有富余）。\n可达性能 = min(峰值, 带宽 × AI)。',
    ref: ROOF,
  }),
  b({
    id: 'bagu-prefill-ai',
    topic: 'roofline',
    q: 'prefill 的算术强度约等于多少？为什么是 compute-bound？',
    a: '一次处理 S 个 token，每个权重读一次、对 S 个 token 各做一次乘加：`AI ≈ 2BS / b_w ~ S`（bf16）。S = 2048 时 AI 上千，远在 ridge 右边。\n所以 **TTFT ≈ FLOPs / (峰值 × MFU)**，优化方向是把 Tensor Core 喂饱：大 tile GEMM、FP8、chunk 别切太小。',
    ref: ROOF,
  }),
  b({
    id: 'bagu-decode-ai',
    topic: 'roofline',
    q: 'decode 的算术强度约等于多少？由此推出哪几类优化？',
    a: '每步每个序列只算 1 个 token，权重整读一遍：`AI ≈ 2B / b_w = B`（bf16），**就是 batch size**。batch 1 时比 ridge 低近 300 倍，是 GEMV 不是 GEMM。\n优化都围绕「少读字节 / 让一次读取被更多 token 用」：\n1. 加大 batch（continuous batching）\n2. 少读权重（W4A16 量化）、少读 KV（GQA / MLA / KV 量化）\n3. 一次验证多个 token（speculative decoding）',
    ref: ROOF,
  }),
  b({
    id: 'bagu-decode-step-time',
    topic: 'roofline',
    q: '口算：70B bf16、TP=4 的 H100，decode 一步的时间下界大约多少？',
    a: '权重 132 GiB ÷ 4 = 每卡 33 GiB；33 GiB ÷ 3.35 TB/s ≈ **10 ms**。\n再加上读 KV 和每层两次 all-reduce，实测 TPOT 约 15–25 ms。decode 的时间由**读多少字节**决定，不是算多少 FLOPs。',
    ref: SD,
  }),
  b({
    id: 'bagu-long-context-prefill',
    topic: 'roofline',
    q: '为什么长上下文的 prefill 特别贵？',
    a: 'prefill FLOPs ≈ `2PBS + 4LdS²B`，attention 项随 **S²** 增长，长了会反超线性的 GEMM 项。70B 在 8k 时 attention 约占 15%，32k 时约占 40%。\n这也是 FlashAttention 对 prefill 至关重要、长上下文要上 context parallel 的原因。',
    ref: ROOF,
  }),

  // ---------------- 显存账 ----------------
  b({
    id: 'bagu-kv-per-token',
    topic: '显存账',
    q: 'KV cache 每 token 多少字节？Llama-3-70B 是多少？',
    a: '`KV/token = 2 · L · H_kv · d_h · b`（2 是 K 和 V）。\nLlama-3-70B：`2 × 80 × 8 × 128 × 2 B` = **320 KiB/token**。\n总量 = KV/token × batch × 序列长度：8k × 32 = 26 万 token，正好 80 GiB。',
    ref: MEM,
  }),
  b({
    id: 'bagu-weight-rule',
    topic: '显存账',
    q: '权重显存怎么口算？70B 在 bf16 / int4 下各多大？',
    a: 'W = 参数量 × 每参数字节。**bf16 下 GiB 数 ≈ 参数量（B）× 2**。\n70B bf16 ≈ 131 GiB（70.55e9 × 2 ÷ 1.07e9）；int4 是 0.5 字节/参数，约 33 GiB 再加上 scale。',
    ref: MEM,
  }),
  b({
    id: 'bagu-how-many-gpus',
    topic: '显存账',
    q: '70B bf16、8k context、batch 32，H100 80G 要几张？',
    a: '推理显存 = 权重 + KV + 激活 + 固定开销，**唯一随 batch 和 context 线性涨的是 KV**。\n权重 132 GiB + KV `320 KiB × 8192 × 32` = 80 GiB ≈ 212 GiB，加激活和开销约 220 GiB → **4 张**（TP=4）。2 张连权重都放不下；3 张显存勉强够，但 TP 要整除 head 数（64 个 Q head、8 个 KV head），3 不行。',
    ref: MEM,
  }),
  b({
    id: 'bagu-layer-params',
    topic: '显存账',
    q: '一层 decoder（GQA + SwiGLU）的参数量怎么算？常见算错点是什么？',
    a: '`2d²`（W_q、W_o）+ `2·d·H_kv·d_h`（W_k、W_v）+ `3·d·d_ff`（SwiGLU）+ `2d`（两个 norm）。再加 embedding `V·d`（不共享就 ×2）。\n算错点：**SwiGLU 是三个矩阵**（gate、up、down），不是两个；GQA 下 K、V 投影比 Q 小。',
    ref: MEM,
  }),
  b({
    id: 'bagu-activation-small',
    topic: '显存账',
    q: '推理时激活为什么不是显存大头？哪一项要留意？',
    a: '不做反向，不用保存中间结果；FlashAttention 不物化 S×S 的 score。每层活跃张量约 `B·S_chunk·(4d + 2d_ff)·b`，用完即丢。\n要留意的是 **logits**：`B × V × 4` 字节（fp32），batch 256 × 128k vocab 就是 128 MiB。',
    ref: MEM,
  }),

  // ---------------- KV cache ----------------
  b({
    id: 'bagu-paged-why',
    topic: 'PagedAttention',
    q: 'PagedAttention 解决了什么问题？怎么解决的？',
    a: '以前按 `max_model_len` 给每个请求预留连续大块，输出长度未知，实际利用率只有 **20–40%**。\nPagedAttention 像 OS 分页：KV 切成固定大小的 block（vLLM 默认 16 token），每个序列一张 **block table** 做逻辑→物理映射，kernel 按表 gather。外部碎片归零，内部碎片上限是每序列半个 block。\n顺带解锁了 prefix 共享（copy-on-write）和细粒度抢占。',
    ref: KV,
  }),
  b({
    id: 'bagu-block-size',
    topic: 'PagedAttention',
    q: 'block_size 为什么不设成 1？设太大又有什么问题？',
    a: '太小：block table 变长，kernel 里间接寻址和 gather 的开销大，访存不连续。\n太大：每个序列最后一个没填满的 block 浪费更多（平均 `block_size / 2` 个 token），prefix 共享的粒度也变粗。16 是折中。',
    ref: KV,
  }),
  b({
    id: 'bagu-prefix-cache',
    topic: 'PagedAttention',
    q: 'prefix caching 怎么实现？为什么只有满 block 参与共享？',
    a: '满 block 按「前缀 + 本块 token」的内容算 hash，命中就把物理 block 号直接填进新序列的 block table，refCount + 1，省掉这段 prefill。\n写 refCount > 1 的 block 要 **copy-on-write**。未满 block 还会被追加写，内容不固定，所以不参与共享。',
    ref: KV,
  }),
  b({
    id: 'bagu-radix',
    topic: 'PagedAttention',
    q: 'SGLang 的 RadixAttention 和 vLLM 的 block hash 前缀缓存有什么区别？',
    a: 'RadixAttention 用 **radix tree** 管理所有前缀，支持任意长度的部分匹配（不要求 block 对齐），LRU 淘汰叶子。\n收益场景：多轮对话、长 system prompt 的 agent、树搜索类推理（共享分支前缀）。两家现在思路已经趋同。',
    ref: KV,
  }),
  b({
    id: 'bagu-preempt-swap-recompute',
    topic: 'PagedAttention',
    q: '显存不够要抢占时，swap 和 recompute 怎么选？vLLM 默认哪个？',
    a: '**swap**：把 block 拷到 CPU，之后拷回；代价是两趟 PCIe，适合很长的序列（重算太贵）。\n**recompute**：直接丢 block，恢复时把已生成的 token 当 prompt 重新 prefill；代价是算力和 TTFT 尖刺，适合短序列或能命中 prefix cache 的。\nvLLM 默认 **recompute**：PCIe 往返通常比重算还慢，而且重算能吃到 prefix cache。',
    ref: KV,
  }),
  b({
    id: 'bagu-kv-quant',
    topic: 'PagedAttention',
    q: 'KV cache 量化到 FP8 有什么收益和注意点？',
    a: '收益：KV 的显存和访存减半，等于 decode 的算术强度翻倍、能放下的并发翻倍。\n注意：\n1. 用 per-token / per-head 量化，不要 per-tensor，outlier 集中在少数 channel。\n2. **K 比 V 难量化**：K 在 softmax 之前参与点积，误差被指数放大。\n3. 早期 token 的 KV 被后续每一步反复读，长 context 下误差影响更大。',
    ref: KV,
  }),

  // ---------------- attention 变体 ----------------
  b({
    id: 'bagu-gqa',
    topic: 'attention 变体',
    q: 'MHA / GQA / MQA 的区别？为什么只减 K、V 的 head，不减 Q 的？',
    a: 'decode 时 Q 只有当前 token，算完就丢；K、V 要留着给后面每个 token 用。所以 **KV cache 大小只由 K/V 的 head 数决定**，Q 保持 H 个 head 不损失表达力。\nLlama-3-70B 规格下每 token：MHA（64 个 KV head）2.5 MiB，GQA（8 个）320 KiB，MQA（1 个）40 KiB。GQA 是质量和访存的工业折中。',
    ref: ATTN,
  }),
  b({
    id: 'bagu-gqa-kernel',
    topic: 'attention 变体',
    q: 'GQA 实现上要不要真的 `repeat_interleave` 复制 K/V？',
    a: '不要。好的 kernel（FlashAttention）让同组的几个 Q head **读同一块 K/V tile**，省的是 HBM 流量而不只是显存。真复制一遍就把 GQA 省下的访存又花回去了。',
    ref: ATTN,
  }),
  b({
    id: 'bagu-mla',
    topic: 'attention 变体',
    q: 'MLA 缓存什么？decode 时怎么避免把 K 展开？',
    a: '只缓存下投影后的 latent `c_kv`（DeepSeek-V3 里 d_c = 512），加一个所有 head 共享、单独加 RoPE 的 `k_pe`（64 维），每 token 576 个元素，约 **68.6 KiB/token**（61 层），比同规模 GQA 还小。\ndecode 用 **weight absorption**：把 `W_UK` 吸收进 Q 侧，`qᵀKᵀ = (qᵀW_UKᵀ) c_kvᵀ`，直接在 latent 空间算 score。\n代价是计算量比 GQA 大，拿算力换带宽，正适合 memory-bound 的 decode。prefill 反而直接展开更快。',
    ref: ATTN,
  }),
  b({
    id: 'bagu-mla-rope',
    topic: 'attention 变体',
    q: 'MLA 为什么需要 decoupled RoPE？',
    a: 'RoPE 是位置相关的旋转，夹在 `c_kv` 和 `W_UK` 之间就没法把 `W_UK` 吸收进 Q 侧（矩阵乘的顺序换不过来）。\n解法：另拿一小段维度（d_r = 64）单独加 RoPE，所有 head 共享、单独缓存；其余部分不带位置信息，照常吸收。',
    ref: ATTN,
  }),
  b({
    id: 'bagu-gqa-tp',
    topic: 'attention 变体',
    q: 'GQA 模型做 TP 时对切分数有什么限制？',
    a: '`H` 和 `H_kv` 都要能被 TP 整除。KV head 只有 8 个、TP 开到 16 时，KV head 不够分，只能在卡之间**复制**，KV cache 跟着复制，白占显存。所以 TP 一般不超过 H_kv。',
    ref: TP,
  }),

  // ---------------- FlashAttention ----------------
  b({
    id: 'bagu-fa-why',
    topic: 'FlashAttention',
    q: 'FlashAttention 省的是什么？为什么能省？',
    a: '省的是 **HBM 读写**，不是 FLOPs（还多了点重算）。\n标准实现把 S×S 的 score 写回 HBM 再读回来做 softmax，流量 O(S²)；FlashAttention 把 Q、K、V 分块搬进 SRAM，用 **online softmax** 在块间增量更新 max 和分母，score 不落盘，流量降到 O(S·d)。长序列 attention 是 memory-bound，所以快。',
    ref: FA,
  }),
  b({
    id: 'bagu-fa-v2-v3',
    topic: 'FlashAttention',
    q: 'FlashAttention v2、v3 分别改了什么？',
    a: '**v2**：调换循环顺序，外层遍历 Q 块（每个 warp 负责一段 Q 行），K/V 在内层流过；减少 shared memory 同步和非 matmul 的 FLOPs，约快 2 倍。\n**v3**：针对 Hopper，用 TMA 异步搬数据、wgmma、warp specialization 让 softmax 和 GEMM 流水重叠，并支持 FP8。',
    ref: FA,
  }),
  b({
    id: 'bagu-flash-decoding',
    topic: 'FlashAttention',
    q: 'decode 阶段 FlashAttention 帮助大吗？用什么？',
    a: '帮助有限：Q 只有 1 行，score 是 1×S，本来就小，瓶颈是读整个 KV cache 的带宽。\n用 **FlashDecoding**（split-KV）：沿 KV 长度切成多段，多个 block 并行读、各算局部结果，再用 online softmax 的方式合并，让足够多的 SM 一起读 KV。',
    ref: FA,
  }),

  // ---------------- 调度 ----------------
  b({
    id: 'bagu-continuous-batching',
    topic: '调度',
    q: 'continuous batching 和静态 batching 的区别？收益从哪来？',
    a: '静态 batching 要等一整批都生成完才换下一批，最长的请求拖住所有人。continuous batching（iteration-level scheduling）**每生成一步就重新组 batch**：结束的踢出，等待的立刻补进。\n收益来自 roofline：decode 的 AI ≈ batch，batch 从 4 涨到 64 就是 16 倍的算力利用率。实测吞吐 2–4 倍，请求越异构收益越大。',
    ref: SCHED,
  }),
  b({
    id: 'bagu-sched-budgets',
    topic: '调度',
    q: '`max_num_batched_tokens` 和 `max_num_seqs` 分别管什么？',
    a: '`max_num_batched_tokens`：每轮 prefill + decode 的 **token 总预算**，算力侧的闸门；开 chunked prefill 时主要约束 chunk 大小。\n`max_num_seqs`：同时 running 的**序列数**上限，KV 侧的闸门，决定 decode 的 batch 上限，也就决定了 decode 的 AI 能到多高。两者要配合 KV block 数一起调。',
    ref: SCHED,
  }),
  b({
    id: 'bagu-chunked-prefill',
    topic: '调度',
    q: 'chunked prefill 解决什么问题？代价是什么？',
    a: '问题：一个 4k prompt 的 prefill 要上百毫秒，这期间所有 decode 请求都在等，**ITL 出现尖刺**。\n做法：prefill 切成 512–2048 token 的 chunk，每轮只做一个 chunk，剩下预算给 decode，一个 batch 里 prefill 和 decode 混跑（attention 走 varlen）。\n代价：每个 chunk 都要重读已有的 KV，prefill 总耗时略增；chunk 不能太小，否则 AI 掉到 ridge 以下。',
    ref: SCHED,
  }),
  b({
    id: 'bagu-pd-disagg',
    topic: '调度',
    q: 'PD 分离是什么？收益和代价各是什么？',
    a: 'prefill 和 decode 跑在**不同的 GPU 池**：prefill 池 compute-bound、追求 TTFT，用高算力卡、大 TP；decode 池 memory-bound、追求 TPOT，用大显存高带宽、多副本大 batch。\n收益：两边独立选型、独立扩缩，prefill 的长尾不再污染 decode 的 ITL。\n代价：**KV 要跨节点传**。70B GQA、2k prompt 是 640 MiB，400 Gbps IB 约 13 ms，通常按层传输、和 prefill 后几层 overlap。MLA 的 KV 小，对 PD 分离特别友好。',
    ref: SCHED,
  }),
  b({
    id: 'bagu-chunked-vs-pd',
    topic: '调度',
    q: 'chunked prefill 和 PD 分离，什么时候选哪个？',
    a: '单机或卡少：**chunked prefill**，没有额外传输和运维复杂度。\n规模大、TTFT 和 TPOT 有独立 SLA：**PD 分离**，两边可以独立调优和扩缩。门槛是要有 RDMA 和一套 KV 传输层，小集群不划算。',
    ref: SCHED,
  }),
  b({
    id: 'bagu-preempt-whom',
    topic: '调度',
    q: '抢占时踢谁？频繁抢占说明什么？',
    a: '通常 **LIFO**，踢最新进来的：老请求已经投入了更多计算，踢它浪费更大，也避免老请求被反复饿死。\n生产上频繁抢占说明 `max_num_seqs` 设得太激进或 KV 空间不够，该调参或加卡，而不是让调度器反复抖动。',
    ref: SCHED,
  }),

  // ---------------- 投机解码 ----------------
  b({
    id: 'bagu-spec-why',
    topic: '投机解码',
    q: 'speculative decoding 为什么能「白赚」？',
    a: 'decode 是 memory-bound，Tensor Core 基本在空转。target 一次前向验证 k 个 token，读的字节数几乎不变（权重还是读一遍），FLOPs 涨 k 倍，AI 从 B 变成 kB。原本在 roofline 左侧有大量余量，这 k 倍 FLOPs 几乎免费。\n本质是**用闲置算力换延迟**，把 k 次 GEMV 变成 1 次小 GEMM。',
    ref: SPEC,
  }),
  b({
    id: 'bagu-spec-exact',
    topic: '投机解码',
    q: '投机采样怎么保证输出分布和 target 单独采样完全一样？',
    a: 'rejection sampling：draft 分布 q，target 分布 p，对每个位置：\n1. 以概率 `min(1, p(x)/q(x))` 接受 draft 的 token x；\n2. 否则拒绝，从残差分布 `norm(max(0, p − q))` 重新采样，后面的 draft token 全部作废。\n可以证明输出分布**严格等于 p**，是精确加速，不用重新评测质量。',
    ref: SPEC,
  }),
  b({
    id: 'bagu-spec-expected',
    topic: '投机解码',
    q: '接受率 α、猜 k 个，一轮平均产出几个 token？k 越大越好吗？',
    a: '`E[tokens] = (1 − α^(k+1)) / (1 − α)`。α = 0.8、k = 4 时约 3.4 个；α = 0.5 时只有 1.9 个。\n加速比还要除以 draft 成本 `c·k + 1`。k 大了 α^k 衰减而 draft 成本线性涨，典型最优 k 在 **3–5**。α 太低（< 0.3）时净收益为负。',
    ref: SPEC,
  }),
  b({
    id: 'bagu-eagle-medusa',
    topic: '投机解码',
    q: 'Medusa、EAGLE、MTP 的 draft 分别怎么来？核心区别？',
    a: '**Medusa**：在最后一层 hidden 上挂多个独立的头，第 i 个头猜第 i+1 个位置，各头互不看对方，α 偏低。\n**EAGLE**：在**特征层**做自回归，输入上一步特征 + 已采样 token 的 embedding，保留了序列依赖，α 明显更高。\n**MTP**：训练时就让模型多预测几个位置（DeepSeek-V3），推理时直接当 draft，α 约 0.85，额外开销极小。\n一句话：Medusa 在 token 空间并行猜，EAGLE 在特征空间自回归猜。',
    ref: SPEC,
  }),
  b({
    id: 'bagu-spec-large-batch',
    topic: '投机解码',
    q: '什么场景下投机解码没用甚至变慢？',
    a: '**大 batch**。batch 一两百时 decode 的 AI 已经接近 ridge，算力不再闲置，投机多出的 FLOPs 变成真实成本；batch 内每个序列接受长度不同，要做 ragged 处理。\n所以它是低延迟（小 batch、交互式）的手段，生产上按 batch 大小动态开关、在线统计接受率调 k。',
    ref: SPEC,
  }),

  // ---------------- 量化 ----------------
  b({
    id: 'bagu-w4a16-vs-w8a8',
    topic: '量化',
    q: 'W4A16 和 W8A8 / FP8 各适合什么场景？',
    a: '**W4A16**（weight-only）：权重 4 bit、计算前反量化成 bf16。decode memory-bound，读权重的字节少了四分之三，直接提速；显存紧、batch 小时首选。\n**W8A8 / FP8**：权重和激活都 8 bit，用 Tensor Core 原生低精度计算，prefill（compute-bound）约快 2 倍；吞吐优先、大 batch 时选。\nW4A16 在大 batch 下可能比 bf16 还慢：瓶颈变成计算，反量化是额外开销。',
    ref: QUANT,
  }),
  b({
    id: 'bagu-gptq-awq',
    topic: '量化',
    q: 'GPTQ 和 AWQ 分别怎么降低 4 bit 权重量化的误差？',
    a: '**GPTQ**：逐层、逐列量化，用二阶信息（Hessian）把当前列的量化误差补偿到还没量化的列上。\n**AWQ**：按**激活幅度**找出少数重要的权重通道，量化前把它们放大（激活侧除回去），让重要权重的相对误差变小。不用反向，校准快。\n两者通常配 per-group（group 128）的 scale。',
    ref: QUANT,
  }),
  b({
    id: 'bagu-smoothquant',
    topic: '量化',
    q: 'SmoothQuant 解决什么问题？怎么做的？',
    a: '问题：激活有离群 channel，per-tensor 量化到 int8 误差很大；权重则比较平滑。\n做法：数学等价变换 `Y = (X·diag(s)⁻¹)(diag(s)·W)`，按通道把激活的难度「搬」一部分到权重上，`s_j = max|X_j|^α / max|W_j|^(1−α)`，α 常取 0.5。之后激活和权重都好量化，可以跑 W8A8。',
    ref: QUANT,
  }),
  b({
    id: 'bagu-quant-granularity',
    topic: '量化',
    q: 'per-tensor / per-channel / per-group 量化的取舍？',
    a: '粒度越细，scale 越多、元数据越大，误差越小。\nper-tensor：一个 scale，最省但最容易被离群值拖累。per-channel：每个输出通道一个，权重常用。per-group：每 128 个元素一个，4 bit 权重的标配。\nKV cache 用 per-token / per-head。',
    ref: QUANT,
  }),

  // ---------------- 指标 ----------------
  b({
    id: 'bagu-metrics',
    topic: '指标',
    q: 'TTFT、TPOT、ITL、E2E 分别是什么？各自主要受什么影响？',
    a: '**TTFT**：第一个 token 返回的时间，受排队和 prefill 影响。\n**TPOT**：后续每个 token 的平均间隔，受 decode 每步时间（batch 大小、读多少字节）影响。\n**ITL**：单个 token 间隔，看 P99 抖动，受混进来的 prefill 影响。\n**E2E** = TTFT + TPOT × 输出长度。',
    ref: METRIC,
  }),
  b({
    id: 'bagu-goodput',
    topic: '指标',
    q: '什么是 goodput？为什么调参要看它而不是吞吐？',
    a: 'goodput = **满足 SLA 的请求**的吞吐（比如 TTFT < 1 s 且 TPOT < 50 ms）。\n加大 batch 往往吞吐涨了，但每步变慢、TPOT 超标的请求变多、抢占增加，goodput 反而降。只看裸吞吐会让「牺牲少数请求换总量」的改动看起来是正收益。',
    ref: METRIC,
  }),
  b({
    id: 'bagu-benchmark-design',
    topic: '指标',
    q: '怎么设计一次推理服务的压测？',
    a: '1. 输入输出长度用真实 trace 或 ShareGPT 分布，别用固定长度。\n2. 按泊松到达**扫 QPS**，每个点跑够久、充分 warmup。\n3. 画 QPS 对 P50 / P99 的 TTFT 和 TPOT 曲线，报 goodput。\n4. 确认客户端不是瓶颈。',
    ref: METRIC,
  }),
  b({
    id: 'bagu-ttft-parts',
    topic: '指标',
    q: 'TTFT 由哪几段组成？高负载下大头通常是哪段？',
    a: '排队 + tokenize + prefill 计算 + 采样 + detokenize + 网络。\n高负载下大头往往是**排队**，不是计算。所以过载时要做准入控制（直接 429），而不是让队列越排越长、P99 雪崩。',
    ref: LIFE,
  }),

  // ---------------- 并行 ----------------
  b({
    id: 'bagu-megatron-tp',
    topic: '并行',
    q: 'Megatron TP 为什么每层只需要两次 all-reduce？',
    a: '**先 column 后 row**。第一个矩阵按列切，每卡拿完整输入、算出输出的一部分列，无通信；第二个矩阵按行切，输入正好是上一步的列分块，每卡得到部分和，最后一次 all-reduce。\nMLP（up/gate 列切，down 行切）一次，attention（QKV 按 head 列切，W_o 行切）一次，每层共两次。\n前提：中间的激活是 elementwise 的；顺序反过来就要两次。',
    ref: TP,
  }),
  b({
    id: 'bagu-tp-comm-size',
    topic: '并行',
    q: 'TP 每次 all-reduce 的通信量和什么有关？和模型大小有关吗？',
    a: '消息大小是一份完整激活 `B · S · d · b`，**和参数量无关**。参数量靠加层变大，只增加通信次数；序列变长才让单次变大。\n70B、batch 8、2k token、bf16：一次 256 MiB，每层两次、80 层就是 160 次，所以 TP 必须在 NVLink 域内。',
    ref: PAR,
  }),
  b({
    id: 'bagu-ring-allreduce',
    topic: '并行',
    q: 'ring all-reduce 每张卡的通信量是多少？为什么说它带宽最优？',
    a: '拆成 reduce-scatter + all-gather 两个阶段，各 N−1 步，每步收发 1/N 的数据，每卡总量 **`2(N−1)/N × D`**，几乎和卡数无关。\n缺点是延迟随 N 线性增长，小消息用 tree 算法更好。NCCL 会按拓扑和消息大小自动选。',
    ref: COMM,
  }),
  b({
    id: 'bagu-tp-intra-node',
    topic: '并行',
    q: '为什么 TP 一般不跨节点，PP 可以？',
    a: 'TP 每层两次 all-reduce，量大且在关键路径上；节点内 NVLink（H100 900 GB/s），跨节点 IB 每端口约 50 GB/s，差一个数量级，跨节点会被通信打死，所以 TP ≤ 8。\nPP 只在 stage 边界传一次激活（P2P），量小两个数量级，还能和计算流水重叠，可以走 IB。',
    ref: COMM,
  }),
  b({
    id: 'bagu-pp-bubble',
    topic: '并行',
    q: 'PP 的 bubble 有多大？怎么减小？',
    a: '`bubble = (p − 1) / m`，p 是 stage 数，m 是 micro-batch 数，要 m ≫ p 才划算。\n**1F1B** 把峰值激活显存从 O(m) 降到 O(p)；**interleaved**（virtual pipeline）把 bubble 再除以 v，代价是通信次数 ×v。\n推理里 decode 的 batch 小、bubble 摊不掉，PP 只在显存实在放不下时用。',
    ref: PAR,
  }),
  b({
    id: 'bagu-sp-cp',
    topic: '并行',
    q: 'sequence parallel（SP）和 context parallel（CP）都切序列，区别在哪？',
    a: '**SP** 是 TP 的补充：TP 下 LayerNorm、dropout 在每卡重复算、激活完整复制。SP 把这段按序列切，把 all-reduce 拆成 reduce-scatter + all-gather，**通信总量不变、激活降到 1/N**，基本白捡。\n**CP / Ring Attention** 给超长上下文用：序列切 N 段，每卡持一段 Q/K/V，环形轮转 K/V，每轮算局部 attention、用 online softmax 合并，通信和计算 overlap。',
    ref: PAR,
  }),
  b({
    id: 'bagu-parallel-choice',
    topic: '并行',
    q: '给一个大模型选并行方式，按什么顺序定？',
    a: '1. **TP** 填满 NVLink 域（≤ 8），且不超过 H_kv。\n2. 还放不下就加 **PP**，跨节点走 IB。\n3. 剩下的卡都给 **DP** 扩吞吐（推理时副本间零通信）。\n4. MoE 的 expert 单独用 **EP**，和 attention 的并行解耦；超长上下文加 **CP**。\n原则：什么放不下、什么链路快，共同决定切哪一维。',
    ref: PAR,
  }),
  b({
    id: 'bagu-moe-ep',
    topic: '并行',
    q: 'MoE 的 Expert Parallel 通信模式是什么？最大的麻烦是什么？',
    a: '每个 MoE 层两次 **all-to-all**：dispatch 把 token 发到专家所在的卡，combine 把结果收回；通信量 ∝ token 数 × top-k × hidden。\n最大的麻烦是**负载不均**：热门专家所在的卡成为短板，其他卡空等。对策：训练时加辅助 loss / bias 调整；capacity factor 丢 token；推理时复制热门专家、重排（EPLB）。',
    ref: MOE,
  }),
  b({
    id: 'bagu-moe-batch',
    topic: '并行',
    q: '为什么 MoE 推理要大 batch、大 EP 才划算？DP attention 是什么？',
    a: 'batch 小时每个专家只分到几个 token，读了整个专家权重却只算几个 token，AI 极低。大 EP 把所有卡的 HBM 带宽凑起来读专家权重，大 batch 让每个专家分到足够多的 token。\n**DP attention**：attention 权重小，用 DP 而不是 TP，避免 KV 被 TP 复制；expert 部分用 EP。DeepSeek 的部署就是这样。',
    ref: MOE,
  }),
  b({
    id: 'bagu-zero',
    topic: '并行',
    q: 'ZeRO 1 / 2 / 3 各切什么？每卡显存分别是多少？',
    a: 'Adam 混合精度每参数 16 字节：参数 2 + 梯度 2 + 优化器 12（fp32 主权重、m、v）。\n**ZeRO-1** 切优化器状态：`4Ψ + 12Ψ/N`。\n**ZeRO-2** 再切梯度：`2Ψ + 14Ψ/N`。\n**ZeRO-3** 连参数也切：`16Ψ/N`，每层前向反向前 all-gather 参数，通信量是普通 DP 的 **1.5 倍**。FSDP 是 PyTorch 原生的 ZeRO-3。',
    ref: ZERO,
  }),
  b({
    id: 'bagu-zero3-vs-tp',
    topic: '并行',
    q: 'ZeRO-3 和 TP 都把参数切到多卡，本质区别？',
    a: 'ZeRO-3 切的是**存储**：计算前把整层参数 gather 回来，每卡算完整的层，通信的是参数（和 batch 无关）。\nTP 切的是**计算**：每卡只算自己那片，all-reduce 合并结果，通信的是激活（和 batch 成正比）。\n所以小 batch 用 TP 划算，大 batch 用 ZeRO 划算。',
    ref: ZERO,
  }),

  // ---------------- GPU ----------------
  b({
    id: 'bagu-gpu-three-basics',
    topic: 'GPU',
    q: '写 CUDA kernel 的三个基本功是什么？',
    a: '1. **访存合并（coalescing）**：一个 warp 的 32 个线程读连续地址，合成少数几次 128 B 事务；跨步访问会把事务数放大几十倍。\n2. **避免 bank conflict**：shared memory 分 32 个 bank，同一 warp 多线程落在同一 bank 的不同地址会串行，常用 padding 一列解决。\n3. **occupancy 够用**：每个 SM 驻留足够多的 warp 来藏访存延迟，受寄存器和 shared memory 用量限制，不是越高越好。',
    ref: GPU,
  }),
  b({
    id: 'bagu-warp-divergence',
    topic: 'GPU',
    q: '什么是 warp divergence？causal attention 为什么不怎么受影响？',
    a: '同一 warp 内线程走不同分支时，硬件串行执行两条路径，吞吐减半。\ncausal mask 按 tile 处理：大部分 tile 要么全可见、要么全被 mask（直接跳过），只有对角线上的 tile 内部有分支，代价很小。',
    ref: GPU,
  }),
  b({
    id: 'bagu-gemv',
    topic: 'GPU',
    q: '为什么 decode 的 GEMV 打不满 Tensor Core？',
    a: 'GEMM 靠 tiling 让 A、B 的块在 shared memory / 寄存器里反复复用，强度 ∝ `BM·BN / (BM + BN)`。GEMV 里 M = batch 很小，每个权重只用一次，没有复用，强度约等于 batch，带宽先打满，Tensor Core 大部分时间在等数据。\n对策：加大 batch、W4A16 少读字节、split-K 让更多 SM 参与读权重。',
    ref: GEMM,
  }),
  b({
    id: 'bagu-cuda-graph',
    topic: 'GPU',
    q: 'CUDA Graph 解决什么问题？为什么只用在 decode？',
    a: 'decode 一步只有几毫秒，几百个小 kernel 的 launch 开销和 Python 调度能占一大块，GPU 在等 CPU。CUDA Graph 把整个 forward 录下来，一次提交。\n要求 shape 固定。decode 每个序列只有 1 个 token，只有 batch 一个维度在变，按 batch 分桶预录几张图、padding 到桶大小即可（memory-bound 下 padding 几乎免费）。prefill 的长度千变万化，录不过来。',
    ref: GRAPH,
  }),
  b({
    id: 'bagu-kernel-fusion',
    topic: 'GPU',
    q: '为什么推理框架要把残差加和 RMSNorm 融合成一个 kernel？',
    a: '两者都是 memory-bound 的逐元素 / 行操作。分开写要把残差和写回 HBM 再读出来做 norm，多一次读写；融合后中间结果留在寄存器里。\n收益按 roofline 算：时间 ∝ 读写字节数，decode 下每层都省一点，累积可观。SiLU × up、RoPE + 写 KV 也是同理。',
    ref: GRAPH,
  }),
  b({
    id: 'bagu-profiling',
    topic: 'GPU',
    q: 'nsys、ncu、torch profiler 各看什么？排查顺序？',
    a: '**torch profiler**：Python 算子和 CPU / GPU 时间线，能对回代码行。\n**nsys**：kernel 序列、launch 间隙、拷贝和 NCCL 有没有 overlap。\n**ncu**：单个 kernel 的 DRAM / SM 吞吐、occupancy、stall 原因。\n顺序：先看时间线上 GPU 有没有空转（有空隙就是 CPU 侧瓶颈，上 CUDA Graph），再用 ncu 看单个 kernel 慢在哪，放到 roofline 上判断。',
    ref: PROF,
  }),

  // ---------------- 框架 ----------------
  b({
    id: 'bagu-request-lifecycle',
    topic: '框架',
    q: '一个请求在 vLLM 里从 HTTP 到第一个 token，经过哪几步？',
    a: 'HTTP 到达 → 校验、渲染 chat template → tokenize → 进 waiting 队列（IPC 到 EngineCore）→ 调度器分配 KV block（先查 prefix cache）→ 组 batch、准备输入拷到 GPU → prefill 前向 → 采样 → detokenize、SSE 推出第一个 token（**TTFT 到此结束**）。\n之后每轮 decode 一个 token，结束后释放 block。',
    ref: LIFE,
  }),
  b({
    id: 'bagu-tokenize-event-loop',
    topic: '框架',
    q: 'API server 里 tokenize 为什么不能直接在 asyncio 事件循环里做？',
    a: 'tokenize 是 CPU 密集操作，长 prompt 要几毫秒到几十毫秒。事件循环是单线程的，**一个同步慢调用会卡住所有连接**。\n要么放进线程池（HF tokenizers 是 Rust 实现，会释放 GIL），要么放独立进程。vLLM V1 把 tokenize / detokenize 放在单独的前端进程里。',
    ref: PY,
  }),
  b({
    id: 'bagu-vllm-v1',
    topic: '框架',
    q: 'vLLM V1 的进程架构？为什么把 EngineCore 单独放一个进程？',
    a: '前端进程（HTTP、tokenize、detokenize）↔ ZMQ ↔ **EngineCore 进程**（调度 + KV 管理）→ 每张卡一个 worker 进程（model runner）。\nV0 里这些在同一个 Python 进程里串行，CPU 干活时 GPU 空转。拆开后 CPU 工作和 GPU 执行重叠，GPU 利用率明显提升；代价是多一次进程间序列化。\nV1 的 scheduler 不再区分 prefill / decode batch，每个请求每步分配若干 token，chunked prefill 和 prefix caching 默认开。',
    ref: V1,
  }),
  b({
    id: 'bagu-flat-input',
    topic: '框架',
    q: 'vLLM 的模型 forward 为什么输入是 `[num_tokens, H]` 而不是 `[B, S, H]`？',
    a: 'continuous batching 下一个 batch 里既有 prefill 的长序列又有 decode 的单 token，padding 成矩形会浪费大量算力。\n所以把所有请求的 token 拼成一维，再用 metadata（每个序列的起止、slot mapping、block table）告诉 attention kernel 怎么分组；Linear 层本来就和 token 怎么分组无关。',
    ref: ADD,
  }),
  b({
    id: 'bagu-torch-compile',
    topic: '框架',
    q: 'torch.compile 的三段分别做什么？什么是 graph break？',
    a: '**Dynamo**：在 CPython 的 frame evaluation 钩子上符号执行字节码，抓出 FX 图，记下 guard，下次 guard 命中直接跑编译好的图。\n**AOTAutograd**：把前向和反向一起 trace 成 ATen 算子图。\n**Inductor**：融合 pointwise / reduction，生成 Triton（GPU）或 C++（CPU）kernel。\n**graph break**：遇到数据相关的控制流（如 `if x.sum() > 0`）、print、不支持的调用时切断图，前后各编一段；太多就退化成 eager。',
    ref: COMPILE,
  }),
  b({
    id: 'bagu-caching-allocator',
    topic: '框架',
    q: 'PyTorch 的 caching allocator 做了什么？显存碎片怎么看出来？',
    a: '在 `cudaMalloc` 之上按 stream 维护分桶的空闲块池，释放时不还给驱动，下次直接复用，避免 cudaMalloc / cudaFree 的同步开销。\n碎片表现为 **reserved 远大于 allocated**。缓解：`expandable_segments`、尽量固定 shape、推理框架启动时一次性预分配 KV 池。`empty_cache()` 解决不了碎片，还会让后续分配变慢。',
    ref: PTI,
  }),

  // ---------------- Post-train ----------------
  b({
    id: 'bagu-train-memory',
    topic: 'Post-train',
    q: '混合精度 Adam 训练，每个参数占多少显存？7B 全参微调要多少？',
    a: '**16 字节**：bf16 参数 2 + bf16 梯度 2 + fp32 主权重 4 + Adam 的 m、v 各 4。7B 就是 112 GB，还没算激活，单卡放不下。\n激活随 batch × seq × 层数涨，gradient checkpointing 只存每层输入、反向时重算，用约 30% 的额外计算换大幅省激活。',
    ref: TMEM,
  }),
  b({
    id: 'bagu-bf16-fp16',
    topic: 'Post-train',
    q: '为什么 fp16 训练要 loss scaling，bf16 不用？那 bf16 为什么还要 fp32 主权重？',
    a: 'fp16 指数只有 5 位，小梯度会下溢成 0，所以先把 loss 乘大再除回来。bf16 指数 8 位、和 fp32 动态范围一样，不会下溢。\n但 bf16 尾数只有 7 位，`lr × grad` 常小于参数的分辨率，直接加会被舍掉，所以用 fp32 主权重累加更新，前向再 cast 回 bf16。',
    ref: TMEM,
  }),
  b({
    id: 'bagu-lora',
    topic: 'Post-train',
    q: 'LoRA 怎么初始化？它省了哪些显存，没省哪些？',
    a: '冻结 W，训练低秩的 `B·A`（秩 r 一般 8–64），forward 是 `Wx + (α/r)·BAx`。**A 随机、B 全零**，训练开始时 BA = 0，模型行为和原模型一致。\n省的是可训练参数的梯度和优化器状态（缩小上百倍）。**没省激活**：算 A 的梯度要用每层的输入 x，激活和全参一样要存。QLoRA 再把冻结的 W 压成 4 bit NF4。',
    ref: LORA,
  }),
  b({
    id: 'bagu-multi-lora',
    topic: 'Post-train',
    q: '一个服务里同时跑很多个 LoRA adapter，怎么做？',
    a: '所有请求共享同一份 base 权重，不合并 adapter；不同请求的 adapter 计算用 batched GEMV（Punica 的 SGMV、S-LoRA）合到一个 kernel 里，adapter 权重按需换入换出显存。\n只服务一个 adapter 时，直接把 BA 加回 W，零额外开销。',
    ref: LORA,
  }),
  b({
    id: 'bagu-ppo-four-models',
    topic: 'Post-train',
    q: 'PPO-RLHF 里有哪四个模型？KL 约束的作用？',
    a: '**policy**（训练）、**reference**（冻结，算 KL）、**reward**（冻结，打分）、**value / critic**（训练，估 advantage 的基线）。\nreward 里减去 `β · KL(policy ‖ reference)`，防止 policy 为了刷分跑偏（reward hacking）。β 太大学不动，太小会跑偏。',
    ref: RLHF,
  }),
  b({
    id: 'bagu-dpo',
    topic: 'Post-train',
    q: 'DPO 的 loss 是什么？和 PPO 比优缺点？',
    a: '把 Bradley-Terry 偏好模型和带 KL 约束的最优解代回去，只剩 policy 和 reference 的对数概率：\n`L = −log σ(β[(log π(y_w|x) − log π_ref(y_w|x)) − (log π(y_l|x) − log π_ref(y_l|x))])`\n优点：不用 reward model、不用采样，像 SFT 一样简单稳定。缺点：off-policy，依赖离线偏好数据的分布。',
    ref: RLHF,
  }),
  b({
    id: 'bagu-grpo',
    topic: 'Post-train',
    q: 'GRPO 相比 PPO 改了什么？适合什么任务？',
    a: '去掉 value model：每个 prompt 采样一组 G 个回答，advantage = (r − 组内均值) / 组内标准差，用组内统计当基线。省掉 critic 的显存和训练不稳定。\n适合 **reward 可验证** 的任务（数学、代码对错），reward 噪声小，组内基线够用。DeepSeek-R1 用它。',
    ref: RLHF,
  }),
  b({
    id: 'bagu-rl-infra',
    topic: 'Post-train',
    q: 'RL 训练为什么要把 rollout 和 training 分开？最难的环节是什么？',
    a: 'rollout 是推理负载（大 batch、KV cache、低精度），training 是训练负载（梯度、优化器、重计算），一份权重布局不可能两边都最优。所以 rollout 交给 vLLM / SGLang，training 交给 FSDP / Megatron。\n最难的是**每轮的权重同步**：训练完把参数广播给推理引擎，两边切分方式不同要 reshard，大模型下可能占一轮的 10–20%。',
    ref: RLI,
  }),
  b({
    id: 'bagu-sft-packing',
    topic: 'Post-train',
    q: 'SFT 做 packing 要注意什么？loss mask 怎么设？',
    a: 'packing 把多条短样本拼成定长序列，GPU 利用率从三到五成提到九成以上。要配 **varlen attention**（传 `cu_seqlens`）或 block-diagonal mask，否则样本之间互相 attend；位置编码按样本重置。\nloss mask：prompt、system、用户轮的 label 设 −100，只在 assistant 回答上算 loss。',
    ref: SFT,
  }),

  // ---------------- 系统设计 ----------------
  b({
    id: 'bagu-sd-order',
    topic: '系统设计',
    q: '「设计一个 LLM 推理服务」按什么顺序答？',
    a: '1. **澄清**：模型规模、QPS 和输入输出长度分布、TTFT / TPOT 的 P99 目标、是否流式、是否多租户。\n2. **容量估算**：显存账定 TP 和单副本并发，roofline 定单副本吞吐，推出副本数。\n3. **分层架构**：网关 → 路由 → 推理副本 → KV 存储。\n4. **单副本内部**：continuous batching、chunked prefill、PagedAttention、流式输出。\n5. **多副本**：prefix 亲和路由、自动扩缩。\n6. **可观测性和降级**。',
    ref: SD,
  }),
  b({
    id: 'bagu-sd-capacity',
    topic: '系统设计',
    q: '70B、平均 2k 输入 500 输出、峰值 100 QPS，大概要多少张 H100？',
    a: 'TP=4 一个副本，batch 64 时 TPOT 约 20 ms，单副本 decode 约 3200 tok/s。\n生成需求 100 × 500 = 5 万 tok/s → **16 个副本 × 4 卡 = 64 张**。\n再看 prefill：100 × 2k × 2 × 70e9 ≈ 28 PFLOP/s，64 卡 × 989 TFLOP/s × 50% MFU ≈ 31 PFLOP/s，**prefill 几乎吃掉全部算力**。结论：上 prefix caching 或 PD 分离单独给 prefill 配卡。',
    ref: SD,
  }),
  b({
    id: 'bagu-sd-routing',
    topic: '系统设计',
    q: '多副本之间怎么路由？过载时怎么办？',
    a: '**不要轮询**。按 prompt 前缀做 hash，路由到持有这段 KV 的副本，prefix 命中率能从接近 0 提到七成以上，直接砍掉大部分 prefill；同时按负载加权，防止热点前缀打爆一个副本。多轮对话固定路由到同一副本。\n过载：队列超阈值直接返回 **429**，准入控制比排队重要，LLM 请求以秒计，排队只会让 P99 雪崩。',
    ref: SD,
  }),
]
