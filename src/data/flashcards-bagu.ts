// 八股闪卡：推理系统、并行通信、GPU、框架、Post-train、系统设计的高频问答。
// 内容和数字取自站内对应文章（ref），改文章里的结论时记得同步这里。
//
// id 是复习记录的 key（src/data/flashcard-progress.json），改题面可以，改 id 会丢掉这张卡的记录。
// a 支持行内 `code`、**加粗**、$公式$ 和 $$单独一行的公式$$（KaTeX），\n 换行；
// fig 是答案下面的示意图（等宽字符画），code 是代码块；ref 是出处（站内链接，不含 base）。

import { lines, type Card } from '@lib/flashcards'

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
const GRPOV = '/posttrain/grpo-variants'
const ASYNC = '/posttrain/rl-async-rollout'
const SYNC = '/posttrain/rl-weight-sync'
const MISM = '/posttrain/rl-train-infer-mismatch'
const MMENC = '/inference/multimodal-encoder'
const SPEECH = '/inference/speech-output'
const AGENT = '/posttrain/rl-agentic'
const DUPLEX = '/inference/omni-duplex'

const b = (c: Omit<Card, 'deck'>): Card => ({ deck: 'bagu', ...c })

export const baguCards: Card[] = [
  // ---------------- roofline ----------------
  b({
    id: 'bagu-ridge-point',
    topic: 'roofline',
    q: 'ridge point 怎么算？用 H100 SXM 的规格算一下：bf16 峰值 989 TFLOP/s，HBM 带宽 3.35 TB/s。',
    a: '公式：ridge = 峰值算力 ÷ 显存带宽，单位 FLOP/B，意思是每读 1 字节要做多少次运算才能把算力用满。\nH100：989 × 10¹² ÷ 3.35 × 10¹² ≈ **295 FLOP/B**。\n用法：算出算子的算术强度 AI = FLOPs ÷ 访存字节。AI 小于 ridge 是 **memory-bound**，大于是 **compute-bound**；可达性能 = min(峰值, 带宽 × AI)。\n295 不用背，记住公式，换张卡拿规格表现算。',
    ref: ROOF,
  }),
  b({
    id: 'bagu-prefill-ai',
    topic: 'roofline',
    q: 'prefill 的算术强度怎么推？为什么每个参数对每个 token 算 2 次 FLOP？',
    a: '符号：$P$ 参数量，$B$ batch，$S$ 每个序列的 token 数，$b_w$ 每个参数的字节数（bf16 = 2）。\n为什么是 2：就是字面意思。线性层 $y = xW$，输出的每个元素 $y_j = \\sum_i x_i W_{ij}$，每个权重 $W_{ij}$ 对每个 token 恰好做 **1 次乘法 + 1 次加法**，就是 2 FLOP。所有线性层的权重加起来是 P 个，所以每个 token 约 2P FLOP。\n1. FLOPs ≈ $2PBS$。\n2. 访存 ≈ $P b_w$：权重读一遍，激活相对小。\n3. $AI = \\dfrac{2PBS}{P b_w} = \\dfrac{2BS}{b_w}$，bf16 下就是 $BS$。\n「忽略 attention」指的是：$QK^\\top$ 和 $PV$ 是激活乘激活，不碰权重，不在 2P 里。它们每层约 $4dS^2B$ FLOP，S 不长时比线性层小得多（比值见长上下文那张卡）。\n例：B = 1、S = 2048，AI ≈ 2048，远大于 H100 的 295，所以 prefill 是 **compute-bound**。',
    ref: ROOF,
  }),
  b({
    id: 'bagu-decode-ai',
    topic: 'roofline',
    q: 'decode 的算术强度随 batch 怎么变？把读 KV cache 也算上，为什么 batch 再大 AI 也上不去？',
    a: '一句话：**权重摊得薄，KV 摊不薄**。\n每步要读两样东西：\n1. 权重：所有序列**共用**，读一遍服务整个 batch。\n2. KV cache：每个序列**自己的**，batch 里有几个序列就读几份。\n于是：\n1. batch 小，读的主要是权重。batch 翻倍，FLOPs 翻倍、字节几乎不变，AI 跟着涨，bf16 下 AI ≈ batch。\n2. batch 大，读的主要是 KV。batch 翻倍，FLOPs 和 KV 字节一起翻倍，AI 停在一个上限。上下文越长，每个序列的 KV 越大，上限越低：70B、8k 上下文约 60，远低于 H100 的 ridge 295。\n要抬高上限只能让每个 token 的 KV 变小：GQA / MLA，或者把 KV 量化到 fp8。公式和数表见 roofline 页。',
    ref: ROOF,
  }),
  b({
    id: 'bagu-decode-step-time',
    topic: 'roofline',
    q: '算一下：70B 参数、bf16、TP = 4，每张 H100 带宽 3.35 TB/s，忽略 KV 和通信，decode 一步最少要多久？',
    a: 'decode 是 memory-bound，一步的时间 ≈ 每张卡要读的字节 ÷ 带宽。通用公式（P 参数量，$b_w$ 每参数字节，BW 单卡带宽）：\n$$t_{\\text{step}} \\gtrsim \\frac{P \\cdot b_w / \\text{TP}}{\\text{BW}}$$\n代入算：\n1. 权重：70 × 10⁹ 参数 × 2 字节 = 140 GB。\n2. TP = 4，每张卡读 1/4：35 GB。\n3. 35 × 10⁹ ÷ 3.35 × 10¹² ≈ **10 ms**。\n要算 KV 就在分子上加每卡要读的 KV 字节 = batch × 上下文长度 × KV/token ÷ TP。再加上每层两次 all-reduce，实际 TPOT 比这个下界高。\n结论：decode 的时间看读多少字节，不看算多少 FLOPs。',
    ref: SD,
  }),
  b({
    id: 'bagu-long-context-prefill',
    topic: 'roofline',
    q: 'prefill 里 attention 和线性层的 FLOPs 之比怎么算？70B（P = 70 × 10⁹，80 层，d = 8192）在 S = 8k 和 32k 时大约是多少？',
    a: '符号：P 参数量，L 层数，d hidden 维度，S 序列长度，B batch。\n1. 线性层 ≈ 2 · P · B · S。\n2. attention 的 QKᵀ 和 PV 每层各约 2 · d · S² · B，合计 ≈ 4 · L · d · S² · B。\n3. 比值 = 4LdS²B ÷ 2PBS = **2LdS ÷ P**，随 S 线性涨。\n代入：2 × 80 × 8192 × S ÷ (70 × 10⁹) ≈ S ÷ 53,000。8k 时约 0.15，32k 时约 0.6，也就是占总量四成左右。\n所以长上下文 prefill 特别贵，FlashAttention 和 context parallel 对它很关键。',
    ref: ROOF,
  }),

  // ---------------- 显存账 ----------------
  b({
    id: 'bagu-kv-per-token',
    topic: '显存账',
    q: 'KV cache 每 token 占多少字节？算一下 Llama-3-70B：80 层，8 个 KV head，head 维度 128，bf16。',
    a: '公式：KV/token = 2 · L · H_kv · d_h · b\n1. 2：K 和 V 各存一份。\n2. L：层数，每层都要存。\n3. H_kv · d_h：一层里 K（或 V）的宽度。GQA 下用 **KV head 数**，不是 Q head 数。\n4. b：每个元素的字节数，bf16 = 2，fp8 = 1。\n代入：2 × 80 × 8 × 128 × 2 = 327,680 字节 = **320 KiB/token**。\n总量再乘上 batch × 序列长度。',
    ref: MEM,
  }),
  b({
    id: 'bagu-weight-rule',
    topic: '显存账',
    q: '70B 参数的模型，bf16 权重多少 GB？int4 呢？GB 和 GiB 差多少？',
    a: '1 B（billion）= **10⁹**。1 GB = 10⁹ 字节，1 GiB = 2³⁰ ≈ 1.074 × 10⁹ 字节，两者差约 7%。\n1. bf16 每参数 2 字节：70 × 10⁹ × 2 = 140 × 10⁹ 字节 = **140 GB**，约 130 GiB。\n2. int4 每参数 0.5 字节：**35 GB**，再加一点 per-group scale。\n口算规则：bf16 下 GB 数 = 参数量（B）× 2。',
    ref: MEM,
  }),
  b({
    id: 'bagu-layer-params',
    topic: '显存账',
    q: '一层 decoder（GQA attention + SwiGLU MLP + 两个 RMSNorm）有多少参数？按矩阵一个个数，写出公式。',
    a: '符号：d 是 hidden 维度，H / H_kv 是 Q / KV head 数，d_h 是 head 维度（H·d_h = d），d_ff 是 MLP 中间维度。按矩阵数（见图）：\n1. attention：W_q、W_o 各 $d^2$；W_k、W_v 各 $d \\cdot H_{kv} d_h$，GQA 下比 W_q 小。\n2. MLP（SwiGLU）：gate、up、down **三个**矩阵，各 $d \\cdot d_{ff}$。常见算错点是只数两个。\n3. 两个 RMSNorm：各 d。\n$$2d^2 + 2 d H_{kv} d_h + 3 d\\, d_{ff} + 2d$$\n整个模型再加 embedding 和 lm_head 各 $Vd$（V 是词表大小；两者共享权重就只算一次）。',
    fig: lines`
      x [d]
      ├─ RMSNorm                 d
      ├─ W_q     d x H*d_h       = d^2
      ├─ W_k     d x H_kv*d_h
      ├─ W_v     d x H_kv*d_h
      ├─ W_o     H*d_h x d       = d^2
      ├─ RMSNorm                 d
      ├─ W_gate  d x d_ff
      ├─ W_up    d x d_ff
      └─ W_down  d_ff x d
    `,
    ref: MEM,
  }),
  b({
    id: 'bagu-layer-params-2',
    topic: '显存账',
    q: '算一下 Llama-3-70B 一层多少参数：$d = 8192$，$H_{kv} = 8$，$d_h = 128$，$d_{ff} = 28672$，每层公式 $2d^2 + 2dH_{kv}d_h + 3d\\,d_{ff} + 2d$。',
    a: '逐项代入：\n1. $2d^2 = 2 \\times 8192^2 = 134.2$ M。\n2. $2dH_{kv}d_h = 2 \\times 8192 \\times 1024 = 16.8$ M。\n3. $3d\\,d_{ff} = 3 \\times 8192 \\times 28672 = 704.6$ M。\n4. $2d = 16$ K，可以忽略。\n合计约 **0.86 B**，MLP 占 82%。80 层是 68.4 B，再加 embedding 和 lm_head 的 2.1 B，约 70.6 B，和 70B 对得上（验算见显存页）。',
    ref: MEM,
  }),
  b({
    id: 'bagu-activation-small',
    topic: '显存账',
    q: '推理时激活为什么不是显存大头？哪一项要留意？',
    a: '符号：T 是这一步处理的 token 数（≤ `max_num_batched_tokens`），d hidden 维度，d_ff MLP 中间维度，b 每元素字节。\n1. 不做反向，不用为反向保存中间结果：一层算完，它的激活就能给下一层复用。\n2. FlashAttention 不物化 S×S 的 score。\n3. 所以同时活着的只有一两层的张量，约 $T(4d + 2d_{ff})\\,b$（4d 是 Q、K、V 和 attention 输出，2d_ff 是 gate 和 up）。70B（d = 8192，d_ff = 28672）、T = 2048、bf16：约 **352 MiB**，和几十 GB 的 KV 比很小。\nCUDA Graph 下中间结果的地址确实要固定：录图时它们从一个**专用内存池**里分配，回放时原地复用。但池子里的 buffer 仍然逐层复用，只给小 batch（decode）录图，所有图共用一个池，所以也就几百 MB。vLLM 启动时先按最大 token 数跑一次假 forward，量出激活峰值，剩下的才分给 KV。\n要留意的是 **logits**：$B \\times V \\times 4$ 字节（fp32），batch 256 × 128k vocab 就是 128 MiB。',
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
    a: '以 vLLM 为例，一个 block 16 个 token（代码见下）：\n1. prompt 按 16 个一块切开，每个**满块**算一个 hash = hash(前一块的 hash, 本块的 16 个 token)。链式算，所以 hash 相同就说明**从开头到这块**全部相同。\n2. 新请求从第一块开始查表：命中就把那个物理 block 号直接填进自己的 block table、引用计数 +1，这 16 个 token 不用再 prefill；遇到第一个没命中的就停，后面照常算。\n3. 为什么只有满块：最后一块没填满，decode 还要往里写新 token，内容会变，hash 就定不下来；满块以后只读不写，才能放心给别人用。\n4. 引用计数归零的块不马上清掉，留在表里等下次命中，显存不够时按 LRU 回收。\n原来提到的 copy-on-write 是另一回事：并行采样（n > 1）时几个分支共用同一个没满的块，谁要往里写就先复制一份，不影响别人。只共享满块的 prefix cache 用不上它。',
    code: lines`
      BLOCK = 16
      cache = {}   # 块 hash -> 物理 block 号

      def block_hashes(tokens):
          hashes, prev = [], None
          for i in range(0, len(tokens) - BLOCK + 1, BLOCK):   # 只取满块
              prev = hash((prev, tuple(tokens[i:i + BLOCK])))
              hashes.append(prev)
          return hashes

      def lookup(tokens):
          table = []
          for h in block_hashes(tokens):
              if h not in cache:
                  break                    # 第一个没命中就停
              table.append(cache[h])       # 复用物理块，引用计数 +1
          return table, len(table) * BLOCK  # 命中的块, 省掉的 prefill token 数
    `,
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
    a: '先说粒度：量化时一组元素共用一个缩放系数 s（$x_q = \\text{round}(x / s)$），组怎么划就是粒度。\n1. per-tensor：一层的整个 K（或 V）共用一个 s。\n2. per-token：每个 token 的那一行一个 s。\n3. per-channel：每一维（一列）一个 s，所有 token 共用。\n收益：bf16 → fp8，KV 的显存和访存减半，decode 的算术强度翻倍、能放下的 token 翻倍。\n注意点：\n1. **FP8 用 per-tensor 就够**：fp8 自带指数位，动态范围大。vLLM 的 `kv_cache_dtype="fp8"` 每层 K、V 各一个 scale。int4 / int2 这种整数格式才要细粒度：K 的离群值集中在少数几个 channel，所以 K 按 per-channel、V 按 per-token（KIVI）。\n2. **K 比 V 敏感**：score $= q \\cdot k / \\sqrt{d_h}$。k 上的误差 δ 让 score 偏 $q \\cdot \\delta / \\sqrt{d_h}$，过了 softmax 变成乘上 $e^{q \\cdot \\delta / \\sqrt{d_h}}$，会改变注意到谁；V 的误差只是线性混进加权平均，一部分被平均掉。',
    ref: KV,
  }),

  // ---------------- attention 变体 ----------------
  b({
    id: 'bagu-gqa',
    topic: 'attention 变体',
    q: 'MHA / GQA / MQA 的区别？为什么只减 K、V 的 head，不减 Q 的？',
    a: '符号：H 是 Q head 数，H_kv 是 KV head 数，d_h 是 head 维度，L 层数。\n1. MHA：H_kv = H，每个 Q head 有自己的一对 K/V。\n2. GQA：H 个 Q head 分成 H_kv 组，组内共用一对 K/V（见图）。\n3. MQA：H_kv = 1，所有 Q head 共用一对。\n为什么只减 KV：目的就是省 KV。K、V 要缓存，每 token 存 $2 L H_{kv} d_h$ 个数，decode 每一步全读一遍；Q 只算当前 token、用完就丢，不进 cache，减 Q head 省不了显存和带宽，只会掉效果。\nLlama-3-70B（L = 80，d_h = 128，bf16）每 token：MHA（64 个 KV head）2.5 MiB，GQA（8 个）320 KiB，MQA（1 个）40 KiB。',
    fig: lines`
      H = 8; KV head read by each Q head

               Q head   0 1 2 3 4 5 6 7
      MHA      KV head  0 1 2 3 4 5 6 7
      GQA (4)  KV head  0 0 1 1 2 2 3 3
      MQA      KV head  0 0 0 0 0 0 0 0
    `,
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
    q: 'MLA（DeepSeek-V2/V3 的 attention）每个 token 每层往 KV cache 里存什么？各几维？',
    a: '符号（DeepSeek-V3）：$h_t$ 是第 t 个 token 进 attention 前的 hidden，$d = 7168$ 维；每个 head 的 K、V 是 $d_h = 128$ 维。MLA 把 K、V 拆成「先压缩、用时再展开」（见图）：\n1. 下投影 $c_t = W^{DKV} h_t$，7168 → **512** 维（latent，维度记作 $d_c$）。**缓存**。\n2. 上投影：第 i 个 head 的 $k^C_{t,i} = W^{UK}_i c_t$、$v_{t,i} = W^{UV}_i c_t$，各 128 维，用的时候现算，**不缓存**。\n3. 位置部分 $k^R_t = \\text{RoPE}(W^{KR} h_t)$，**64** 维（记作 $d_r$），所有 head 共用一份，**缓存**。\n所以每 token 每层存 512 + 64 = **576** 个数。RoPE 为什么要单独一份，见 decoupled RoPE 那张卡。',
    fig: lines`
      one token, one layer; [ ] = cached

      h (7168)
      ├ W_DKV -> [c 512] ┬ W_UK_i -> k_i 128
      │                  └ W_UV_i -> v_i 128
      └ W_KR -> RoPE -> [kR 64]  all heads

      cached: 512 + 64 = 576 numbers
    `,
    ref: ATTN,
  }),
  b({
    id: 'bagu-mla-3',
    topic: 'attention 变体',
    q: '算 DeepSeek-V3 每 token 的 KV 字节：61 层，每层缓存 576 个数（512 维 latent + 64 维 RoPE key），bf16。换成 8 个 KV head、head 维度 128 的 GQA 是多少？',
    a: '1. MLA：$61 \\times 576 \\times 2 = 70{,}272$ B ≈ **68.6 KiB**。\n2. GQA：K、V 各一份，$2 \\times 61 \\times 8 \\times 128 \\times 2 = 249{,}856$ B = **244 KiB**，是 MLA 的 3.6 倍。\nlatent 是所有 head 共用的一份，大小像 MQA；但每个 head 用自己的上投影矩阵从它还原出不同的 K、V，所以不像 MQA 那样掉效果。',
    ref: ATTN,
  }),
  b({
    id: 'bagu-mla-2',
    topic: 'attention 变体',
    q: 'MLA decode 时不把 K 展开成每个 head 的 128 维，直接拿缓存的 512 维 latent $c_t$ 算 score，靠的是哪一步变换？',
    a: '符号：$c_t$ 是缓存的 latent；第 i 个 head 的 $k^C_{t,i} = W^{UK}_i c_t$，$W^{UK}_i$ 是 128 × 512 的常数矩阵；$q^C_i$ 是当前 token 第 i 个 head 的 query，128 维。\n按矩阵乘结合律换个括号（weight absorption）：\n$$q^{C\\top}_i k^C_{t,i} = q^{C\\top}_i W^{UK}_i c_t = \\big(W^{UK\\top}_i q^C_i\\big)^\\top c_t$$\n1. 括号里的 $\\tilde q_i = W^{UK\\top}_i q^C_i$ 是 512 维，每个 head 每步只算一次。\n2. 之后直接和每个历史 token 的 $c_t$ 点积，K 从头到尾不展开。\n3. 输出侧同理：$\\sum_t p_t\\, v_{t,i} = W^{UV}_i \\big(\\sum_t p_t\\, c_t\\big)$，先在 512 维上加权求和，最后乘一次 $W^{UV}_i$。\n结果：128 个 head 读同一份 latent，形状上就是只有一个 KV head 的 MQA（代码见下；RoPE 那 64 维单独加，见 decoupled RoPE 那张卡）。',
    code: lines`
      # decode 一步、一层。缓存 c: [S, 512], kR: [S, 64]
      # 当前 token: qC: [H, 128], qR: [H, 64]; H = 128
      # W_UK, W_UV: [H, 128, 512]
      q_lat = einsum('hd,hdc->hc', qC, W_UK)         # [H, 512]
      score = (q_lat @ c.T + qR @ kR.T) / sqrt(192)  # [H, S]
      p = softmax(score, dim=-1)
      o_lat = p @ c                                  # [H, 512]
      o = einsum('hc,hdc->hd', o_lat, W_UV)          # [H, 128]
    `,
    ref: ATTN,
  }),
  b({
    id: 'bagu-mla-4',
    topic: 'attention 变体',
    q: 'MLA decode，一层、一个历史 token：吸收后 128 个 head 读同一份 576 个数的 KV（bf16），每个 head 做一次 576 维点积算 score、一次 512 维乘加算输出。读多少字节、做多少 FLOP、算术强度多少？',
    a: '一次 n 维点积或乘加是 n 次乘法加 n 次加法，2n FLOP。\n1. 字节：$576 \\times 2 = 1152$ B，128 个 head 共用这一份。\n2. FLOPs：每个 head $2 \\times (576 + 512) = 2176$，128 个 head 共 278,528。\n3. AI = 278528 ÷ 1152 ≈ **242 FLOP/B**，接近 H100 的 ridge 295。\n对比展开成每个 head 的 K（128 + 64 维）和 V（128 维）：每个 head 读自己的 320 个数（640 B），做 $2 \\times 320$ FLOP，AI = **1**。\n代价是每个历史 token 的计算从 320 维变成 1088 维，多约 3.4 倍。decode 是 memory-bound，拿闲着的算力换带宽划算。',
    ref: ATTN,
  }),
  b({
    id: 'bagu-mla-rope',
    topic: 'attention 变体',
    q: 'MLA 为什么要 decoupled RoPE？直接给上投影出来的 K 加 RoPE，weight absorption 会坏在哪？',
    a: 'RoPE：位置 t 的向量乘一个旋转矩阵 $R_t$，并且 $R_m^\\top R_n = R_{n-m}$。\n1. 吸收能成立，靠的是 $q^\\top W^{UK} c_n$ 中间的 $W^{UK}$ 是**常数**，可以提前并进 q（见 weight absorption 那张卡）。\n2. 给 K 加 RoPE：$k_n = R_n W^{UK} c_n$，位置 m 的 query 是 $R_m q$，\n$$(R_m q)^\\top R_n W^{UK} c_n = q^\\top R_{n-m}\\, W^{UK} c_n$$\n中间的 $R_{n-m} W^{UK}$ 对**每个历史位置 n** 都不一样，并不进 q，只能对每个 n 把 K 展开（DeepSeek-V2，arXiv 2405.04434 §2.1.3）。\n3. 解法：K 拆两段，score 是两段点积之和。内容段 $W^{UK}_i c_n$ 不加 RoPE，照常吸收；位置段 $k^R_n = \\text{RoPE}(W^{KR} h_n)$ 只有 64 维、所有 head 共用，直接缓存。',
    ref: ATTN,
  }),
  b({
    id: 'bagu-gqa-tp',
    topic: 'attention 变体',
    q: 'GQA 模型做 TP 时对切分数有什么限制？',
    a: 'TP 要整除 H（Q head 数）。KV head 分两种情况：\n1. TP ≤ H_kv：TP 要整除 H_kv，每卡分到 H_kv / TP 个 KV head，KV cache 正好按 1/TP 切。\n2. TP > H_kv：KV head 不够分，只能**复制**，每个 KV head 在 TP / H_kv 张卡上各存一份，每卡的 KV 不再是 1/TP，白占显存。\n例：Llama-3-70B，H = 64，H_kv = 8。TP = 8 时每卡 1 个 KV head；TP = 16 时每两张卡存同一个 KV head，KV 总量翻倍。\n所以 TP 一般 **≤ H_kv**。',
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
    q: 'A100 上 bf16 矩阵乘峰值 312 TFLOP/s，fp32 的普通运算只有 19.5 TFLOP/s。FlashAttention-2 为什么让输出 O 在循环里一直不除分母 $\\ell$，最后才除一次？',
    a: '符号：Q 按 $B_r$ 行一块，K/V 按 $B_c$ 行一块；$m$ 是每行的最大值，$\\ell$ 是 softmax 的分母（推导见 online softmax 页）。\n1. 312 ÷ 19.5 = 16：一次非 matmul 的 FLOP 和 16 次 matmul FLOP 一样费时，所以要尽量少做除法、exp 这类标量运算。\n2. v1 每处理一块 K/V，都把 O 除以 $\\ell$ 归一化一次。\n3. v2 让 O 一直保持**没除 $\\ell$** 的状态，每块只乘修正系数 $e^{m_{\\text{old}} - m_{\\text{new}}}$，循环结束才除一次（代码见下）。结果一样，每块少一次 $B_r \\times d$ 的除法。',
    code: lines`
      # v2 前向：一个 thread block 算一块 Qi [Br, d]
      m = full(Br, -inf); l = zeros(Br); O = zeros(Br, d)
      for j in range(Tc):                    # K/V 块在内层流过
          S = Qi @ K[j].T                    # [Br, Bc]
          m_new = maximum(m, S.max(-1))
          P = exp(S - m_new[:, None])
          scale = exp(m - m_new)
          l = scale * l + P.sum(-1)
          O = scale[:, None] * O + P @ V[j]  # 不除 l
          m = m_new
      O = O / l[:, None]                     # 循环结束只除一次
      write(O); write(m + log(l))            # 反向只需要 logsumexp
    `,
    ref: FA,
  }),
  b({
    id: 'bagu-fa-v2-v3-3',
    topic: 'FlashAttention',
    q: 'FlashAttention v1 一个 thread block 管一个 (batch, head)。batch B = 1、head 数 H = 16 时能开几个 thread block？A100 有 108 个 SM。v2 改成每个 Q 块一个 thread block，序列长 S = 16k、Q 块 $B_r$ = 128 行时是多少？',
    a: '1. v1：grid = B · H = **16** 个 block，108 个 SM 只用上 16 个。长序列时显存只放得下小 batch，这种情况很常见。\n2. v2：Q 块之间互不依赖，每块由一个 block 独立算完，grid = B · H · S / $B_r$ = 16 × 16384 / 128 = **2048** 个 block，能填满所有 SM。\nv1 没法这么切：它外层遍历 K/V 块，不同 K/V 块要累加进同一行输出，有写冲突。',
    ref: FA,
  }),
  b({
    id: 'bagu-fa-v2-v3-4',
    topic: 'FlashAttention',
    q: 'FlashAttention 一个 thread block 里有 4 个 warp。v1 把 K/V 分给 4 个 warp，v2 改成把 Q 分给 4 个 warp。为什么 v2 更快？',
    a: '1. v1（split-K）：每个 warp 拿一段 K/V，只算出每行输出的**一部分**，要写进 shared memory、同步、再加起来。\n2. v2：每个 warp 拿一段 Q 行、看全部 K/V，各自算出自己那几行的**完整**输出，warp 之间不用通信，省掉 shared memory 读写和同步。\n这一处加上少做非 matmul 运算、沿序列并行，合起来 v2 比 v1 快约 2 倍，A100 上到峰值的 50–73%（作者自测，arXiv 2307.08691）。',
    ref: FA,
  }),
  b({
    id: 'bagu-fa-v2-v3-2',
    topic: 'FlashAttention',
    q: 'FlashAttention-2 在 H100 上只有约 35% 利用率。FA3 用了 Hopper 的哪些新硬件、做了哪几件事？',
    a: '先说 Hopper 新增的两样：\n1. **TMA**：专门搬数据的硬件单元，一条指令把一整块 tile 从 HBM 异步拷进 shared memory，不用线程逐个算地址。\n2. **wgmma**：一个 warpgroup（4 个 warp，128 个线程）一起发的**异步**矩阵乘，操作数直接从 shared memory 读；发出去以后线程可以先干别的。\n瓶颈在哪：H100 SXM 的 bf16 矩阵乘是 989 TFLOP/s，exp 这类特殊函数只有 3.9 TFLOP/s。head 维度 128 时，每个 score 元素有 $2 \\times 128$（$QK^\\top$）+ $2 \\times 128$（$PV$）= 512 次 matmul FLOP、1 次 exp：\n$$\\frac{1 / 3.9}{512 / 989} \\approx 0.5$$\nexp 的耗时是 matmul 的一半，串着做就多出 50% 的时间。FA3（arXiv 2407.08608）做了三件事：\n1. **warp specialization**：一部分 warp 只发 TMA 加载（producer），其余 warpgroup 只做计算（consumer），中间用 shared memory 里的环形缓冲，搬数据和计算重叠。\n2. **把 softmax 藏进 GEMM**：两个 warpgroup 乒乓，一个在算 exp 时，另一个的 wgmma 在跑；同一个 warpgroup 内也让下一块的 $QK^\\top$ 和这一块的 softmax 重叠。\n3. **FP8**：按块量化，并给 Q、K 乘一个随机正交矩阵（Hadamard）把离群值摊开，降低量化误差。\n效果（作者自测）：bf16 到 740 TFLOP/s（75% 利用率），FP8 接近 1.2 PFLOP/s。',
    ref: FA,
  }),
  b({
    id: 'bagu-flash-decoding',
    topic: 'FlashAttention',
    q: 'decode 阶段 FlashAttention 帮助大吗？FlashDecoding 改了什么？',
    a: '可以这么记：**FlashDecoding = FlashAttention + 沿 KV 长度切分（split-KV，和 GEMM 的 split-K 一个思路）+ 最后合并一次**。\n1. 问题：decode 时 Q 只有 1 行，FlashAttention 按 batch × head 分 block。B = 1、32 个 head 就只有 32 个 block，H100 有 132 个 SM，大部分闲着，读 KV 的带宽用不满。\n2. 切分：长度 S 的 KV 切成 s 段，block 数变成 B × H × s。每段各算局部输出 $o_j$、局部最大值 $m_j$、局部分母 $\\ell_j$（见图）。\n3. 合并（一个小 kernel），和 online softmax 的合并一样：\n$$m = \\max_j m_j,\\qquad o = \\frac{\\sum_j e^{m_j - m}\\,\\ell_j\\, o_j}{\\sum_j e^{m_j - m}\\,\\ell_j}$$',
    fig: lines`
      KV  [seg 0|seg 1|seg 2|seg 3]
             |     |     |     |
            SM    SM    SM    SM   o_j m_j l_j
              \    \    /    /
             merge: rescale+sum -> o
    `,
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
    a: '**踢最新进来的**（LIFO）：它算得最少，丢掉重算浪费最小；老请求也不会被反复踢、饿死。\n对的，出处是 vLLM V1 源码 https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py ，`schedule()` 里 `allocate_slots` 给 running 请求分配 block 失败时：\n1. FCFS 策略：`self.running.pop()`，弹出 running 列表末尾，也就是最后被调度进来的请求。\n2. priority 策略：踢优先级最低的，同优先级踢到达最晚的。\n被踢的请求 KV 全部释放、`num_computed_tokens` 清零，`waiting.prepend_request` 放回 waiting 队首，之后重新 prefill（V1 只有 recompute，没有 swap）。\n频繁抢占说明 `max_num_seqs` 设得太激进或 KV 空间不够，该调参或加卡，而不是让调度器反复抖动。',
    ref: SCHED,
  }),

  // ---------------- 投机解码 ----------------
  b({
    id: 'bagu-spec-why',
    topic: '投机解码',
    q: 'speculative decoding 为什么能「白赚」？',
    a: 'decode 是 memory-bound：每一步把权重整读一遍，只算 B 个 token，Tensor Core 大半在闲着。\n投机解码让小模型先猜 k 个 token，target 一次前向把 k 个一起验证：\n1. 读权重：还是一遍。\n2. FLOPs：涨约 k 倍，但算力本来就闲着，几乎不加时间。\n3. 一次前向平均能产出好几个 token（见接受率那张卡）。\n一句话：**多花闲置的算力（猜错的 token 白算），换少读几遍权重**；读权重的次数少了，延迟就降了。',
    ref: SPEC,
  }),
  b({
    id: 'bagu-spec-exact',
    topic: '投机解码',
    q: '投机采样里 draft 猜的 token 按什么规则接受？证明这样采出来的分布和 target 自己采样完全一样。',
    a: '符号：在某个位置，draft 模型给出分布 $q$，target 给出 $p$（都是词表上的概率）；draft 已经按 q 采了一个 token x。\n规则（Leviathan et al. arXiv 2211.17192；Chen et al. arXiv 2302.01318）：\n1. 以概率 $\\min\\big(1, p(x) / q(x)\\big)$ 接受 x。也就是 p(x) ≥ q(x) 一定接受，p(x) < q(x) 按比例接受。\n2. 拒绝时从残差分布重采一个 token，这个位置之后的 draft token 全部作废：\n$$\\tilde x \\sim \\frac{\\max(0,\\ p - q)}{\\sum_y \\max(0,\\ p(y) - q(y))}$$\n证明：最后输出 x 只有两条路。\n1. draft 采到 x 且被接受：$q(x) \\cdot \\min\\big(1, \\frac{p(x)}{q(x)}\\big) = \\min(p(x), q(x))$。\n2. 先被拒绝，再重采到 x。拒绝的总概率是 $1 - \\sum_y \\min(p(y), q(y))$。因为每个 y 都有 $p(y) = \\min(p, q) + \\max(0, p - q)$，对 y 求和得 $1 - \\sum_y \\min(p, q) = \\sum_y \\max(0, p - q)$，正好等于残差分布的分母，所以这条路的概率是 $\\max(0, p(x) - q(x))$。\n两条加起来：\n$$P(\\text{out} = x) = \\min(p, q) + \\max(0,\\ p - q) = p(x)$$\n看图（每个 # 是 0.1）：p 的每一格，要么被「接受」那一行盖住（min 部分），要么由「重采」那一行补上（p 超出 q 的部分），合起来正好是 p。A 被 draft 高估了，多出的 0.3 概率被拒绝，挪给了被低估的 B 和 C。',
    fig: lines`
      each # = 0.1

                 A      B      C
      q (draft)  ###### ###    #
      p (target) ###    #####  ##
      accept     ###    ###    #     min(p,q)
      resample          ##     #     p-q > 0
      output     ###    #####  ##    = p

      reject prob = 0.6 - 0.3 = 0.3
    `,
    ref: SPEC,
  }),
  b({
    id: 'bagu-spec-expected',
    topic: '投机解码',
    q: '接受率 α、一轮猜 k 个 token，一轮平均产出几个 token？bonus token 算在哪？k 越大越好吗？',
    a: '假设每个 draft token 独立地以概率 α 被接受（Leviathan et al. 的简化假设，arXiv 2211.17192）。一轮的产出 = **接受的 draft token 数 N** + **target 自己出的 1 个 token**。\n1. N 的分布：前 i 个都接受才有 N ≥ i，所以 $P(N \\ge i) = \\alpha^i$（i = 1..k）。\n2. 期望用尾概率求和：$E[N] = \\sum_{i=1}^{k} P(N \\ge i) = \\sum_{i=1}^{k} \\alpha^i$。\n3. bonus：不管停在哪，target 这次前向都会多给 1 个 token。在第 j 个位置被拒，就在那个位置从残差分布重采一个；k 个全接受，target 在第 k + 1 个位置的分布已经顺手算好了，再采一个。所以每轮**恰好 +1**。\n4. 合起来：\n$$E[\\text{tokens}] = 1 + \\sum_{i=1}^{k} \\alpha^i = \\sum_{i=0}^{k} \\alpha^i = \\frac{1 - \\alpha^{k+1}}{1 - \\alpha}$$\n求和从 i = 0 开始，多出来的 $\\alpha^0 = 1$ 就是 bonus token。\n算一下：α = 0.8、k = 4，接受 0.8 + 0.64 + 0.512 + 0.41 = 2.36，加 bonus 共 **3.36**。\nk 越大越好吗：一轮的成本是 draft 跑 k 步加 target 跑 1 步。记 c = draft 一步的耗时 ÷ target 一步，并假设 target 一次验证 k + 1 个 token 和算 1 个一样快（decode 是 memory-bound）：\n$$\\text{speedup} = \\frac{1 - \\alpha^{k+1}}{(1 - \\alpha)(ck + 1)}$$\n分子随 k 饱和（上限 $\\frac{1}{1-\\alpha}$），分母随 k 线性涨，所以有最优 k（见表）：α 高、draft 便宜时 k 可以大一些；α = 0.6、c = 0.2 时 k = 2 最好，也只有 1.4 倍，k = 8 反而变慢。',
    fig: lines`
      speedup by k

           α=0.8   α=0.8   α=0.6
      k    c=0.1   c=0.2   c=0.2
      1    1.64    1.50    1.33
      2    2.03    1.74    1.40
      3    2.27    1.85    1.36
      4    2.40    1.87    1.28
      6    2.47    1.80    1.10
      8    2.40    1.66    0.95
    `,
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
    a: '两者都是 weight-only 4 bit，都配 per-group（每 128 个权重一个 scale）。区别在怎么降误差（见图）：\n**GPTQ**：一列一列量化。每量化一列，就把这列的量化误差按二阶信息（$H = 2XX^\\top$，X 是校准数据的激活）**补偿到还没量化的列上**，让这一层的输出 $WX$ 整体误差最小。\n**AWQ**：看激活。激活大的输入通道对输出影响大（约 1%）。量化前把这些通道的权重乘 s、对应的激活除 s，输出不变：\n$$Wx = \\big(W \\operatorname{diag}(s)\\big)\\big(\\operatorname{diag}(s)^{-1} x\\big)$$\n乘 s 后量化的舍入误差大小不变，除回去以后就缩小到 1/s。不用反向，只要搜 s，校准快。',
    fig: lines`
      GPTQ: quantize column by column
        W columns   c0   c1   c2   c3
        step 1      Q -> error spread to c1..c3
        step 2           Q -> spread to c2..c3
        step 3                Q -> spread to c3
        step 4                     Q

      AWQ: protect salient channels
        |x| per channel   1   1   9   1
                                  ^ salient
        weights of c2 *= s,  x2 /= s
    `,
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
    a: '定义：量化把一组浮点数映射成整数，$w_q = \\text{round}(w / s)$，int4 对称量化时 $s = \\max|w| / 7$。**粒度 = 多少个元素共用一个 s**（见图，W 是 out × in）：\n1. per-tensor：整个矩阵一个 s。\n2. per-channel：每个输出通道（一行）一个 s。\n3. per-group：每行再按 g 个元素一组（常用 g = 128）。\n取舍：\n1. 误差：s 由组里绝对值最大的元素决定，一个离群值会把整组的 s 撑大，其余元素的分辨率就变差。组越小，离群值祸害的范围越小。\n2. 开销：scale 要存，kernel 里要按组反量化。int4 + g = 128 + fp16 scale，每个权重多 16 / 128 = 0.125 bit。\n所以 int8 权重常用 per-channel，int4 权重用 per-group；激活只能运行时现算 scale，一般 per-token 或 per-tensor。KV cache 见 KV 量化那张卡。',
    fig: lines`
      W: 4 x 8 (out x in)
      same letter = shares one scale

      per-tensor     per-channel    per-group g=4
      aaaaaaaa       aaaaaaaa       aaaabbbb
      aaaaaaaa       bbbbbbbb       ccccdddd
      aaaaaaaa       cccccccc       eeeeffff
      aaaaaaaa       dddddddd       gggghhhh
      1 scale        4 scales       8 scales
    `,
    ref: QUANT,
  }),

  // ---------------- 指标 ----------------
  b({
    id: 'bagu-metrics',
    topic: '指标',
    q: '一个请求 0 ms 发出，在 300、320、340、500、520 ms 收到第 1 到第 5 个 token。TTFT、TPOT、ITL、E2E 各是多少？各自主要受什么影响？',
    a: '符号：$t_0$ 是发出请求的时刻，$t_1 \\ldots t_n$ 是收到第 1 到第 n 个 token 的时刻，n = 5。\n1. **TTFT** $= t_1 - t_0 = 300$ ms。包含排队、tokenize、prefill、第一次采样和网络，高负载下大头通常是排队，其次是 prefill。\n2. **ITL**：相邻两个 token 的间隔 $t_k - t_{k-1}$，这里是 20、20、160、20 ms，共 n − 1 = 4 个样本。中间那个 160 ms 一般是这一步 batch 里混进了别人的长 prefill，所以看 ITL 要看 P99。\n3. **TPOT**：除首 token 外的平均间隔 $\\frac{t_n - t_1}{n - 1} = \\frac{220}{4} = 55$ ms，每个请求只算 1 个数。它受 decode 每步的时间影响，也就是 batch 多大、每步读多少字节。\n4. **E2E** $= t_n - t_0 = 520$ ms $=$ TTFT $+ (n - 1) \\times$ TPOT。常见写法「TPOT × 输出长度」多算了一个 token，n 大时可以忽略。\nTPOT 和 ITL 的区别在加权：TPOT 每个请求一票，ITL 每个间隔一票，长输出的请求在 ITL 里权重更大。vLLM 压测脚本 `vllm/benchmarks/serve.py` 就按这个定义算。',
    ref: METRIC,
  }),
  b({
    id: 'bagu-goodput',
    topic: '指标',
    q: '什么是 goodput？什么时候看它，什么时候看吞吐？',
    a: 'goodput = **满足 SLA 的请求**的吞吐，比如每秒完成多少个 TTFT < 1 s 且 TPOT < 50 ms 的请求。\n看哪个要分场景：\n1. 离线批处理（评测、数据合成）：没有延迟要求，看吞吐。\n2. 在线服务：看 goodput。加大 batch 往往吞吐涨了，但每步变慢、TPOT 超标的请求变多、抢占增加，goodput 反而降；只看吞吐会把「牺牲一部分请求换总量」的改动当成正收益。',
    ref: METRIC,
  }),
  b({
    id: 'bagu-benchmark-design',
    topic: '指标',
    q: '怎么设计一次推理服务的压测？',
    a: '1. 输入输出长度用真实 trace 或 ShareGPT 的分布，别用固定长度。\n2. 请求按**泊松过程**到达，逐档扫 QPS，每档跑够久、先 warmup。泊松到达就是相邻请求的间隔服从指数分布、均值 1/QPS（见图和代码）：平均速率一样，但会扎堆，比匀速发更像真实流量，也更容易暴露排队。\n3. 画 QPS 对 P50 / P99 的 TTFT、TPOT 曲线，报 goodput。\n4. 确认**压测客户端自己**不是瓶颈：客户端也要发请求、解析流式返回、算指标，它的 CPU 打满时，测到的延迟里混着客户端自己的排队。看客户端 CPU，或者和服务端自己记的指标对一下。',
    fig: lines`
      1 second, 10 requests each

      uniform  |  |  |  |  |  |  |  |  |  |
      poisson  || |   |||    | |  |      |
    `,
    code: lines`
      import numpy as np
      qps, n = 10, 1000
      gaps = np.random.exponential(1 / qps, n)   # 间隔 ~ Exp(λ = qps)，均值 0.1 s
      send_at = np.cumsum(gaps)                  # 每个请求的发送时刻
    `,
    ref: METRIC,
  }),
  b({
    id: 'bagu-ttft-parts',
    topic: '指标',
    q: 'TTFT 由哪几段组成？高负载下大头通常是哪段？',
    a: '按时间顺序：\n1. **排队**：在 waiting 队列里等调度器分到 KV block 和 token 预算。\n2. **tokenize**：文本 → token id，在 CPU 上做，长 prompt 要几毫秒。\n3. **prefill**：GPU 前向算完整个 prompt，拿到最后一个位置的 logits。\n4. **采样**：logits 按 temperature / top-p 选出第一个 token id，GPU 上一般不到 1 ms。\n5. **detokenize**：token id → 文本片段。一个汉字常常跨几个 token 的字节，要凑齐才能输出。\n6. **网络**：这段文本通过 SSE 推回客户端，同机房一两毫秒，跨地域几十毫秒。\n高负载下大头往往是**排队**，不是计算。所以过载时要做准入控制（直接返回 429），别让队列越排越长、P99 雪崩。',
    ref: LIFE,
  }),

  // ---------------- 并行 ----------------
  b({
    id: 'bagu-megatron-tp',
    topic: '并行',
    q: 'MLP 是 $Y = \\text{GeLU}(XA)\\,B$，用 TP 切到 N 张卡上。A 按列切、B 按行切，为什么整个 MLP 只要一次 all-reduce？反过来 A 按行切会怎样？',
    a: '符号：X 是 [T, d] 的输入（每行一个 token），A 是 [d, f]，B 是 [f, d]，N 张卡。「按列切」是把 A 竖着切成 N 条，「按行切」是把 B 横着切成 N 条（说的是数学上的矩阵，和存储布局无关）。\n1. **A 按列切**：$A = [A_1, \\dots, A_N]$，每卡 $A_k$ 是 [d, f/N]。每卡拿完整的 X 算 $XA_k$，得到中间结果的第 k 段列，**不用通信**。\n2. **GeLU 逐元素**：第 k 段列的 GeLU 只依赖第 k 段，每卡各算各的 $\\text{GeLU}(XA_k)$。\n3. **B 按行切**：$B_k$ 是 [f/N, d]，正好接住第 k 段：\n$$Y = \\sum_{k=1}^{N} \\text{GeLU}(XA_k)\\, B_k$$\n每卡算出一个 [T, d] 的部分和，最后**一次 all-reduce** 求和（见图）。\n反过来 A 按行切：X 也得按列切，每卡得到的是部分和 $X_k A_k$，但\n$$\\text{GeLU}\\Big(\\sum_k X_k A_k\\Big) \\ne \\sum_k \\text{GeLU}(X_k A_k)$$\n所以 GeLU 之前就得 all-reduce，后面还要再通信一次。\nattention 同理：$W_q, W_k, W_v$ 按列切就是按 head 切，每卡算自己的 H/N 个 head，head 之间不用通信；$W_o$ 按行切，一次 all-reduce。所以每层前向共 **2 次**（Megatron-LM，arXiv 1909.08053 第 3 节）。\n例：Llama-3-70B（d = 8192，f = 28672，SwiGLU 的 gate、up 用同样的列切法，SiLU(gate) × up 仍是逐元素），TP = 4 时每卡的 gate、up 是 [8192, 7168]，down 是 [7168, 8192]。',
    fig: lines`
      TP = 2, Y = GeLU(X A) B

      card 0          card 1
      X  [T,d]        X  [T,d]     copy
      A1 [d,f/2]      A2 [d,f/2]   col
      GeLU            GeLU         local
      B1 [f/2,d]      B2 [f/2,d]   row
      Y1 [T,d]        Y2 [T,d]     partial
       └── all-reduce: Y = Y1+Y2 ──┘
    `,
    ref: TP,
  }),
  b({
    id: 'bagu-tp-comm-size',
    topic: '并行',
    q: '70B（d = 8192，80 层）TP = 8、bf16，prefill 一步 8 条 × 2k token。每次 all-reduce 多大、每卡发多少、NVLink 上要多久？参数量变大，这个数怎么变？',
    a: '符号：T 是这一步的 token 数（batch × 序列长度），d hidden 维度，b 每元素字节，N = TP 度。\n1. 消息大小：TP 的 all-reduce 求和的是一层的输出激活 [T, d]，$D = T d b$ = 16384 × 8192 × 2 = **256 MiB**。\n2. 每卡发送（ring）：$\\frac{2(N-1)}{N} D$ = 1.75 × 256 MiB ≈ 470 MB。\n3. 时间：NVLink 单向 450 GB/s，约 **1.04 ms**。每层前向 2 次，80 层 160 次，约 **167 ms**。\n对比计算：$2PT$ = 2 × 70 × 10⁹ × 16384 ≈ 2.3 PFLOP，8 卡、50% MFU 约 0.58 s。通信不和计算重叠的话，占三成左右。\n和模型大小的关系：$D$ 里只有 d，没有参数量。参数量 ∝ $L d^2$：靠加层数 L 变大时，单次消息不变，只是次数变多；d 翻倍时参数翻 4 倍，消息只翻 2 倍。真正让单次变大的是 **T**。\ndecode 时 T = batch：batch 64 只有 64 × 8192 × 2 = 1 MiB，带宽项约 4 µs，比每次通信的固定延迟还小，这时拼的是延迟（见 ring all-reduce 那张卡）。',
    ref: PAR,
  }),
  b({
    id: 'bagu-ring-allreduce',
    topic: '并行',
    q: '8 张卡 ring all-reduce 一份 256 MiB 的数组，每张卡发多少字节？换成 64 张卡呢？为什么说 ring 带宽最优？',
    a: '符号：N 张卡，每张卡上的数组 D 字节；all-reduce 之后每张卡都拿到 N 份逐元素求和的结果。\n做法：N 张卡连成环，每张卡只发给右边、只从左边收；数组切成 N 块，每块 D/N（4 卡的过程见图）。\n1. **reduce-scatter**，N − 1 步：每步每张卡把一块发给右边，右边加到自己同编号的块上。一块沿环走 N − 1 步，正好加齐 N 份。结束时每张卡各持有一块完整的和。\n2. **all-gather**，N − 1 步：加好的块沿环再转一圈，收到的直接覆盖。\n3. 每张卡共发 $2(N-1)$ 次、每次 D/N：\n$$\\text{每卡发送} = \\frac{2(N-1)}{N} D$$\n代入：N = 8 时 1.75 × 256 = **448 MiB**；N = 64 时 1.97 × 256 = **504 MiB**。卡多了 8 倍，每卡流量只多 12%。NVLink 单向 450 GB/s 下，448 MiB（约 470 MB）约 **1.04 ms**。\n为什么带宽最优：每张卡的结果都需要其余 N − 1 张卡的贡献。求和阶段每卡至少要收 $\\frac{N-1}{N}D$，把结果发到所有卡的阶段每卡至少再收 $\\frac{N-1}{N}D$，合计就是上式，ring 正好达到这个下界（Patarasuk & Yuan 2009 https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf ）。\n代价是步数 $2(N-1)$：每步有固定延迟 α，总时间 $2(N-1)\\alpha + \\frac{2(N-1)}{N} \\cdot \\frac{D}{BW}$。消息小（比如 decode 时 1 MiB）或卡很多时延迟项占大头，NCCL 会改用 tree。',
    fig: lines`
      N = 4, array in chunks 0..3 (D/4 each)
      every step: each card sends 1 chunk
      to its right neighbour

      reduce-scatter (receiver adds)
        step 0: a chunk holds 2 of 4 parts
        step 1: 3 of 4
        step 2: 4 of 4, each card owns 1 sum
      all-gather (receiver copies)
        3 more steps
      per card: 6 sends x D/4 = 1.5 D
    `,
    ref: COMM,
  }),
  b({
    id: 'bagu-tp-intra-node',
    topic: '并行',
    q: '同一份 256 MiB 的 TP all-reduce，在节点内 NVLink 和跨节点 IB 上各要多久？PP 跨节点每步传多少？据此说明为什么 TP 不跨节点、PP 可以。',
    a: '带宽（每个方向）：H100 节点内 NVLink 约 450 GB/s（标称 900 GB/s 是双向合计），跨节点 IB NDR 每张网卡 400 Gb/s ≈ 50 GB/s，差 9 倍。\n沿用 70B、TP = 8、一步 16384 个 token 的例子（d = 8192，bf16，见 TP 通信量那张卡）：\n1. **TP**：一次 all-reduce 每卡发 1.75 × 256 MiB ≈ 470 MB。NVLink 约 1.04 ms，IB 约 **9.4 ms**。每层 2 次、80 层共 160 次：NVLink 0.17 s，IB **1.5 s**，比这一步的计算（约 0.58 s）还长。而且它在关键路径上：下一层要等这次求和完才能开始。\n2. **PP**：只在 stage 边界把激活 [T, d] 用 P2P 发给下一个 stage，一次 256 MiB，IB 上约 5 ms。p = 2 时整个前向只传 1 次，TP 是 160 次，差两个数量级；还能和别的 micro-batch 的计算重叠。\n结论：TP 组放在一个节点里（≤ 8 卡），跨节点用 PP 或 DP。Llama 3 405B 的 bf16 推理就是这么摆的：节点内 TP，两台机器之间 PP（arXiv 2407.21783 第 6.1 节）。',
    ref: COMM,
  }),
  b({
    id: 'bagu-pp-bubble',
    topic: '并行',
    q: 'PP 切成 p 个 stage，一个 batch 拆成 m 个 micro-batch。bubble 占多少时间？p = 4 时 m = 8 和 m = 32 各是多少？',
    a: '符号：p 是 stage 数（模型按层切成 p 段，每段一张卡），m 是 micro-batch 的**个数**，$t_f, t_b$ 是一个 micro-batch 在一个 stage 上前向、反向的时间。\n1. 前向像流水线（见图，GPipe 的调度，arXiv 1811.06965）：第 1 个 micro-batch 要依次走过 p 个 stage，最后一个 stage 要等 p − 1 格才有活干；第一个 stage 送走最后一个 micro-batch 后，也要空等 p − 1 格。\n2. 前向共 $m + p - 1$ 格，每个 stage 只有 m 格在干活；反向一样。总时间 $(m + p - 1)(t_f + t_b)$，有用的是 $m(t_f + t_b)$。\n3. bubble 和有用时间之比：\n$$\\frac{(p - 1)(t_f + t_b)}{m\\,(t_f + t_b)} = \\frac{p - 1}{m}$$\n占总时间是 $\\frac{p - 1}{m + p - 1}$。\n代入 p = 4：m = 8 时 3/8 = 37.5%（占总时间 27%）；m = 32 时 9.4%（8.6%）。所以要 $m \\gg p$。\n怎么减：\n1. **1F1B**（PipeDream，arXiv 1806.03377）：每个 stage 前向反向交替做。bubble 不变，但每个 stage 同时只存最多 p 个 micro-batch 的激活，不是 m 个，m 才敢开大。\n2. **interleaved**（Megatron，arXiv 2104.04473）：每张卡放 v 段不连续的层，bubble 降到 $\\frac{p - 1}{m v}$，代价是 stage 间通信多 v 倍。\n推理时 decode 每步的 batch 本来就小，拆不出很多 micro-batch，PP 主要用在显存放不下的时候。',
    fig: lines`
      forward, p = 4, m = 4
      digit = micro-batch, . = idle

      time     1 2 3 4 5 6 7
      stage 0  1 2 3 4 . . .
      stage 1  . 1 2 3 4 . .
      stage 2  . . 1 2 3 4 .
      stage 3  . . . 1 2 3 4

      each stage: m busy + (p-1) idle
    `,
    ref: PAR,
  }),
  b({
    id: 'bagu-sp-cp',
    topic: '并行',
    q: 'TP = 8 时，LayerNorm、dropout 那段的激活在每张卡上都是完整的一份。sequence parallel（SP）怎么把它降到 1/8？为什么通信量不变？',
    a: '先看 TP 的一层（见 Megatron TP 那张卡）：attention 和 MLP 里的矩阵乘按 head / 列切开，每卡只有 1/N；但两段之间的 LayerNorm、dropout、残差加，每张卡都拿着**完整的** [T, d] 激活，重复做同样的计算。\n例：T = 8192 个 token，d = 8192，bf16，一个这样的张量是 8192 × 8192 × 2 B = **128 MiB**，每层有好几个，8 张卡各存一份一样的。\nSP（Korthikanti et al. arXiv 2205.05198）：\n1. LayerNorm 和 dropout 对每个 token **独立**计算，所以这段可以按序列切：每卡只拿 T/N 个 token，[T/N, d]，128 MiB → 16 MiB。\n2. 进入 TP 区需要完整的输入：用 **all-gather** 把 T/N 拼回 T。\n3. 离开 TP 区本来是 all-reduce 求部分和；改成 **reduce-scatter**：求和的同时按 token 切开，每卡只拿自己那 T/N 行，直接进下一段 LayerNorm。\n4. 通信量：ring 上 all-reduce 每卡发 $\\frac{2(N-1)}{N} D$，reduce-scatter、all-gather 各 $\\frac{N-1}{N} D$，加起来一样。all-reduce 本来就是这两步拼成的。\n所以 SP 是白捡的：这段激活降到 1/N，通信不变（见图）。它只管 attention 外面那段；attention 里面还是按 head 切，每卡仍要处理完整的序列长度，那是 CP 解决的（下一张卡）。',
    fig: lines`
      one TP layer, N cards

      [T/N, d]  LayerNorm, dropout  (SP)
         | all-gather
      [T, d]    attn / MLP by head  (TP)
         | reduce-scatter
      [T/N, d]  LayerNorm, dropout  (SP)

      all-reduce = reduce-scatter + all-gather
    `,
    ref: PAR,
  }),
  b({
    id: 'bagu-sp-cp-2',
    topic: '并行',
    q: 'Llama-3-70B 做 128k 上下文的 prefill，context parallel（CP）= 8。每张卡算什么、传什么？通信能被计算藏住吗？',
    a: '符号：S 序列长度，N 张卡，d hidden 维度；KV 是每 token 每层的 K、V 字节，70B 是 $2 \\times 8 \\times 128 \\times 2$ = 4 KiB。\n问题：S = 128k 时 attention 的 FLOPs ∝ $S^2$。TP 按 head 切，每卡仍要处理完整的 128k 个 token。CP 把**序列**切开：\n1. 每卡拿连续的 S/N = 16k 个 token，算这一段的 Q、K、V。\n2. Q 留在本卡；K、V 块沿环转 N − 1 步，每步每卡用自己的 Q 和手上那块 K、V 算一次局部 attention（因果时整块在后面的直接跳过）。\n3. 各块结果用 online softmax 合并（和 FlashAttention 块间合并一样），结果严格等于整段 attention。这就是 Ring Attention（arXiv 2310.01889）。\n能不能藏住，算一步的两边：\n1. 计算：$QK^\\top$ 和 $PV$ 每层约 $4d\\,(S/N)^2$ = 4 × 8192 × 16384² ≈ 8.8 TFLOP，H100 满算力约 **9 ms**。\n2. 通信：一块 K、V = 16384 × 4 KiB = 64 MiB，IB 50 GB/s 约 **1.3 ms**。\n计算是通信的约 7 倍，下一块 K、V 在算当前块的时候就收完了。这个比值 ∝ S/N，序列越长越好藏。\n和 SP 的区别：SP 切的是 attention **外面**的 LayerNorm、dropout，通信量和 TP 相同；CP 切的是 attention **里面**的序列，用传 K、V 换掉每卡 $S^2$ 的计算和激活。',
    ref: PAR,
  }),
  b({
    id: 'bagu-parallel-choice',
    topic: '并行',
    q: '用 H100（80 GB，每节点 8 卡）部署两个模型：Llama-3-70B bf16（权重 140 GB）和 Llama-3.1-405B bf16（约 810 GB）。TP、PP、DP 各怎么定？',
    a: '先定三条约束，再代数字：\n1. TP 每层两次 all-reduce、在关键路径上，只能在 NVLink 域内：**TP ≤ 8**。GQA 下还要 **TP ≤ H_kv**（这两个模型都是 8），否则 KV head 要复制。\n2. PP 只在 stage 边界传一次激活，可以跨节点，但有 bubble：**放得下就不用**。\n3. 剩下的卡做 **DP**（多副本）：推理时副本之间不通信，吞吐线性涨。\n70B：单卡放不下 140 GB。按 90% 显存可用（72 GB）算：TP = 2 每卡权重 70 GB，只剩约 2 GB 给 KV；TP = 4 每卡 35 GB，再留几 GB 给激活，剩约 30 GB 给 KV。所以 **TP = 4**，一台 8 卡机器放 2 个副本（DP = 2）。更看重单请求延迟时用 TP = 8，每步读的权重更少、副本更少。\n405B：810 GB > 8 × 80 = 640 GB，一个节点放不下，TP 又不能跨节点，所以 **节点内 TP = 8 × 跨节点 PP = 2**。Llama 3 论文的 405B bf16 推理就是这么做的（arXiv 2407.21783 第 6.1 节）。换成 FP8 权重约 405 GB，一个节点 TP = 8 就放得下，PP 就省了。\nMoE 的专家另用 EP，超长上下文加 CP，原则一样：先看什么放不下，再看哪条链路快。',
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
    q: 'DeepSeek-V3 有 256 个路由专家，每个 token 选 8 个。单卡 batch 64 时每个专家分到几个 token、算术强度多少？要多少 token 才到 H100 的 ridge？',
    a: '符号：E 路由专家数（256），k 每个 token 选几个（8），T 这一步的 token 总数，N 卡数，b 每卡的 token 数。\n1. 均匀路由时每个专家分到 $m = Tk/E$ 个 token。专家的 GEMM 读一遍专家权重、只算 m 个 token，bf16 下 AI ≈ m（推法同 decode：每个权重 2m FLOP、2 字节）。\n2. T = 64：m = 64 × 8 / 256 = **2**，AI ≈ 2，比同 batch 的 dense 模型（AI = 64）低 E/k = 32 倍。\n3. 更糟的是专家几乎全被点到：某个专家一个 token 都没分到的概率是 $(1 - k/E)^T$ = 0.969⁶⁴ ≈ 13%，所以约 **87%** 的专家权重都要读一遍，只为 64 个 token。\n4. 要 m ≈ 295，T ≈ 295 × 32 ≈ **9400** 个 token。一张卡的 KV 显存撑不住这么大的 batch。\n解法是 EP：N 张卡各跑 b 个 token，专家分散到各卡，token 用 all-to-all 发到专家所在的卡：\n$$m = \\frac{N b k}{E}$$\nDeepSeek 公开的 decode 配置是 EP128、每卡 128 个请求：m = 128 × 128 × 8 / 256 = **512**，越过 ridge；同时每卡只读 1/N 的专家权重（DeepSeek-V3 技术报告 arXiv 2412.19437，配置见 https://github.com/deepseek-ai/profile-data ）。',
    ref: MOE,
  }),
  b({
    id: 'bagu-moe-batch-2',
    topic: '并行',
    q: 'DeepSeek-V3 部署时 attention 用 DP、专家用 EP（DP attention）。如果 attention 改用 TP = 8，MLA 的 KV cache 会出什么问题？',
    a: '先说 TP 怎么切 KV：TP 按 head 切，每卡只存自己那几个 KV head 的缓存。GQA 有 8 个 KV head 时，TP = 8 每卡 1 个，KV 正好 1/8。\n1. MLA 的缓存是一份所有 head 共用的 576 维 latent（见 MLA 那张卡），没有 KV head 可切。TP = 8 时每张卡都要存**完整的** latent，KV 显存是 8 份一样的。\n2. **DP attention**：attention 不切，每张卡放一份完整的 attention 权重，只处理分给自己的那批请求，每个请求的 KV 只存在一张卡上。\n3. 权重放得下吗：V3 每层 attention 约 1.9 亿参数（$W^{DQ}, W^{UQ}, W^{DKV}, W^{UK}, W^{UV}, W^O$），61 层约 **114 亿**，FP8 约 11 GB，每卡复制一份可以接受。占大头的约 6500 亿专家参数才用 EP 切开。\n4. 层内流程：每卡算完自己请求的 attention → all-to-all 把 token 发给专家所在的卡 → 专家算完再发回原卡。\nSGLang 给 DeepSeek 模型做了 DP attention，报告 decode 吞吐最多提升 1.9 倍（作者自测，https://lmsys.org/blog/2024-12-04-sglang-v0-4/ ）。',
    ref: PAR,
  }),
  b({
    id: 'bagu-zero',
    topic: '并行',
    q: '7B 模型全参微调，8 张卡数据并行，混合精度 Adam。普通 DDP 每卡要 112 GB，放不下。ZeRO-1 / 2 / 3 各切掉什么，每卡剩多少？',
    a: '符号：$\\Psi$ 是参数个数（7 × 10⁹），N 是数据并行的卡数（8）。混合精度 Adam 每个参数存 16 B：bf16 参数 2 + bf16 梯度 2 + 优化器状态 12（fp32 主权重、m、v 各 4，为什么要这些见训练显存那张卡）。下面都不含激活。\n1. **DDP**：每卡一份完整的 $16\\Psi$ = 112 GB。8 张卡存 8 份一模一样的东西，还做一模一样的优化器更新。\n2. 关键观察：Adam 更新第 i 个参数，只用到第 i 个参数自己的梯度、m、v、主权重，和别的参数无关。所以可以分工：参数均分成 N 段，卡 k 只更新第 k 段、只存第 k 段的状态（ZeRO，arXiv 1910.02054）。\n3. **ZeRO-1** 切优化器状态：$2\\Psi + 2\\Psi + 12\\Psi/N$ = 14 + 14 + 10.5 = **38.5 GB**。\n4. **ZeRO-2** 再切梯度（reduce-scatter 以后别人那段梯度用不上了，直接丢）：$2\\Psi + 14\\Psi/N$ = 14 + 12.25 = **26.3 GB**。\n5. **ZeRO-3** 连参数也切：$16\\Psi/N$ = **14 GB**。算某一层之前临时 all-gather 出这层的完整参数，算完就丢。\n通信（每卡约发多少，单位是 Ψ 个元素）：DDP 的梯度 all-reduce = reduce-scatter + all-gather，约 2Ψ。ZeRO-1/2 只是把这两半拆开，中间插进优化器更新，还是 2Ψ。ZeRO-3 前向、反向各 all-gather 一次参数，再 reduce-scatter 梯度，约 3Ψ，**多 50%**。PyTorch 的 FSDP 就是 ZeRO-3。',
    fig: lines`
      7B, N = 8: bytes per param -> per GPU

              param  grad  optim  total
      DDP       2     2     12    112 GB
      ZeRO-1    2     2     12/8  38.5 GB
      ZeRO-2    2     2/8   12/8  26.3 GB
      ZeRO-3    2/8   2/8   12/8  14 GB
    `,
    ref: ZERO,
  }),
  b({
    id: 'bagu-zero3-vs-tp',
    topic: '并行',
    q: 'ZeRO-3 和 TP 都把参数切到 N 张卡上。Llama-2-7B（d = 4096，每层约 2 亿参数）、N = 8、一个 micro-batch 4096 个 token，两者每层各通信多少？token 多到多少时 ZeRO-3 反而更省？',
    a: '先说定义上的区别：\n1. **ZeRO-3 切存储**：每卡平时只存 1/N 的参数，算某层前 all-gather 拼回完整参数，**每卡算完整的层**，处理自己那份数据。通信的是**参数**，和 token 数无关。\n2. **TP 切计算**：每卡一直只有 1/N 的参数，只算自己那一片，用 all-reduce 把部分结果加起来。通信的是**激活**，和 token 数成正比。\n每层的通信（ring，每卡发送；$P_\\ell$ 每层参数，T token 数，b = 2 字节）：\n1. ZeRO-3：前向 all-gather + 反向 all-gather + 梯度 reduce-scatter，各 $\\frac{N-1}{N} P_\\ell b$：\n$$3 \\times \\tfrac{7}{8} \\times 2.02 \\times 10^8 \\times 2 \\approx 1.06\\ \\text{GB}$$\n2. TP：前向 2 次、反向 2 次 all-reduce，各 $\\frac{2(N-1)}{N} T d b$：\n$$4 \\times 1.75 \\times 4096 \\times 4096 \\times 2 \\approx 0.23\\ \\text{GB}$$\n3. 两者相等时 $3P_\\ell = 8Td$，$T = 3P_\\ell / (8d) \\approx$ **1.85 万 token**。一个 micro-batch 的 token 比这多，ZeRO-3 的通信就更少。\n别的区别：ZeRO-3 的 all-gather 可以提前预取下一层、和计算重叠，也能跨节点；TP 的 all-reduce 在关键路径上、只能在 NVLink 内，但它把每卡的计算和激活也切成了 1/N，ZeRO-3 不切激活。大规模训练常把两者组合：节点内 TP，节点间 ZeRO / DP。',
    ref: ZERO,
  }),

  // ---------------- GPU ----------------
  b({
    id: 'bagu-gpu-coalescing',
    topic: 'GPU',
    q: '行主序 float 矩阵，一个 warp 的 32 个线程（t = 0..31）分别读 `A[row][t]` 和 `A[t][col]`，各要读多少个 32 B sector？转置怎么避开不合并的那一边？',
    a: 'global memory 按 32 B sector 读，一个 warp 的访问会合并成尽量少的 sector。\n1. `A[row][t]`：32 个 float 地址连续，共 128 B = **4 个 sector**，每个字节都有用。\n2. `A[t][col]`：相邻线程差一整行，每个线程落在不同的 sector，要 **32 个 sector**，每个 32 B 只用 4 B，带宽利用率 4 / 32 = **12.5%**。\n转置必然有一边按列：读和写里总有一个跨行。做法是让 global 的读和写都按行：一个 block 先按行把 32×32 的 tile 读进 shared memory，再从 shared memory 按列取出、按行写回。跨行的那一步挪到了片上，shared memory 没有 sector 浪费（但要处理 bank conflict，见下一张）。\n出处：Mark Harris《How to Access Global Memory Efficiently in CUDA C/C++ Kernels》 https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/ 和《An Efficient Matrix Transpose in CUDA C/C++》 https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/ （有各版本的带宽实测）；CUDA C++ Best Practices Guide「Coalesced Access to Global Memory」 https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ 。',
    ref: GPU,
  }),
  b({
    id: 'bagu-gpu-bank-conflict',
    topic: 'GPU',
    q: '`__shared__ float tile[32][32]`，一个 warp 读一列 `tile[t][c]`（t = 0..31），是几路 bank conflict？为什么声明成 `tile[32][33]` 就没了？',
    a: 'shared memory 分 32 个 bank，每 4 B 一个 bank，下标 i 的 float 落在 bank `i % 32`。同一 warp 里多个线程访问同一 bank 的**不同地址**会串行（同一地址是广播，不冲突）。\n1. `[32][32]`：`tile[t][c]` 下标 `32t + c`，bank 都是 `c`，32 个线程打到同一个 bank，**32 路冲突**，这次读要串行 32 次。\n2. `[32][33]`：下标 `33t + c`，bank 是 `(t + c) % 32`，t 从 0 到 31 各不相同，32 个线程落在 32 个 bank，**无冲突**。代价只是每行多 1 个 float（3%）。\n按行读 `tile[r][t]` 两种声明都没冲突，所以 padding 专门解决矩阵转置这种按列读 tile 的情况。\n出处：Mark Harris《Using Shared Memory in CUDA C/C++》 https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/ ；转置博客里 `[32][32]` → `[32][33]` 的带宽对比 https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/ 。',
    ref: GPU,
  }),
  b({
    id: 'bagu-gpu-occupancy',
    topic: 'GPU',
    q: 'H100 每个 SM 有 65536 个寄存器、最多驻留 64 个 warp。kernel 每线程用 128 个寄存器，occupancy 是多少？要不要想办法提到 100%？',
    a: 'occupancy = SM 上驻留的 warp 数 / 上限。warp 多，一个 warp 等内存时就有别的 warp 可换上，延迟被藏住。\n1. 每个 warp 要 128 × 32 = 4096 个寄存器，65536 / 4096 = **16 个 warp**，occupancy 16 / 64 = **25%**。\n2. 要 100% 得把寄存器压到 65536 / 2048 = 32 个/线程，多数计算 kernel 放不下，会溢出（spill）到 local memory，反而变慢。\n**不一定要提**：目标是藏住延迟，不是 occupancy 本身。每个线程多发几个互不依赖的 load（ILP），少量 warp 也能跑满带宽；matmul、FlashAttention 这类 kernel 常年低 occupancy、用大量寄存器存 tile。确实是延迟没藏住时，再用 `__launch_bounds__` 限制寄存器或减少 shared memory 用量，并用 Nsight Compute 确认。\n出处：Vasily Volkov《Better Performance at Lower Occupancy》（GTC 2010） https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf ；CUDA C++ Best Practices Guide「Occupancy」 https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ ；Simon Boehm 的 matmul worklog 在真实 kernel 上按寄存器和 shared memory 算 occupancy https://siboehm.com/articles/22/CUDA-MMM 。',
    ref: GPU,
  }),
  b({
    id: 'bagu-warp-divergence',
    topic: 'GPU',
    q: '下面两个版本，哪个会 warp divergence，各要多少周期？causal attention 在 S = 4096、tile 128 时，有多少个 tile 需要逐元素 mask？',
    qcode: lines`
      // a() 和 b() 各要 100 个周期
      int t = threadIdx.x, w = t / 32;   // w: warp 编号
      // 版本 1
      if (t % 2 == 0) a(); else b();
      // 版本 2
      if (w % 2 == 0) a(); else b();
    `,
    a: 'warp 是 32 个线程，**共用一条指令流**：同一时刻发一条指令，32 个线程一起执行。分支时如果线程要走不同的路，硬件只能把每条路依次走一遍，不在当前这条路上的线程被屏蔽、空等，这就是 divergence（CUDA 编程指南 SIMT 一节 https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html ）。\n1. 版本 1：同一个 warp 里偶数线程走 a、奇数线程走 b，**divergence**。这个 warp 先跑 a（一半线程闲着）再跑 b（另一半闲着），**200 个周期**，慢一倍。\n2. 版本 2：`w = threadIdx.x / 32` 是 warp 编号，同一个 warp 的 32 个线程条件相同，整个 warp 只走一条路，**100 个周期**。不同 warp 走不同的路没关系，warp 之间本来就各自调度。\n3. 常见的无害分支：边界检查 `if (i < n)`，只有最后一个 warp 里有部分线程为假，代价可以忽略。\ncausal attention 为什么基本不受影响：mask 先按 tile 判断（见图），一个 tile 要么全可见、要么全被 mask，这个判断对整个 warp 是一样的。S = 4096、tile 128：一共 32 × 32 个 tile，上三角 496 个整块跳过，下三角 496 个全可见、直接算，只有对角线上 **32 个**（要算的 528 个里的 6%）需要逐元素 mask。逐元素 mask 也用 `where(allowed, s, -inf)` 这样的选择指令，所有线程执行同一条指令，不分支。',
    fig: lines`
                   K tile
                   0  1  2  3
        Q tile 0   D  .  .  .
               1   F  D  .  .
               2   F  F  D  .
               3   F  F  F  D

      F: fully visible, no mask
      D: diagonal, mask per element
      .: fully masked, skipped
    `,
    ref: GPU,
  }),
  b({
    id: 'bagu-gemv',
    topic: 'GPU',
    q: 'decode 时 MLP 的 up 投影：x 是 [B, 8192]，W 是 [8192, 28672]，bf16。B = 1、16、256 时算术强度各是多少？为什么 Tensor Core 打不满？',
    a: '符号：x 是 [B, K]，W 是 [K, N]，K = 8192，N = 28672，bf16 每个元素 2 B。B 是 batch（decode 时每个序列 1 个 token）。\n1. FLOPs：$2BKN$（每个权重对每个 token 一次乘加）。\n2. 字节：读 W 是 $2KN$，读 x 是 $2BK$，写 y 是 $2BN$。B 远小于 K、N 时几乎全是读 W。\n$$\\text{AI} = \\frac{2BKN}{2KN + 2B(K + N)} \\approx B$$\n代入：B = 1 → 1.0，B = 16 → 16.0，B = 256 → 246。\n3. 可达性能 = min(峰值, 带宽 × AI)。H100 上 B = 1 时是 3.35 TB/s × 1 = 3.35 TFLOP/s，只有峰值 989 的 **0.3%**；B 要到两三百才接近 ridge 295。\n为什么：GEMM 快，是因为每个权重从 HBM 读进来以后，在 shared memory 和寄存器里被很多行输入反复使用。B = 1 时每个权重只用一次（这就是 GEMV，矩阵乘向量），Tensor Core 算完马上就要等下一批数据。\n对策：\n1. 加大 B（continuous batching）：AI 直接跟着涨。\n2. 少读字节：W4A16 每个权重只读 0.5 B，同样的 B 下 AI 是 4 倍。\n3. 让足够多的 SM 一起读：比如 o_proj 的 W 是 [8192, 8192]，每个 block 负责 128 列输出时只有 64 个 block，H100 有 132 个 SM，一半闲着，连带宽都跑不满。**split-K** 再把 K 维切成 4 段，256 个 block 各读一段，最后把部分和加起来。',
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
    q: '为什么推理框架要把残差加和 RMSNorm 融合成一个 kernel？它在层里的什么位置？',
    a: 'pre-norm 的 decoder 层：$h = x + \\text{Attn}(\\text{Norm}(x))$，$y = h + \\text{MLP}(\\text{Norm}(h))$。\n位置：融合的是「**残差加 + 紧跟着的 norm**」，不是 norm + linear，每层两处（见图和代码）：\n1. o_proj 之后：$h = x + \\text{attn\\_out}$，紧接着 $\\text{Norm}(h)$ 喂给 MLP。\n2. down_proj 之后：加回残差，紧接着下一层的 input norm。第一层前面还没有残差，只做 norm。\n为什么融合：两步都是 memory-bound 的逐元素 / 逐行操作。分开做是 add 读 x 和 attn_out、写 h，norm 再读 h、写结果；融合后 h 留在寄存器里，少读一遍 h，少一次 kernel launch。\n单次省得不多，但每层两处、80 层就是 160 次，decode 下累积可观。SiLU × up、RoPE + 写 KV 也是同理。',
    fig: lines`
      x (residual)
       |
      [add + norm]   input_layernorm
       |
      qkv_proj -> attention -> o_proj
       |
      [add + norm]   post_attention_layernorm
       |
      gate_up -> SiLU * up -> down_proj
       |
       v  next layer: [add + norm]
    `,
    code: lines`
      # vllm/model_executor/models/llama.py  LlamaDecoderLayer.forward
      if residual is None:   # 第一层：还没有残差
          residual = hidden_states
          hidden_states = self.input_layernorm(hidden_states)
      else:                  # 融合：residual += hidden_states; hidden_states = norm(residual)
          hidden_states, residual = self.input_layernorm(hidden_states, residual)
      hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
      hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
      hidden_states = self.mlp(hidden_states)
      return hidden_states, residual
    `,
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
    q: '一个请求在 vLLM 里从 HTTP 到第一个 token，经过哪几个阶段？',
    a: '四个大阶段（见图），以一句「你好」为例：\n1. **前端**（API server 进程）：HTTP 请求进来 → 套 chat template（加上角色标记）→ tokenize 成一串 token id。\n2. **调度**（EngineCore 进程）：进 waiting 队列 → 查 prefix cache、分配 KV block → 放进这一步的 batch。\n3. **执行**（GPU worker）：prefill 前向算完整个 prompt → 最后位置的 logits → 采样出第一个 token id。\n4. **回传**（前端）：detokenize 成文字 → SSE 推给客户端，**TTFT 到此结束**。\n之后每步 decode 重复 2–4，每步出一个 token；结束后释放 KV block。',
    fig: lines`
      client
        |  HTTP: "你好"
        v
      API server   template, tokenize
        |  ZMQ: token ids
        v
      EngineCore   queue, KV blocks, batch
        |
        v
      GPU worker   prefill, logits, sample
        |  token id
        v
      API server   detokenize, SSE
        |                      <- TTFT ends
        v
      client
    `,
    ref: LIFE,
  }),
  b({
    id: 'bagu-tokenize-event-loop',
    topic: '框架',
    q: 'API server 用 asyncio 同时服务上千个连接，为什么 tokenize 要挪到线程池或单独的进程，不能在请求处理函数里直接调？',
    a: '1. asyncio 是**一个线程**跑很多协程：一个协程只有在 `await` 时才让出，事件循环才能去处理别的连接（收新请求、往 SSE 里推 token）。\n2. tokenize 是同步的 CPU 计算，长 prompt 要几到几十毫秒，中间没有 `await`。直接调用的话，这几十毫秒里**所有连接都卡住**，正在流式输出的请求 ITL 一起抖。\n3. 解法：放进线程池（HF tokenizers 是 Rust 实现，计算时释放 GIL，能真并行），或者放进独立进程。vLLM V1 把 tokenize / detokenize 放在前端进程里，和跑调度的 EngineCore 进程分开。',
    code: lines`
      async def handle(req):
          ids = tokenizer.encode(req.prompt)   # 同步 20 ms：整个事件循环停 20 ms

      async def handle(req):
          loop = asyncio.get_running_loop()
          ids = await loop.run_in_executor(pool, tokenizer.encode, req.prompt)
          # await 期间事件循环继续服务别的连接
    `,
    ref: PY,
  }),
  b({
    id: 'bagu-vllm-v1',
    topic: '框架',
    q: 'vLLM V1 的进程架构？为什么把 EngineCore 单独放一个进程？',
    a: '三层进程（见图）：前端进程（HTTP、tokenize、detokenize）↔ ZMQ ↔ **EngineCore 进程**（调度 + KV block 管理）→ 每张卡一个 worker 进程（model runner）。TP = 1 时 worker 就在 EngineCore 进程里。\n为什么拆：V0 里这些在同一个 Python 进程里串行，CPU 干活时 GPU 空转。拆开后 CPU 的工作和 GPU 执行重叠，GPU 利用率明显提升；代价是多一次进程间序列化。\nV1 的 scheduler 不再区分 prefill / decode batch，每个请求每步分配若干 token，chunked prefill 和 prefix caching 默认开。',
    fig: lines`
      API server process
        HTTP, tokenize, detokenize
           |  ZMQ
           v
      EngineCore process
        scheduler, KV block manager
           |  one batch per step
           v
      worker 0   worker 1   ...   worker N-1
        one process per GPU, runs the model
    `,
    ref: V1,
  }),
  b({
    id: 'bagu-flat-input',
    topic: '框架',
    q: 'vLLM 的模型 forward 为什么输入是 `[num_tokens, H]` 而不是 `[B, S, H]`？',
    a: 'continuous batching 下，一个 batch 里既有 prefill 的长序列，也有 decode 的单个 token。\n例（见图）：A 在 prefill 1000 个 token，B、C 各 decode 1 个。补成矩形 `[B, S, H]` 要 3 × 1000 = 3000 行，有效的只有 1002 行，**三分之二是 padding**，白算还白占显存。\n所以把所有请求的 token 拼成一维 `[num_tokens, H]`，再用 metadata 告诉 attention kernel 怎么分组：每个序列的起止（`cu_seqlens`）、每个 token 的 KV 写到哪（slot mapping）、block table。Linear 层本来就和 token 怎么分组无关，直接算。',
    fig: lines`
      [B, S, H]: pad every sequence to S = 1000
      A  ####################   1000
      B  #...................      1
      C  #...................      1
         real 1002 of 3000 rows

      [num_tokens, H]: concatenate
         A x 1000 | B | C   = 1002 rows
         cu_seqlens = [0, 1000, 1001, 1002]
    `,
    ref: ADD,
  }),
  b({
    id: 'bagu-torch-compile',
    topic: '框架',
    q: '下面的函数被 torch.compile 编译，会切成几段图？torch.compile 的三段（Dynamo、AOTAutograd、Inductor）各做什么？',
    qcode: lines`
      @torch.compile
      def f(x):
          y = torch.relu(x) * 2
          if y.sum() > 0:
              return y + 1
          return y - 1
    `,
    a: '先说目标：eager 模式下每个算子是一次 Python 调用、一个 kernel，`relu`、`* 2`、`+ 1` 是三个 kernel，各自读写一遍显存。torch.compile 想先把 Python 函数抓成一张图，再把图编译成更少、更快的 kernel（PyTorch 2 论文 https://pytorch.org/assets/pytorch2-2.pdf ）。\n1. **Dynamo（抓图）**：在 Python 执行函数的字节码之前接管，逐条符号执行，把张量操作记进一张 FX 图，同时记下 **guard**，也就是这张图成立的前提，比如 x 是 float32、形状 [8, 4096]。下次调用先检查 guard：满足就直接跑编译好的代码；不满足（比如形状变成 [16, 4096]）就重新编译。\n2. **graph break**：Dynamo 只记录对张量做了什么，不知道张量里的**值**。`if y.sum() > 0` 要看值才能决定走哪支，图只能在这里断开：第一段图算到 `y.sum() > 0`，回到普通 Python 判断条件，再从分支里开第二段图。所以本题是 **2 段**：`relu, * 2, sum` 一段，`+ 1`（或 `- 1`）一段。`print`、不支持的第三方调用也会断。断得越多越接近 eager，用 `TORCH_LOGS=graph_breaks` 查断在哪。\n3. **AOTAutograd**：训练时提前把反向图也 trace 出来，前向、反向都拆成底层的 ATen 算子，交给后端一起优化。\n4. **Inductor（生成代码）**：把相邻的逐元素、归约算子融合，生成 Triton kernel（GPU）或 C++（CPU）；矩阵乘默认仍调 cuBLAS。\n融合省多少：x 是 100 万个 fp32（4 MB）。eager 下 relu、× 2、+ 1 三个 kernel 各读 4 MB、写 4 MB，共 24 MB；融合成一个 kernel 只读一次、写一次，8 MB。这类算子是 memory-bound，时间约降到 1/3。',
    ref: COMPILE,
  }),
  b({
    id: 'bagu-caching-allocator',
    topic: '框架',
    q: 'PyTorch 的 caching allocator 做了什么？显存碎片怎么看出来？',
    a: '先和 Java 对比：Java GC 管的是对象**什么时候死**；PyTorch 张量靠 Python 引用计数，引用一归零就释放。caching allocator 管的是**释放后的显存去哪**，更像 malloc 的空闲链表。\n1. 为什么要它：`cudaMalloc` / `cudaFree` 很慢，`cudaFree` 还会同步整个 device。\n2. 怎么做：向驱动要一大段显存（segment），切成块给张量用；张量释放后块回到池子（按 stream 分开），**不还给驱动**；下次分配找够大的最小空闲块，大了就切开，相邻的空闲块合并。\n3. 碎片：Java GC 会搬动对象、把空隙压实；PyTorch 不能搬，张量的地址已经交给 kernel 了。空闲块散在各处就拼不成大块（见图），表现是 **reserved 远大于 allocated**，明明还有空闲却 OOM。\n缓解：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（用虚拟内存让 segment 能原地变长）、尽量固定 shape、推理框架启动时一次性预分配 KV 池。`empty_cache()` 只把整段空闲的 segment 还给驱动，解决不了碎片，还会让之后的分配变慢。',
    fig: lines`
      segment, 20 MiB
      [used 4][free 6][used 4][free 6]

      malloc(8 MiB): no free block >= 8
        -> new cudaMalloc, or OOM
      reserved 20, allocated 8
      free 12, but no block fits
    `,
    ref: PTI,
  }),

  // ---------------- Post-train ----------------
  b({
    id: 'bagu-train-memory',
    topic: 'Post-train',
    q: '混合精度 Adam 训练，每个参数要存哪 5 样东西？各自为什么非存不可？7B 全参微调光这些就要多少？',
    a: '从一个训练 step 走一遍（代码见下），看每一步用到什么：\n1. **bf16 参数 w（2 B）**：前向、反向的矩阵乘都用它。bf16 比 fp32 算得快、占一半显存。\n2. **bf16 梯度 g（2 B）**：反向给每个参数算出 $\\partial L / \\partial w$，个数和参数一样多，要一直存到优化器用完。\n3. **Adam 的 m（4 B）**：梯度的滑动平均 $m \\leftarrow \\beta_1 m + (1 - \\beta_1) g$，也就是「动量」，让更新方向更稳。它要跨 step 累积，所以**每个参数一个、一直存着**。\n4. **Adam 的 v（4 B）**：梯度平方的滑动平均 $v \\leftarrow \\beta_2 v + (1 - \\beta_2) g^2$。更新量是 $\\eta\\, m / (\\sqrt{v} + \\epsilon)$：梯度一直很大的参数步子小，一直很小的步子大，每个参数有自己的步长（Adam，arXiv 1412.6980）。同样跨 step 累积。\n5. **fp32 主权重（4 B）**：$\\eta\\, m / \\sqrt v$ 常常只有 1e-4 量级，直接加到 bf16 的 w 上会被舍入掉（见 bf16 主权重那张卡），所以在一份 fp32 副本上更新，再 cast 成 bf16 的 w。m、v 用 fp32 也是这个原因。\n合计 2 + 2 + 4 + 4 + 4 = **16 B / 参数**（ZeRO 论文的记法，arXiv 1910.02054）。7B：7 × 10⁹ × 16 = **112 GB**，一张 80 GB 的 H100 放不下，而且还没算激活（前向留给反向用的中间结果，另算）。\n各框架略有出入：Megatron 把梯度累加在 fp32 里（梯度 4 B，合计 18）；8-bit Adam 把 m、v 各压到 1 B（合计 10）。',
    fig: lines`
      per parameter   bytes   dtype
      w               2       bf16
      grad            2       bf16
      master w        4       fp32
      Adam m          4       fp32
      Adam v          4       fp32
      total           16
    `,
    code: lines`
      # 一个训练 step（省略 Adam 的偏差修正）
      loss = model(x, weights=w)
      g = grad(loss, w)
      g32 = g.float()
      m = b1 * m + (1 - b1) * g32
      v = b2 * v + (1 - b2) * g32 ** 2
      w32 -= lr * m / (v.sqrt() + eps)
      w = w32.to(bfloat16)
    `,
    ref: TMEM,
  }),
  b({
    id: 'bagu-float-formats',
    topic: 'Post-train',
    q: 'fp16 和 bf16 都是 16 位，各给指数几位、尾数几位？为什么大模型训练从 fp16 换成了 bf16？',
    a: '位数见图（fp32 放着对照）。\n1. **指数位定范围**：fp16 只有 5 位，最大 65504，最小正规数约 $6 \\times 10^{-5}$，小梯度会下溢成 0，得靠 loss scaling 救。bf16 和 fp32 一样是 8 位，范围约 $10^{-38}$ 到 $3 \\times 10^{38}$，不会下溢。\n2. **尾数位定精度**：相邻两数的相对间隔约 $2^{-m}$（m 是尾数位数）。fp16 约 0.001，bf16 约 0.008，只有两三位有效数字。\n换 bf16 是拿精度换范围：范围不够会直接丢梯度，还得调 loss scaling；精度不够可以靠 fp32 主权重补。两边的算例见 1e-8 梯度和 fp32 主权重那两张卡。',
    fig: lines`
              sign   exp   mantissa
      fp32     1      8      23
      fp16     1      5      10
      bf16     1      8       7
    `,
    ref: TMEM,
  }),
  b({
    id: 'bagu-float-formats-2',
    topic: 'Post-train',
    q: 'fp8 有两种：e4m3（4 位指数、3 位尾数）和 e5m2（5 位指数、2 位尾数）。为什么要两种？训练时分别给谁用？',
    a: '8 位去掉符号位只剩 7 位，指数和尾数怎么分都顾此失彼，所以分成两种（Micikevicius et al. arXiv 2209.05433）：\n1. **e4m3 精度高、范围小**：相邻两数相对间隔 $2^{-3}$ = 12.5%，最大 448。给**前向的权重和激活**用：它们的数值集中，更需要精度。\n2. **e5m2 范围大、精度低**：相对间隔 $2^{-2}$ = 25%，最大 57344，范围和 fp16 一样。给**反向的梯度**用：梯度跨度大，更需要范围。\n两种的范围都比 bf16 窄得多，所以每个张量还要配一个 fp32 缩放系数，先把数值挪进可表示范围再 cast。',
    fig: lines`
              sign   exp   mantissa     max
      e4m3     1      4       3         448
      e5m2     1      5       2       57344
    `,
    ref: TMEM,
  }),
  b({
    id: 'bagu-bf16-fp16',
    topic: 'Post-train',
    q: 'fp16 有 5 位指数，bf16 有 8 位（位数见浮点格式那张卡）。一个梯度是 1e-8，存成 fp16 和 bf16 各变成多少？fp16 训练的 loss scaling 怎么救它？',
    a: '先回顾浮点数：值 = $(-1)^s \\times 2^{e - \\text{bias}} \\times 1.f$。指数 e 的位数决定能表示多大、多小的数，尾数 f 的位数决定精度。\n1. fp16 指数 5 位，最小的正规数是 $2^{-14} \\approx 6.1 \\times 10^{-5}$；再往下还有非正规数（尾数前面不补 1），最小到 $2^{-24} \\approx 6 \\times 10^{-8}$，更小就**下溢成 0**。\n2. 所以 1e-8 存成 fp16 是 **0**，这个梯度就丢了。Micikevicius et al. 统计过激活梯度的分布，有相当一部分落在 fp16 能表示的范围以下（arXiv 1710.03740，作者自测）。\n3. bf16 指数 8 位，和 fp32 一样，最小正规数约 $1.2 \\times 10^{-38}$。1e-8 存成 bf16 还是 **约 1e-8**（只有两三位有效数字，但不会变成 0）。\n**loss scaling**：反向之前把 loss 乘一个大数 S，比如 $2^{16}$ = 65536。由链式法则，所有梯度都跟着乘 S：1e-8 × 65536 ≈ 6.6e-4，进了 fp16 的正规范围。更新参数之前在 fp32 里除回 S。动态版本：梯度里出现 inf / NaN 就跳过这一步、S 减半；连续若干步正常就把 S 加倍。\nbf16 范围够大，**不需要 loss scaling**（Kalamkar et al. arXiv 1905.12322：bf16 不改超参就训到和 fp32 相当，作者自测）。bf16 的问题在精度，见下一张卡。',
    ref: TMEM,
  }),
  b({
    id: 'bagu-bf16-fp16-2',
    topic: 'Post-train',
    q: 'bf16 训练为什么还要一份 fp32 主权重？算一下：w = 1.0，这一步的更新量 η·g = 1e-4，直接在 bf16 里算 w − 1e-4 得到什么？',
    a: '关键量是 **ulp**：某个数附近相邻两个可表示数的间隔。尾数有 m 位时，[1, 2) 之间的间隔是 $2^{-m}$，[0.5, 1) 之间是 $2^{-m-1}$。\n1. bf16 尾数 7 位：1.0 往上的下一个数是 $1 + 2^{-7} \\approx 1.0078$，往下是 $1 - 2^{-8} \\approx 0.9961$。\n2. $1.0 - 10^{-4} = 0.9999$，离 1.0 只有 1e-4，离 0.9961 有 0.0038，按最近舍入回到 **1.0**，这一步白更新了。\n3. 每一步都这样：100 步、每步 1e-4，本该挪 0.01，bf16 里的 w 一动不动。\n4. fp32 尾数 23 位，1.0 附近间隔约 $1.2 \\times 10^{-7}$，1e-4 的更新能留下、能累积。\n所以混合精度训练（Micikevicius et al. arXiv 1710.03740）这样分工：\n1. 前向、反向用 bf16 的 w：矩阵乘快、省显存。\n2. 优化器在 **fp32 主权重**上累加更新，更新完再 cast 成 bf16 给下一步用。\n3. Adam 的 m、v 也存 fp32，同样是为了小量能累积（v 是梯度的平方，数值更小）。\n代价是每个参数多 4 B，就是「每参数 16 字节」里的那一项（见训练显存那张卡）。',
    ref: TMEM,
  }),
  b({
    id: 'bagu-lora',
    topic: 'Post-train',
    q: 'LoRA 怎么初始化？它省了哪些显存，没省哪些？',
    a: '公式（见图）：冻结原权重 $W_0 \\in \\mathbb{R}^{d \\times k}$，只训练两个小矩阵 $B \\in \\mathbb{R}^{d \\times r}$、$A \\in \\mathbb{R}^{r \\times k}$，$r \\ll \\min(d, k)$，一般 8–64：\n$$h = W_0 x + \\frac{\\alpha}{r} B A x$$\n可训练参数从 $dk$ 降到 $r(d + k)$。d = k = 4096、r = 16：1678 万 → 13 万，约 0.8%。\n初始化：**A 随机、B 全零**，一开始 BA = 0，模型和原模型完全一样。不能两个都是 0：$\\partial L / \\partial B \\propto (Ax)^\\top$、$\\partial L / \\partial A \\propto B^\\top$，两个都是 0 时梯度也都是 0，永远学不动。\n省了：可训练参数的梯度和优化器状态，缩小上百倍。\n没省：**激活**。算 A 的梯度要用每层的输入 x，激活和全参一样要存。QLoRA 再把冻结的 $W_0$ 压成 4 bit NF4。',
    fig: lines`
                x   (k)
              /    \
           W0        A    r x k   random
         d x k       |
         frozen      B    d x r   zeros
            |        |    * alpha / r
              \    /
                +
                h   (d)
    `,
    ref: LORA,
  }),
  b({
    id: 'bagu-multi-lora',
    topic: 'Post-train',
    q: '一个服务挂 100 个 LoRA adapter（Llama-2-7B，r = 16，7 个线性层都加）。每个 adapter 多大？为什么不把每个都合并进 W？一个 batch 里混着不同 adapter 的请求怎么算？',
    a: 'LoRA 回顾：冻结的 $W_0$（d × k）旁边加 $BA$，B 是 d × r、A 是 r × k，输出 $h = W_0 x + \\frac{\\alpha}{r} BAx$（见 LoRA 那张卡）。每个线性层多 $r(d + k)$ 个参数。\n1. 一个 adapter 多大：每层 q、k、v、o 各 16 × (4096 + 4096)，gate、up 各 16 × (4096 + 11008)，down 16 × (11008 + 4096)，合计约 125 万；32 层约 4000 万参数，bf16 **80 MB**。base 是 13.5 GB。\n2. 为什么不合并：$W_0 + BA$ 对每个 adapter 都是一份新的 13.5 GB 权重，100 个就是 1.35 TB；而且一个 batch 里只能跑同一个 adapter 的请求。\n3. 不合并怎么算：所有请求共用 $W_0$，一个大 GEMM 算 $W_0 x$；每个请求再按自己的 adapter 编号加上 $B_a A_a x$（代码见下）。100 个 adapter 共 8 GB 放得下，更多的放 CPU 按需换入。\n4. 额外 FLOPs 很少：LoRA 部分和 base 之比是 $r(d + k) / (dk)$ = 16 × 8192 / 4096² ≈ 0.8%。难点是一个 batch 里各请求用不同的 A、B，要把「按 adapter 分段的小矩阵乘」做成一个 kernel：Punica 的 SGMV（arXiv 2310.18547）。S-LoRA 再把 adapter 权重和 KV cache 放进同一个分页显存池统一管理（arXiv 2311.03285）。\n只服务一个 adapter 时就直接合并，零额外开销。',
    code: lines`
      # x: [T, k]，batch 里所有请求的 token 拼在一起
      # idx[t]: 第 t 个 token 用哪个 adapter
      y = x @ W0.T                         # 共享的 base，一个大 GEMM
      for a in idx.unique():               # SGMV 把这个循环做成一个 kernel
          rows = idx == a
          y[rows] += (x[rows] @ A[a].T) @ B[a].T * (alpha / r)
    `,
    ref: LORA,
  }),
  b({
    id: 'bagu-ppo-four-models',
    topic: 'Post-train',
    q: '7B 模型做 PPO-RLHF，要同时放哪四个模型？每个做什么、训不训、占多少显存？',
    a: '显存口径：要训练的模型每参数 16 B（bf16 参数 + 梯度 + fp32 主权重、Adam m、v），只做前向的 bf16 模型每参数 2 B。7B 分别是 112 GB 和 14 GB。\n1. **actor / policy**（训，112 GB）：被优化的 LLM，从 SFT 模型初始化，负责生成回答。\n2. **critic / value**（训，112 GB）：通常和 actor 一样大，在每个 token 位置输出一个数 $V(s_t)$，估计「从这里往下平均能拿几分」。用它算 advantage：$A_t$ ≈ 实际得分 − $V(s_t)$，比平均好的 token 推高、差的压低。\n3. **reward model**（冻，14 GB）：在人类偏好对上训好的打分器，给整条回答打一个分 $r(x, y)$。\n4. **reference**（冻，14 GB）：冻结的 SFT 模型，只用来算 KL，防止 policy 跑偏（见下一张卡）。\n合计约 **252 GB**，还不含激活和生成时的 KV cache，所以 PPO-RLHF 天然多卡、多引擎（InstructGPT，arXiv 2203.02155）。\n一个 PPO step：actor 生成回答 → reward model 打分 → reference 和 actor 算每个 token 的 log 概率，得到 KL → critic 算 V，用 GAE 得到 advantage → 用 PPO 的 clip loss 更新 actor、用回归 loss 更新 critic（PPO，arXiv 1707.06347）。\nGRPO 去掉了 critic；DPO 再去掉 reward model 和生成。',
    ref: RLHF,
  }),
  b({
    id: 'bagu-ppo-four-models-2',
    topic: 'Post-train',
    q: 'PPO-RLHF 里的 KL 惩罚为什么要加、加在哪？β 太大、太小各会怎样？',
    a: '符号：$\\pi_{\\theta_{\\text{old}}}$ 是生成这批回答时的 policy（参数快照），$\\pi_{\\text{ref}}$ 是冻结的 SFT 模型，$s_t$ 是 prompt 加前 t − 1 个 token，$y_t$ 是第 t 个 token，T 是回答长度，$r_\\psi(x, y)$ 是 reward model 的分，β > 0。\n为什么要：reward model 只在 SFT 模型的输出附近训过，是人类偏好的不完美代理。policy 跑到它没见过的地方，会找到「打分高但人觉得烂」的回答，比如堆讨好的话、越写越长，这叫 **reward hacking**。KL 把 policy 拴在 reference 附近。\n加在哪：InstructGPT 放进每个 token 的 reward 里（arXiv 2203.02155）：\n$$r_t = -\\beta \\log \\frac{\\pi_{\\theta_{\\text{old}}}(y_t \\mid s_t)}{\\pi_{\\text{ref}}(y_t \\mid s_t)} + \\mathbb{1}[t = T]\\, r_\\psi(x, y)$$\n每个 token 都扣掉「偏离 reference 的程度」，最后一个 token 再加上 reward model 的分，一起交给 critic 和 GAE 算 advantage。\n例：某个 token policy 给 0.5、reference 给 0.05，log 比 = ln 10 ≈ 2.3，β = 0.05 时这个 token 扣 0.115。这样的 token 多了，扣分就抵掉 reward model 给的好处。\n1. β 太大：稍一偏离就扣分，policy 几乎不动，学不到东西。\n2. β 太小：拴不住，reward 分数一路涨，人工评估反而变差。\n和 PPO 的 clip 不重复：clip 只限制相邻两轮之间改多少，多轮累积下来照样能跑远；KL 管的是离固定锚点的总距离。',
    ref: RLHF,
  }),
  b({
    id: 'bagu-dpo',
    topic: 'Post-train',
    q: 'DPO 的 loss 是什么，为什么不需要 reward model？算一个例子：β = 0.1，policy 相对 reference 把好回答的 log 概率提高了 2、把差回答的降低了 1，loss 是多少？',
    a: '符号：x 是 prompt，$y_w$ 是被选中的好回答，$y_l$ 是被拒绝的差回答；$\\pi_\\theta$ 是在训的模型，$\\pi_{\\text{ref}}$ 是冻结的 SFT 模型；$\\log \\pi(y|x)$ 是整条回答每个 token 的 log 概率之和；σ 是 sigmoid；β 是 KL 约束的强度。\n推导三步（DPO，arXiv 2305.18290）：\n1. RLHF 的目标「最大化 reward − β · KL(π ‖ π_ref)」有闭式最优解：$\\pi^*(y|x) \\propto \\pi_{\\text{ref}}(y|x)\\, e^{r(x,y)/\\beta}$。\n2. 反过来把 reward 用策略表示：$r(x,y) = \\beta \\log \\frac{\\pi^*(y|x)}{\\pi_{\\text{ref}}(y|x)} + \\beta \\log Z(x)$。$Z(x)$ 是归一化常数，要对所有可能的回答求和，算不出来，但它只和 x 有关。\n3. 偏好模型（Bradley-Terry）只看两个回答的 reward 差：$P(y_w \\succ y_l) = \\sigma(r_w - r_l)$，同一个 x 的 $\\beta \\log Z(x)$ 正好相减抵消。把 $\\pi^*$ 换成 $\\pi_\\theta$，在偏好数据上做最大似然：\n$$\\mathcal{L} = -\\log \\sigma\\Big(\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\Big)$$\nreward 被「policy 和 reference 的 log 概率比」代替了，所以不用单独训 reward model，也不用生成回答。\n代入例子：括号里 = 0.1 × (2 − (−1)) = 0.3，$\\mathcal{L} = -\\log \\sigma(0.3) \\approx 0.55$。这一对的梯度权重是 $\\sigma(-0.3) \\approx 0.43$：已经排对了一些，权重就小于 0.5；排错时括号为负，权重接近 1。',
    ref: RLHF,
  }),
  b({
    id: 'bagu-dpo-2',
    topic: 'Post-train',
    q: '7B 模型做偏好对齐，DPO 和 PPO 各要放几个模型、多少显存？DPO 省掉了什么，代价是什么？',
    a: '显存口径：要训练的模型每参数 16 B，只做前向的 bf16 模型每参数 2 B（7B 分别是 112 GB、14 GB），不含激活和 KV。\n1. **PPO-RLHF**：actor（训）112 + critic（训，同尺寸）112 + reward model 14 + reference 14 ≈ **252 GB**，外加生成回答时的 KV cache。\n2. **DPO**：policy（训）112 + reference 14 ≈ **126 GB**。reference 的 log 概率只和数据有关，可以提前算好存下来，训练时连它都不用放。训练形态和 SFT 一样：读数据、前向、算 loss、反向。\n省掉的：reward model（reward 隐含在 log 概率比里）、critic（不需要逐 token 的基线）、生成（只在固定数据上算 log 概率）。\n代价是**离线**（off-policy）：偏好对是别的模型事先生成的，训练过程中 policy 变了，数据不跟着变（DPO，arXiv 2305.18290）。\n1. policy 自己会生成、但数据里没有的回答，loss 管不到。常见现象是 $y_w$ 和 $y_l$ 的 log 概率**一起下降**，只是 $y_l$ 降得更多，概率流向了数据之外的回答。\n2. 学不到「自己试、按对错改」，推理类任务的提升有限。缓解办法是迭代 / online DPO：每轮用当前 policy 重新采样、打标签再训，但这又需要生成和打分器了。',
    ref: RLHF,
  }),
  b({
    id: 'bagu-grpo',
    topic: 'Post-train',
    q: 'GRPO 不用 critic 怎么算 advantage？一道数学题采 4 个回答，对错是 [1, 0, 0, 1]，各自的 advantage 是多少？4 个全对呢？',
    a: '先说 advantage：这个回答的 reward 减去一个基线（「平均能拿几分」）。高于平均的回答推高它的 token 概率，低于平均的压低。PPO 用一个和 policy 一样大的 critic 网络学这个基线。\nGRPO（DeepSeekMath，arXiv 2402.03300）：同一个 prompt 用当前 policy 采 G 个回答，reward 是 $r_1, \\dots, r_G$，直接拿组内的统计量当基线：\n$$\\hat A_i = \\frac{r_i - \\text{mean}(r)}{\\text{std}(r)}$$\n第 i 个回答的所有 token 共用这个 $\\hat A_i$。\n代入 [1, 0, 0, 1]：均值 0.5，标准差 0.5（按总体算），advantage = **[+1, −1, −1, +1]**：答对的两条推高，答错的压低。\n全对 [1, 1, 1, 1]：std = 0，advantage 全是 0（实现里分母会加一个小 ε），这一组**没有梯度**，白采了。题太简单或太难都会这样，DAPO 的动态采样就是把这种组过滤掉再补采（arXiv 2503.14476）。\n省了什么：7B 下 critic 要 7 × 10⁹ × 16 B = 112 GB（参数、梯度、Adam 状态）。GRPO 只剩 actor 112 + reference 14 ≈ 126 GB，PPO 约 252 GB。\n适合 reward **可验证**的任务（数学答案对不对、代码能不能过测试）：reward 由规则算，不用 reward model，噪声也小，组均值当基线够用。DeepSeek-R1 就用它做推理 RL（arXiv 2501.12948）。',
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
    q: 'SFT packing 把 3 条样本拼成一条：a a a | b b | c c c。attention mask 和 position_ids 怎么设？不设会怎样？',
    a: 'packing：把多条短样本首尾拼成一条定长序列，免得每条都 pad 到最大长度，GPU 利用率能从三到五成提到九成以上。拼起来以后有两件事要处理。\n1. **attention 不能跨样本**。只用普通的因果 mask（j ≤ i）的话，b 的 token 会看到 a 的内容，训练时多出推理时不存在的上下文（cross-contamination，Krell et al. arXiv 2107.02027）。允许 query i 看 key j 要同时满足两个条件（见图和代码）：\n同一条样本：`seg[i] == seg[j]`，seg 记每个 token 属于第几条；\n因果：`j <= i`。\n前一个条件是 block-diagonal 矩阵，后一个是下三角矩阵，按位与就是最终的 mask。\n2. **位置编码按样本重置**：position_ids = 0 1 2 0 1 0 1 2，每条从 0 开始，和单独推理时一样。\n实际训练不会物化这个 mask：FlashAttention 的 varlen 接口传 `cu_seqlens = [0, 3, 5, 8]`（每条样本在拼接序列里的起止位置），kernel 按段做因果 attention，效果和这个 mask 一样。',
    fig: lines`
      first 5 tokens: a a a b b
      row = query i, col = key j, 1 = allowed

      same seg      j <= i        allowed
      1 1 1 . .     1 . . . .     1 . . . .
      1 1 1 . .     1 1 . . .     1 1 . . .
      1 1 1 . .  &  1 1 1 . .  =  1 1 1 . .
      . . . 1 1     1 1 1 1 .     . . . 1 .
      . . . 1 1     1 1 1 1 1     . . . 1 1
    `,
    code: lines`
      seg = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])   # 每个 token 属于第几条样本
      i = torch.arange(8)[:, None]                   # (8, 1) query 下标
      j = torch.arange(8)[None, :]                   # (1, 8) key 下标
      allowed = (seg[i] == seg[j]) & (j <= i)        # (8, 8)
      scores = scores.masked_fill(~allowed, float('-inf'))

      starts = torch.tensor([0, 3, 5])               # 每条样本的起点
      position_ids = torch.arange(8) - starts[seg]   # 0 1 2 0 1 0 1 2
    `,
    ref: SFT,
  }),
  b({
    id: 'bagu-sft-packing-2',
    topic: 'Post-train',
    q: '一条拼好的 SFT 序列里有 system、user、assistant 三种内容，labels 怎么设才能只在回答上算 loss？两条样本的接缝处要注意什么？',
    a: '先说 label：语言模型在位置 t 预测第 t + 1 个 token。HF 的约定是 labels 和 input_ids 对齐，模型内部错开一位再算 cross entropy；label 为 **−100** 的位置不算 loss（PyTorch `cross_entropy` 的 `ignore_index` 默认就是 −100）。\n1. **只学回答**：system、user 的 token 是给定的条件，不该学着去生成，label 设 −100；assistant 的 token（包括结束符）保留原来的 token id（见图）。多轮对话一次前向就算完所有 assistant 轮的 loss，不用拆成多条样本、重复算前面的对话。\n2. **接缝**：错开一位以后，a 的最后一个 token 会被训练去预测 b 的第一个 token，这是两条无关样本之间的「预测」。所以每条样本**第一个 token 的 label 设 −100**。HF 的 `DataCollatorWithFlattening` 就是这样拼 labels 的：每条样本是 `[-100] + labels[1:]`（https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py ）。\n如果每条样本都以 system / user 开头，第一个 token 本来就是 −100，接缝自然被盖住；不 mask prompt 的数据（比如续写式语料）就必须单独处理。',
    fig: lines`
      tok     label   role
      <sys>   -100    system  (sample 1)
      Hi      -100    user
      Hello   Hello   assistant
      <eos>   <eos>   assistant
      <sys>   -100    system  (sample 2)
      2+2?    -100    user
      4       4       assistant
      <eos>   <eos>   assistant
    `,
    ref: SFT,
  }),

  b({
    id: 'bagu-dapo-clip-higher',
    topic: 'RL',
    q: 'PPO / GRPO 的 clip 让 $\\hat A > 0$ 时新概率最多是旧概率的 $1 + \\epsilon$ 倍。$\\epsilon = 0.2$ 时，旧概率 0.9 和 0.01 的两个 token 一步最多各涨到多少？DAPO 为什么把上界单独放到 0.28？',
    a: '1. 0.9 × 1.2 = 1.08，超过 1，等于不受限。\n2. 0.01 × 1.2 = **0.012**，被卡得很死。\n上界只卡低概率 token，而探索恰恰靠它们：一条做对的回答里采到了一个本来不太可能的 token，本该大幅加强。卡住它们，熵就越来越低（熵塌缩）。\nDAPO 把上下界分开：$\\epsilon_{\\text{high}} = 0.28$，0.01 一步能到 0.0128；下界 $\\epsilon_{\\text{low}}$ 保持 0.2，因为放宽下界会把低概率 token 压到接近 0，探索反而更少。',
    ref: GRPOV,
  }),
  b({
    id: 'bagu-dynamic-sampling',
    topic: 'RL',
    q: 'GRPO 每个 prompt 采 $G = 16$ 条，答对奖励 1、答错 0。模型对某题的正确率 $p = 0.9$，一组 16 条全对的概率是多少？这组对梯度的贡献是多少？',
    a: '1. 全对的概率 $p^{16} = 0.9^{16} \\approx$ **0.19**。\n2. 全对时 16 个奖励相同，advantage $= r_i - \\text{mean} = 0$，这组对梯度的贡献是 **0**，白采了。\n训练越往后，模型做对的题越多，一个 batch 里这样的组越多，有效样本越少。DAPO 的动态采样：多采一些 prompt，把正确率恰好是 0 或 1 的组过滤掉，一直采到 batch 填满。',
    ref: GRPOV,
  }),
  b({
    id: 'bagu-grpo-length-bias',
    topic: 'RL',
    q: 'GRPO 的 loss 先对每条回答按长度平均（除以 $|y_i|$），再对 $G$ 条平均。两条答错的回答，advantage 都是 −1，长度 100 和 1000 token。每个 token 分到的系数各是多少（略去 $1/G$）？长期会让模型怎样？',
    a: '1. 100 token 那条：每个 token **1/100**。\n2. 1000 token 那条：每个 token **1/1000**，只挨十分之一的罚。\n错误回答写得越长，每个 token 受罚越轻，模型的错误回答就越写越长（Dr. GRPO 说的长度偏差）。\n两种改法，都让同一 batch 里每个 token 的系数相同：\n1. DAPO：除以 batch 的总 token 数，这里两条都是 1/1100。\n2. Dr. GRPO：除以一个常数 $L_{\\max}$（生成长度上限）。',
    ref: GRPOV,
  }),
  b({
    id: 'bagu-grpo-std-bias',
    topic: 'RL',
    q: 'GRPO 的 advantage $= (r_i - \\text{mean}) / \\text{std}$，组内 $G = 8$，答对 1 分、答错 0 分，std 用总体标准差。只有 1 条答对时，答对那条的 advantage 是多少？4 条答对时呢？',
    a: '1. 1 条答对：mean = 0.125，std $= \\sqrt{0.125 \\times 0.875} = 0.331$，答对那条 $(1 - 0.125) / 0.331 =$ **+2.65**。\n2. 4 条答对：mean = 0.5，std = 0.5，答对那条 **+1.0**。\nstd 小的组（几乎全对或几乎全错，也就是太难或太简单的题）被除以一个小数，advantage 被放大，在梯度里的权重比中等难度的题大。Dr. GRPO 的改法是只减均值、不除 std：两种情况分别是 +0.875 和 +0.5。',
    ref: GRPOV,
  }),
  b({
    id: 'bagu-gspo-ratio',
    topic: 'RL',
    q: 'GSPO 的序列级 ratio：$s = \\exp\\big(\\frac{1}{|y|}\\sum_t \\log\\rho_t\\big)$，$\\rho_t$ 是第 t 个 token 的新旧概率比。一条 4 个 token 的回答，$\\rho_t = [1.1, 0.9, 1.5, 0.8]$，$s$ 是多少？不开 $1/|y|$ 次方、直接连乘是多少？',
    a: '1. $\\log\\rho_t = [0.095, -0.105, 0.405, -0.223]$，平均 0.043，$s = e^{0.043} =$ **1.044**。\n2. 直接连乘：1.1 × 0.9 × 1.5 × 0.8 = **1.188**。\n开方就是取几何平均：连乘几百上千个 token 会极大或极小，长短回答的数量级不同，没法共用一个 clip 范围；几何平均把它拉回 1 附近。所以 GSPO 的 clip 范围很窄（论文用 $3 \\times 10^{-4}$ / $4 \\times 10^{-4}$）。整条回答共用一个 ratio，单个 token 的波动（包括 MoE 路由变化引起的）被平均掉。',
    ref: GRPOV,
  }),
  b({
    id: 'bagu-cispo',
    topic: 'RL',
    q: '一个 token 的新旧概率比 $\\rho = 2$，$\\hat A > 0$，clip 上界 $1 + \\epsilon = 1.28$。PPO 的 $\\min(\\rho\\hat A, \\operatorname{clip}(\\rho)\\hat A)$ 下，这个 token 对 $\\theta$ 的梯度是多少？CISPO 下呢？',
    a: '1. PPO：$\\rho > 1.28$，min 取到 $1.28\\hat A$，这是个常数，对 $\\theta$ 的梯度是 **0**。clip 不是「截断权重」，而是「这个 token 这一步不再推」。\n2. CISPO：权重截到 1.28 并 stop-gradient，梯度从 $\\log\\pi_\\theta$ 走：$1.28 \\cdot \\hat A \\cdot \\nabla\\log\\pi_\\theta$，**不为 0**。\nMiniMax 的动机：However、Wait 这类低概率的转折 token，更新一次后 ratio 就很大，在 PPO 下被 clip 掉、再也没有梯度；一批 rollout 要做 16 次更新时尤其严重。',
    ref: GRPOV,
  }),
  b({
    id: 'bagu-rollout-tail',
    topic: 'RL',
    q: '同步 RL 一批 8 条回答，长度（千 token）是 $[2, 3, 3, 4, 4, 5, 8, 20]$，设每个 decode 步耗时相同。序列槽位的利用率是多少？最后只剩一条在跑的那段占总时长多少？',
    a: '利用率 = 实际生成的 token ÷（条数 × 最长长度）：\n1. 共 49k token，8 × 20k = 160k，利用率 **31%**。\n2. 第 7 条在 8k 处结束，之后 12k 步只剩一条：12 / 20 = **60%** 的时间。\n实际更糟：只剩几条时 decode 读一遍权重只算一两个 token，每步耗时并不随 batch 等比例缩短；同步 RL 的训练卡这段时间全在等。解法：partial rollout（没写完的挂起下轮接着写）、异步 rollout、长短分开跑。',
    ref: ASYNC,
  }),
  b({
    id: 'bagu-areal-staleness',
    topic: 'RL',
    q: 'AReaL 只在 $\\lfloor (N_r - 1)/B \\rfloor \\le i + \\eta$ 时接受新的生成请求：$N_r$ 是累计发出的请求数，$B$ 是训练 batch，$i$ 是 trainer 当前的版本，$\\eta$ 是最大陈旧度。$B = 512$、$\\eta = 4$、$i = 10$ 时，最多累计能发出多少条？',
    a: '$\\lfloor (N_r - 1)/512 \\rfloor \\le 14$，即 $N_r - 1 \\le 15 \\times 512 - 1$，最多 **7680** 条。\n含义：rollout 最多领先 trainer 4 个 batch，样本最多落后 4 个版本。$\\eta = 0$ 就是同步 RL。\n样本变旧后，AReaL 用 decoupled PPO：importance 比 $\\pi_{\\text{prox}} / \\pi_{\\text{behav}}$ 修正「样本来自旧策略」，clip 只约束相对 $\\pi_{\\text{prox}}$（这一步更新前的参数）走多远。论文消融里 $\\eta = 4$ 不改目标掉到 23.3 分，改了是 42.2。',
    ref: ASYNC,
  }),
  b({
    id: 'bagu-bf16-sum-order',
    topic: 'RL',
    q: 'bf16 有 8 位有效位，[256, 512) 之间相邻两个数差 2。把 $[256, 1, 1, 1, 1]$ 按这个顺序逐个累加，结果是多少？先加四个 1、最后加 256 呢？这和 RL 的训推不一致有什么关系？',
    a: '1. 先 256：每次 256 + 1 都舍入回 256，结果 **256**。\n2. 先加四个 1 得 4，再加 256：**260**。\n浮点加法不满足结合律，结果取决于归约顺序。matmul 会按形状选 split-K，attention 会按 KV 长度切块，这些都随 batch 大小变。推理引擎和训练引擎的 batch 组织、并行方式、kernel 都不同，所以同一份权重算出的 token 概率不一样，「on-policy」其实是 off-policy。Thinking Machines 的解法是让 RMSNorm、matmul、attention 的归约顺序与 batch 无关（batch-invariant）。',
    ref: MISM,
  }),
  b({
    id: 'bagu-tis',
    topic: 'RL',
    q: '训推修正里 $\\rho_t = \\pi_{\\text{learner}}(y_t) / \\pi_{\\text{sampler}}(y_t)$（训练引擎和推理引擎在同一份旧权重下给这个 token 的概率比）。一个 $\\rho_t = 16$ 的 token，不截断时梯度噪声被放大多少倍？TIS 截到 $C = 2$ 呢？',
    a: 'importance 权重 $w$ 把梯度方差放大约 $w^2$ 倍：\n1. 不截断：$16^2 =$ **256** 倍。\n2. TIS（$w = \\min(\\rho_t, C)$）截到 2：$2^2 =$ **4** 倍。\n截断引入偏差，换来方差有界，$C = 2$ 是 verl、slime 的常用默认值。\n注意 $\\rho_t$ 和 PPO 的 ratio 是两回事：PPO 的 ratio 是训练引擎在新旧权重下的比，管「这一步走多远」；$\\rho_t$ 管「样本其实来自推理引擎的分布」。',
    ref: MISM,
  }),
  b({
    id: 'bagu-r3',
    topic: 'RL',
    q: 'MoE 做 RL 时，R3（Rollout Routing Replay）在训练时回放什么、不回放什么？为什么 gate 权重不直接用推理侧的？',
    a: '1. **回放**：推理引擎在 rollout 时每层选中的 top-K 专家（一个 mask）。训练时强制用这几个专家。\n2. **不回放**：gate 权重。用训练侧的 router logit 在这 K 个专家上做 softmax：$g_i = \\frac{I_i\\, e^{s_i}}{\\sum_j I_j\\, e^{s_j}}$，$I$ 是推理侧的 mask，$s$ 是训练侧的 logit。\n这样梯度照常传回 router，router 还能学。\n为什么要回放：top-K 是不连续的，两边 logit 差一点就可能换专家。R3 论文统计 Qwen3-30B-A3B 上 94% 的 token 至少有一层选的专家不同；回放后训推 KL 从 $1.5 \\times 10^{-3}$ 降到 $7.5 \\times 10^{-4}$，rollout 慢不到 3%（作者自测）。',
    ref: MISM,
  }),
  b({
    id: 'bagu-weight-sync-time',
    topic: 'RL',
    q: 'RL 每步训练完要把权重同步给推理引擎。70B bf16，走一张 400 Gb/s 的网卡，带宽下界是多少秒？一个节点 8 张网卡并行呢？',
    a: '下界 = 字节数 ÷ 带宽：\n1. 70 × 10⁹ × 2 B = 140 GB。\n2. 400 Gb/s = 50 GB/s，140 / 50 = **2.8 s**。\n3. 8 张网卡约 400 GB/s：**0.35 s**。\n实测通常是下界的几倍：训练侧要 all-gather 出完整张量、改名字和布局（QKV、gate/up 融合）、可能量化成 fp8，每个张量单独发有调用开销，推理侧还要加载。常用的优化是分桶（几百 MB 一块）和流水（gather、传输、加载重叠）。',
    ref: SYNC,
  }),
  b({
    id: 'bagu-megatron-qkv',
    topic: 'RL',
    q: '8 个 Q head、2 个 KV head（GQA，每组 4 个 Q head）。Megatron 的 `linear_qkv` 按 head 是什么顺序？HF / vLLM 要的是什么顺序？',
    a: 'Megatron 按 GQA 组交错，每组是「4 个 q、1 个 k、1 个 v」：\n`q0 q1 q2 q3 k0 v0 | q4 q5 q6 q7 k1 v1`\nHF 是 q、k、v 三个矩阵分开，vLLM 的 `qkv_proj` 是三者按顺序拼：\n`q0 … q7 | k0 k1 | v0 v1`\n权重同步时要转换：先 view 成（组数，每组 q 数 + 2，head_dim，hidden），沿第 1 维切成 [4, 1, 1] 三份，再分别拼回去。不转换的话推理侧加载不报错，但 attention 全错。',
    ref: SYNC,
  }),

  b({
    id: 'bagu-tito',
    topic: 'RL',
    q: '多轮 agent 的 RL：调 chat 接口拿到文本，最后把整段对话用 chat template 重新分词再拿去训练。这样做哪三处会让训练看到的 token 和采样时不一样？为什么 importance sampling 修不了？',
    a: '1. **分词不唯一**：模型采样出 `H` + `AVING`，重新分词变成 `HAV` + `ING`，文本相同、token 不同。\n2. **工具调用被解析再渲染**：空格变了，有的解析器还会自动修正模型写错的 JSON。\n3. **chat template 改历史**：Qwen3 的模板会删掉之前轮次的思考内容，rollout 时的上下文和训练时拼出来的不同。\nimportance sampling（如 TIS）要逐个 token 比较两边的概率，前提是两边是同一串 token；重新分词后序列都对不齐，没法比。\n做法：推理服务返回 token id（vLLM 的 `return_token_ids`，SGLang 直接收发 `input_ids`），下一轮直接拼 token，不重新套模板。',
    ref: AGENT,
  }),
  b({
    id: 'bagu-agentic-mask',
    topic: 'RL',
    q: '一条 3 轮的工具调用轨迹：模型分别生成 300、200、400 个 token，中间两次工具返回 1500 和 800 个 token。回答部分有多少 token 算 loss，占多少？为什么工具返回的不算？',
    a: '算 loss 的只有模型生成的：300 + 200 + 400 = **900** 个；回答部分共 900 + 2300 = 3200 个，占 **28%**。\n工具返回的 token 只当上下文（`loss_mask = 0`）：\n1. 它不是策略采出来的，没有策略概率，算 ratio、advantage 都没有意义。\n2. 训练模型去预测工具输出，等于教它编造工具结果。',
    ref: AGENT,
  }),
  b({
    id: 'bagu-agentic-prefix',
    topic: 'RL',
    q: '多轮 rollout，每轮在上下文末尾新增 2000 个 token（工具结果 + 指令），共 10 轮。每轮都从头 prefill，总共 prefill 多少 token？前缀 KV 都命中呢？',
    a: '1. 从头 prefill：第 k 轮的输入有 2000k 个 token，$2000 \\times (1 + 2 + \\dots + 10) =$ **110,000**。\n2. 前缀命中：每轮只算新增的 2000 个，共 **20,000**，少 5.5 倍。\n前提是同一条轨迹的后续轮次落在同一个推理实例上：sticky session（按 request id 绑实例）或前缀感知的路由（SGLang router 的 `cache_aware`）。每次权重同步后这些 KV 要作废。',
    ref: AGENT,
  }),
  b({
    id: 'bagu-agentic-concurrency',
    topic: 'RL',
    q: '推理服务想保持 $B = 256$ 条序列同时在 decode。agent 轨迹有 2/3 的时间在等环境（跑代码、搜索），只有 1/3 的时间在生成。要多少条轨迹同时在跑？',
    a: '同时在生成的轨迹数 ≈ 轨迹总数 × 生成时间占比，所以\n$$N_{\\text{traj}} \\approx \\frac{B}{f} = \\frac{256}{1/3} =$$ **768** 条。\n环境越慢，$f$ 越小，要的并发轨迹越多，沙箱规模也要跟上（Kimi K2 用 Kubernetes 跑了一万多个并发沙箱）。配合环境级异步（不按批等环境）和冗余 rollout（凑够就停）。',
    ref: AGENT,
  }),

  // ---------------- 多模态 ----------------
  b({
    id: 'bagu-image-tokens',
    topic: '多模态',
    q: 'Qwen2.5-VL：ViT 的 patch 是 14 × 14，之后相邻 2 × 2 个 patch 合成一个 token；图的长宽各自缩放到最近的 28 的倍数。一张 1920 × 1080 的图变成多少个 token？',
    a: '每个 token 对应 28 × 28 像素：\n1. 1920 / 28 = 68.6，取最近的倍数 69 → 1932。\n2. 1080 / 28 = 38.6 → 39 → 1092。\n3. 69 × 39 = **2691** 个，前后再加两个特殊 token。\n总像素有上下限（默认 4 到 16384 个 token），超出会先按比例缩放。对比 LLaVA-1.5 固定缩到 336 × 336，不管原图多大都是 $(336/14)^2 = 576$ 个。',
    ref: MMENC,
  }),
  b({
    id: 'bagu-video-tokens',
    topic: '多模态',
    q: 'Qwen2-VL 处理视频：每秒采 2 帧，相邻两帧合成一组，每组的 token 数和同分辨率的一张图一样。360p（640 × 360）每组 299 个 token，一分钟视频多少 token？',
    a: '1. 一分钟 60 × 2 = 120 帧，两帧一组，60 组。\n2. 60 × 299 = **17,940** 个 token。\n2 fps、两帧一组时，每秒视频约等于一帧的 token 数。所以视频请求的 prefill 很长，各模型都设了上限（Qwen2-VL 每段视频最多 16384 个 token），帧多了只能降分辨率或降帧率。',
    ref: MMENC,
  }),
  b({
    id: 'bagu-audio-tokens',
    topic: '多模态',
    q: '16 kHz 音频每 10 ms 算一帧 mel 频谱；Whisper 编码器开头的卷积 stride 2；Qwen2-Audio / Qwen2.5-Omni 再接一个 stride 2 的池化。每秒音频变成多少 token？一分钟呢？',
    a: '1. mel：每 10 ms 一帧，**100 帧/秒**。\n2. 卷积 stride 2：**50 帧/秒**（Whisper 30 s 正好 1500 帧）。\n3. 池化 stride 2：**25 token/s**，每个 token 40 ms。\n一分钟 60 × 25 = **1500** 个 token。Qwen3-Omni 的 AuT 编码器下采样 8 倍，12.5 token/s，一分钟 750 个，prefill 和 KV 都减半。',
    ref: MMENC,
  }),
  b({
    id: 'bagu-vit-flops',
    topic: '多模态',
    q: 'Qwen2.5-VL 的 ViT 主体约 632M 参数，一张 1920 × 1080 的图合并前有 10764 个 patch。只算线性层，ViT 编码这张图要多少 FLOP？7.6B 的 LLM 对合并后的 2691 个 token 做 prefill 要多少？',
    a: '线性层每个 patch 每个参数一次乘加，2 FLOP：\n1. ViT：$2 \\times 632\\text{M} \\times 10764 \\approx$ **13.6 TFLOP**（加上 4 层全局 attention 约 2.4 TFLOP，合计约 16）。\n2. LLM prefill：$2 \\times 7.6\\text{B} \\times 2691 \\approx$ **41 TFLOP**。\n编码器约占四成。原因：ViT 看的是合并前的 patch，数量是 LLM token 的 4 倍；而且 ViT 大小固定（3B、7B、72B 共用），换成 72B 的 LLM，编码器只占约 4%。',
    ref: MMENC,
  }),
  b({
    id: 'bagu-mrope',
    topic: '多模态',
    q: 'Qwen2-VL 的 M-RoPE 给每个 token 三个位置 ID（时间、高、宽）。序列是 3 个文本 token，接一张合并后 2 × 2 的图（4 个 token），再接 1 个文本 token。写出所有 token 的三个 ID。',
    a: '规则：文本三个 ID 相同；图像的时间 ID 不变，高、宽 ID 按网格的行、列；图像之后的文本从「前面所有 ID 的最大值 + 1」接着编。\n1. 文本：(0,0,0)、(1,1,1)、(2,2,2)。\n2. 图像从 3 开始：(3,3,3)、(3,3,4)、(3,4,3)、(3,4,4)。\n3. 最大 ID 是 4，下一个文本：(5,5,5)。\n图像占 4 个 token，位置 ID 只往前走了 2。所以多模态序列的位置 ID 比 token 数小，但 KV 长度仍按 token 数算。',
    ref: MMENC,
  }),
  b({
    id: 'bagu-encoder-no-chunk',
    topic: '多模态',
    q: '长文本 prompt 可以 chunked prefill，分几步算。一张图的编码为什么不能切开分步算？vLLM 的调度器怎么处理？',
    a: 'LLM 是因果 attention，前面的 token 不依赖后面的，所以能切；ViT 是**双向 attention**，一张图的所有 patch 互相看，必须一起算。\nvLLM 的做法：\n1. 每步有一个编码器预算（等于 `max_num_batched_tokens`，并保证至少放得下最大的一张图）。\n2. 这一步要处理到某张图的占位 token 时，预算够就整张编码；不够，这一步只调度到这张图之前的文本为止。\n3. 编码结果按图像内容的 hash 缓存，后面几步 prefill 和其他请求都能复用。',
    ref: MMENC,
  }),
  b({
    id: 'bagu-codec-bitrate',
    topic: '多模态',
    q: '语音 codec 的码率 = 帧率 × 码本数 × $\\log_2$(码本大小)。Mimi（Moshi 用的 codec）是 12.5 Hz、8 个码本、每个码本 2048 个码字，码率是多少？',
    a: '$12.5 \\times 8 \\times \\log_2 2048 = 12.5 \\times 8 \\times 11 =$ **1100 bps**，即 1.1 kbps。\n对比 EnCodec（24 kHz）：75 Hz、每个码本 1024 个码字（10 bit），一个码本就是 750 bps，6 kbps 要 8 个码本。帧率低的 codec 每秒要生成的步数少，对 LLM 更友好。',
    ref: SPEECH,
  }),
  b({
    id: 'bagu-rvq',
    topic: '多模态',
    q: '残差向量量化（RVQ）：每一级在码本里找离「当前残差」最近的码字，再减掉它。$x = 0.83$，三级码本依次是 $\\{-1, 0, 1\\}$、$\\{-0.3, 0, 0.3\\}$、$\\{-0.1, 0, 0.1\\}$。每级选哪个？最后还原值和残差是多少？',
    a: '1. 第 1 级逼近 0.83：选 **1.0**，残差 −0.17。\n2. 第 2 级逼近 −0.17：−0.3 离它 0.13、0 离它 0.17，选 **−0.3**，累计 0.7，残差 0.13。\n3. 第 3 级逼近 0.13：选 **0.1**，累计 **0.8**，残差 **0.03**。\n每多一级误差小一截：前几级定大轮廓（内容、音色），后几级补细节。还原时把各级选中的码字加起来。',
    ref: SPEECH,
  }),
  b({
    id: 'bagu-codec-steps',
    topic: '多模态',
    q: 'Qwen3-Omni 的 codec 是 12.5 Hz、16 个码本。如果把每帧的 16 个 token 拍平，让 Talker 一个个自回归生成，每秒音频要走多少步？Qwen3-Omni 实际怎么做？',
    a: '拍平：12.5 × 16 = **200 步/秒**，每步都过一遍 Talker，太慢。\nQwen3-Omni 分两层：\n1. Talker（3B 总参数、0.3B 激活的 MoE）每帧只走一步，预测第 0 个码本：**12.5 步/秒**。\n2. MTP 模块（80M 的 dense transformer）接着把剩下 15 个码本补齐，每帧走 15 小步。\n大模型的步数降到 1/16，小模型步数多但每步便宜。Moshi 的 depth transformer 是同一个思路。',
    ref: SPEECH,
  }),
  b({
    id: 'bagu-omni-rtf',
    topic: '多模态',
    q: 'RTF = 生成耗时 ÷ 音频时长。Qwen3-Omni 生成一帧（80 ms 音频）要走：Thinker 一步、Talker 一步、MTP 补齐一帧、解码器解一帧。并发 1 时 Thinker 75 token/s、Talker 140 token/s、MTP 14 ms/帧、解码 3 ms/帧。RTF 是多少？',
    a: '1. Thinker 一步：1000 / 75 = 13.3 ms。\n2. Talker 一步：1000 / 140 = 7.1 ms。\n3. 加上 MTP 14 ms、解码 3 ms：共 37.5 ms。\n4. RTF = 37.5 / 80 = **0.47**。\nRTF < 1 才不卡：每 80 ms 的音频要在 80 ms 内生成出来。0.47 意味着生成比播放快一倍，余量应该换成更高的并发（论文里并发 6 时 RTF 0.66）。',
    ref: SPEECH,
  }),
  b({
    id: 'bagu-omni-first-packet',
    topic: '多模态',
    q: '语音对话的首包延迟（用户说完到听到第一段声音）由哪几段串行组成？用 Qwen3-Omni 并发 1、音频输入的数字算：预处理和编码 72 ms，Thinker 首 token 88 ms，Talker 首 token 57 ms，MTP 一帧 14 ms，解码一帧 3 ms。',
    a: '五段依次相加：预处理和编码器 → Thinker 出第一个 token → Talker 出第一帧 → MTP 补齐这一帧 → 解码器出第一段波形。\n72 + 88 + 57 + 14 + 3 = **234 ms**。\n并发升到 6 时首包变成 1172 ms，涨得最多的是 Thinker 首 token（88 → 673 ms），也就是 prefill 在排队。所以并发上限常常由首包延迟决定，而不是 RTF。压首包的办法：Talker 拿到一段文本就开始（流式接力），解码器改成纯因果、逐帧出声。',
    ref: SPEECH,
  }),

  b({
    id: 'bagu-barge-in-truncate',
    topic: '多模态',
    q: '语音助手的 RTF = 0.47（生成 1 秒音频要 0.47 秒），用户在播放到 1.5 秒时插话。这时大约已经生成了多少秒音频（忽略首包）？这条回答的文字共 60 个 token、音频共 6 秒，历史里应该保留多少个 token？',
    a: '1. 已生成：1.5 / 0.47 ≈ **3.2 秒**，比播放的多一倍多。\n2. 按播放比例截文字：60 × 1.5 / 6 = **15** 个 token。\n对话历史要截到**用户实际听到的位置**，不是生成到的位置，否则下一轮模型以为用户听过后面那半段。OpenAI Realtime API 用 `conversation.item.truncate` 的 `audio_end_ms` 做这件事，会同时删掉没播出的文字。完整流程：VAD 检测到开口 → 停止播放 → `response.cancel` → 截断历史。',
    ref: DUPLEX,
  }),
  b({
    id: 'bagu-duplex-tick',
    topic: '多模态',
    q: 'Moshi 是全双工模型，每 80 ms 一步，每步都要吃进用户的音频、吐出自己的一帧（可能是静音）。$B$ 个会话一起跑一个 batch 步耗时 $t(B)$，实时的条件是什么？Moshi 上下文 4096 步，够聊多久？',
    a: '1. 实时条件：$t(B) \\le 80$ ms，满足它的最大 $B$ 就是并发上限，再留出余量。超了播放会断、用户音频会积压，不像文本 serving 那样只是慢一点。\n2. 上下文：4096 ÷ 12.5 步/秒 ≈ **328 秒**，约 5.5 分钟。\n全双工和按轮次的区别：模型一直在跑，没人说话时也占算力，什么时候说、什么时候停由模型自己决定。',
    ref: DUPLEX,
  }),
  b({
    id: 'bagu-rtf-direction',
    topic: '多模态',
    q: 'Qwen3-Omni 报告 RTF 0.47，Kyutai 的 DSM-ASR 报告 batch 64 时「RTF 3.5」。两者的 RTF 定义一样吗？各自代表比实时快几倍？DSM-ASR 这时相当于多少路实时流？',
    a: '定义相反：\n1. Qwen：RTF = 生成耗时 ÷ 音频时长，越小越好，0.47 表示比实时快约 **2.1 倍**，必须 < 1。\n2. Kyutai：写的是音频时长 ÷ 耗时，越大越好，3.5 表示快 **3.5 倍**，必须 > 1。\n3. DSM-ASR：3.5 × 64 = **224** 路实时流的处理量（作者报告一张 H100 能实时处理约 400 路）。\n读 RTF 先看定义，换成同一个方向再比较。',
    ref: DUPLEX,
  }),

  // ---------------- 系统设计 ----------------
  b({
    id: 'bagu-sd-order',
    topic: '系统设计',
    q: '面试题：设计一个 70B 聊天模型的推理服务，峰值 100 QPS，平均输入 2k、输出 500 token，TTFT P99 < 1 s、TPOT < 50 ms。前几分钟应该算出哪几个数？它们怎么决定架构？',
    a: '顺序：先把 SLA 和流量问清楚（这里题目已给），然后**先算容量**，架构从数字里推出来。以 H100 80 GiB、bf16 为例：\n1. **显存定 TP**：权重 132 GiB，单卡放不下。TP = 4 每卡 33 GiB；按 90% 显存可用、再留 3 GiB 给激活，每卡剩约 36 GiB 给 KV。KV 每 token 320 KiB，4 卡共约 **45 万 token**；每个请求最多约 2.5k token，单副本能同时放约 180 个请求。\n2. **roofline 定单副本吞吐**：decode 一步每卡读 33 GiB ÷ 3.35 TB/s ≈ 10 ms，加上读 KV 和 all-reduce，实际约 20 ms。batch 64 时 64 ÷ 0.02 s = **3200 token/s**。\n3. **副本数**：100 QPS × 500 = 5 万 token/s ÷ 3200 ≈ **16 个副本、64 张卡**。\n4. **核对 prefill**：100 QPS × 2000 token × 2 × 70 × 10⁹ = **28 PFLOP/s**；64 卡 × 989 TFLOP/s × 50% MFU ≈ 31.6 PFLOP/s。prefill 几乎吃满算力，这是最关键的发现。\n由数字推架构：\n1. prefill 是瓶颈 → prefix caching（复用 system prompt 和多轮历史），考虑 PD 分离给 prefill 单独配卡。\n2. TPOT 要求 → chunked prefill，避免长 prompt 的 prefill 卡住正在 decode 的请求。\n3. 多副本 → 网关 + 按前缀亲和路由，命中 prefix cache；按排队长度扩缩容。\n4. 最后是可观测性（TTFT / TPOT 分位数、KV 使用率）和过载降级（准入控制，直接返回 429）。',
    ref: SD,
  }),
  b({
    id: 'bagu-sd-capacity',
    topic: '系统设计',
    q: '估单个副本的 decode 吞吐：TPOT 20 ms、batch 64，每秒生成多少 token？要支撑每秒 5 万 token 需要几个副本？',
    a: '吞吐 = batch ÷ TPOT = 64 ÷ 0.02 s = **3200 token/s**。\n5 万 ÷ 3200 ≈ **16 个副本**。每个副本 TP = 4 的话就是 64 张卡。\n这只算了生成侧。prefill 要单独按 FLOPs 估：输入长的时候，常常是 prefill 先把算力吃满，这时要上 prefix caching 或 PD 分离。',
    ref: SD,
  }),
  b({
    id: 'bagu-sd-routing',
    topic: '系统设计',
    q: '多副本之间怎么路由才能命中 prefix cache？路由器怎么知道哪个副本有哪段 KV？',
    a: '**不要轮询**：同一前缀的请求被打散到各个副本，每个副本都要重新 prefill。\n1. 需要一层路由（可以多实例）。它的状态是「软」的：猜错只是 cache miss，不影响正确性。\n2. 每个副本本来就维护着自己的 prefix cache（block hash 表或 radix tree），那是它自己复用用的。\n3. 路由器怎么知道，有两种做法：\n近似：路由器不问副本。按前缀（比如 system prompt 或 session id）做一致性哈希，同前缀总去同一个副本；或者像 SGLang router 那样，按自己路由过的请求，给每个副本维护一棵近似的 radix tree。\n精确：副本把 KV block 的存入、淘汰事件上报（vLLM 能发这类 KV events），路由器维护全局索引，挑前缀匹配最长的副本，比如 llm-d。\n4. 都要按负载加权，防止热门前缀把一个副本打爆。\n效果：prefix 命中率能从接近 0 提到七成以上，直接砍掉大部分 prefill。过载时的准入控制见 TTFT 那张卡。',
    ref: SD,
  }),
]
