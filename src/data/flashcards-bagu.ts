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
const DIT = '/inference/dit-serving'
const MON = '/posttrain/rl-monitoring'

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

  b({
    id: 'bagu-entropy-cov',
    topic: 'RL',
    q: '一个位置只有两个候选 token，概率 $(0.9, 0.1)$，熵 $H = -\\sum p\\log p$ 是多少？如果高概率的那个被奖励、推到 0.95，熵变成多少？如果低概率的那个被奖励、推到 0.15 呢？',
    a: '1. 初始：$-(0.9\\ln 0.9 + 0.1\\ln 0.1) =$ **0.325**。\n2. 推高高概率的到 0.95：**0.199**，熵降。\n3. 推高低概率的到 0.15：**0.423**，熵升。\n一般规律：一步策略梯度后 $\\Delta H \\approx -\\eta\\,\\text{Cov}(\\log\\pi, \\pi A)$，高概率动作拿正 advantage 让熵降。训练中被奖励的多半是模型本来就倾向的回答，所以熵整体下降，而且前期掉得很快（Cui et al. 统计前 1/12 的步数用掉 73% 的熵）。对策：clip-higher、Clip-Cov / KL-Cov。',
    ref: MON,
  }),
  b({
    id: 'bagu-ppo-kl-vs-kl-loss',
    topic: 'RL',
    q: 'verl 的面板上有 `actor/ppo_kl` 和 `actor/kl_loss` 两条 KL 曲线。它们分别是谁对谁的 KL？各说明什么？',
    a: '1. `ppo_kl`：当前策略对 **rollout 时的策略**（$\\pi_{\\text{old}}$）的 KL。一批数据只更新一次时接近 0，更新多次逐渐变大。管的是单步步长，配合 `pg_clipfrac`（clip 生效的 token 比例）看每步走了多远。\n2. `kl_loss`：当前策略对 **reference**（SFT 模型）的 KL，开了 `use_kl_loss` 才有。管的是累计离初始模型多远。\nDAPO、Dr. GRPO 等推理 RL 配方直接不加 kl_loss。训推不一致是第三种 KL（`rollout_corr/kl`），是推理引擎对训练引擎的。',
    ref: MON,
  }),
  b({
    id: 'bagu-ess',
    topic: 'RL',
    q: 'importance 权重的有效样本比例 ESS：先把权重归一化到均值 1，再算 $1/\\mathbb{E}[w^2]$。4 个 token 的权重是 $[1, 1, 1, 4]$，ESS 是多少？权重全是 1 呢？',
    a: '1. 均值 1.75，归一化后 $[0.57, 0.57, 0.57, 2.29]$。\n2. $\\mathbb{E}[w^2] = (3 \\times 0.327 + 5.22) / 4 = 1.55$，ESS $= 1/1.55 =$ **0.65**。\n3. 全是 1：$\\mathbb{E}[w^2] = 1$，ESS **= 1**。\nESS 越小，梯度越被少数几个大权重的样本主导，方差越大。verl 文档给的参考告警线是 ESS < 0.3（另外还有训推 KL 的绝对值 > 0.1、chi2 > 1），文档说明只是参考。',
    ref: MON,
  }),
  b({
    id: 'bagu-collapse-order',
    topic: 'RL',
    q: 'RL 训练里由训推不一致导致的崩溃，面板上的几条曲线通常按什么顺序出问题？为什么不能直接跳过训推 KL 大的 batch？',
    a: '顺序（Liu, Li et al. 的观察）：\n1. 训推 KL（推理引擎对训练引擎）先冒尖。\n2. 熵的尖峰几乎同时出现；训练侧的困惑度跳起来。\n3. 梯度范数突然爆炸。\n4. 奖励崩溃。\n排查时从奖励往前倒着看。\n不能直接跳过：他们试过训推 KL > 0.1 的 batch 不更新，结果模型越来越多地产出高差距的 batch，训练停滞。要用截断 / 屏蔽的 importance 权重修，或从源头消除差距（换掉有问题的 kernel、fp16、MoE 用路由回放）。',
    ref: MON,
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

  b({
    id: 'bagu-dit-tokens',
    topic: '多模态',
    q: 'Wan2.1 生成视频：VAE 时间上压 4 倍（第一帧单独）、空间上压 8 倍，之后再把 $2 \\times 2$ 的 latent 合成一个 token。token 数 $L = (1 + \\frac{T-1}{4}) \\cdot \\frac{H}{16} \\cdot \\frac{W}{16}$。1280 × 720、81 帧（16 fps 约 5 秒）是多少 token？',
    a: '1. 时间：$1 + 80/4 = 21$。\n2. 高：$720/16 = 45$；宽：$1280/16 = 80$。\n3. $21 \\times 45 \\times 80 =$ **75,600** 个 token。\n相当于对 7.5 万个 token 做一次没有因果 mask 的 prefill，而且每一步去噪都要重做一遍。832 × 480 是 $21 \\times 30 \\times 52 = 32{,}760$。Wan2.2 的 5B 模型换了空间压 32 倍（含 patchify）的 VAE，token 数约为 1/4。',
    ref: DIT,
  }),
  b({
    id: 'bagu-dit-flops',
    topic: '多模态',
    q: 'Wan2.1-14B 一次前向：每个 token 碰到的权重约 12B，hidden $d = 5120$，40 层，序列 $L = 75{,}600$。线性层按 $2PL$、self-attention 按 $4L^2d$ 每层算，一次前向多少 FLOP？50 步、每步 CFG 两次前向，一共多少？',
    a: '1. 线性层：$2 \\times 12\\text{B} \\times 75{,}600 \\approx$ **1.8 PFLOP**。\n2. attention：$4 \\times 75{,}600^2 \\times 5120 \\times 40 \\approx$ **4.7 PFLOP**，占七成。\n3. 一次前向约 6.5 PFLOP；× 50 步 × 2 ≈ **0.65 EFLOP**。\nH100 bf16 峰值 989 TFLOP/s，按 50% 利用率约 22 分钟。视频的 $L^2$ 项压倒线性层，所以加速重点在 attention：稀疏（滑动 tile）、量化（SageAttention）、序列并行。',
    ref: DIT,
  }),
  b({
    id: 'bagu-dit-no-kv',
    topic: '多模态',
    q: 'LLM decode 靠 KV cache 省计算。DiT 去噪为什么没有 KV cache？那它能复用什么？',
    a: '1. LLM 能缓存 KV，是因为前面 token 的 K、V 算完就不变。DiT 每一步去噪都在更新**整段** latent，所有 token 的输入都变了，K、V 每步都不同，没有可以精确复用的。所以每步都是一次完整的大前向，全程 compute-bound。\n2. 能复用的是「相邻两步的输出很接近」。TeaCache 用被时间步调制后的输入变化量估计输出变化，累计变化小于阈值就直接复用上一步的输出、跳过这一步（作者自测 Open-Sora-Plan 上 4.4 倍）。这是近似：跳多了误差会累积，出现细节丢失。\n另一条路是少走几步：步数蒸馏（LCM、FLUX.1-schnell 用 1–4 步）。',
    ref: DIT,
  }),
  b({
    id: 'bagu-ulysses-ring',
    topic: '多模态',
    q: '视频 DiT 的序列太长，要沿序列维切到多张卡上。Ulysses 和 Ring 两种序列并行各怎么做？Wan2.1-14B 有 40 个 head，Ulysses 最多切几张卡？',
    a: '1. **Ulysses**：每张卡先持有一段序列；attention 前用 all-to-all 换成「每张卡拿一部分 head、看完整序列」，算完再 all-to-all 换回来。并行度不能超过 head 数，Wan2.1-14B **最多 40**。\n2. **Ring**：每张卡用自己的 Q，K、V 沿环在卡之间传，边收边按块算，用 online softmax 合并结果。没有 head 数限制，通信是点对点。\n3. **USP**：两者组成二维网格。Wan2.1 外层 Ring、内层 Ulysses，作者报告 256K token 时通信开销降到 1% 以下。\n另外每步的有条件 / 无条件两次前向可以放到两张卡上（CFG 并行）。',
    ref: DIT,
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

  // ================ 训练系统 ================

  // ---------------- 训练算力账 ----------------
  b({
    id: 'bagu-compute-mfu-6n-linear',
    topic: '训练算力',
    q: '一个线性层 $Y = XW$，$X$ 是 $[T, k]$，$W$ 是 $[k, m]$。取 $T = 4096$、$k = m = 8192$，训练时 forward 加 backward 一共多少 FLOPs？由此说明为什么每 token 训练是 $6N$。',
    a: '符号：$T$ token 数，$k, m$ 是权重的两维，$km$ 就是这一层的参数个数。\n1. forward $Y = XW$：每个权重对每个 token 做 1 次乘 + 1 次加，$2Tkm$。\n2. backward 对激活 $\\partial X = \\partial Y\\,W^\\top$：梯度要往前一层传，$2Tkm$。\n3. backward 对权重 $\\partial W = X^\\top \\partial Y$：更新这一层的参数，$2Tkm$。\n合计 $6Tkm = 6 \\times 4096 \\times 8192^2 \\approx 1.65\\times10^{12}$ FLOPs。\n每个参数、每个 token 是 6 FLOPs（forward 2、backward 4），把所有参与矩阵乘的权重加起来就是每 token $6N$，训练总量 $C \\approx 6ND$（$D$ 是 token 数）。',
    ref: '/train/compute-mfu',
  }),
  b({
    id: 'bagu-compute-mfu-6nd',
    topic: '训练算力',
    q: '70B 参数的 dense 模型训 15T token，训练总 FLOPs 是多少？（1 B = $10^9$，1 T = $10^{12}$）',
    a: '公式 $C \\approx 6ND$：$N$ 参数量，$D$ 训练 token 数，6 是每参数每 token 的 forward 2 + backward 4 FLOPs。\n$C = 6 \\times 7\\times10^{10} \\times 1.5\\times10^{13} = 6.3\\times10^{24}$ FLOPs。\n这里没算 attention 的 $QK^\\top$、$PV$（激活乘激活，不在 $6N$ 里），序列不长时可以忽略，比值另见 attention 项那张卡。',
    ref: '/train/compute-mfu',
  }),
  b({
    id: 'bagu-compute-mfu-attn-term',
    topic: '训练算力',
    q: '每 token 训练 FLOPs 写成 $6N + 12Lhs$。Llama 3 70B：$L = 80$，$h = 8192$，$N = 70$ B。序列 $s = 32768$ 时，attention 项占 $6N$ 的多少？',
    a: '符号：$L$ 层数，$h$ hidden size（所有 head 维度之和），$s$ 序列长度。\n$12Lhs$ 怎么来：一层里一个 token 的 query 和 $s$ 个 key 点积 $2sh$，再对 $s$ 个 value 加权求和 $2sh$，forward $4sh$，backward 是两倍，合计 $12sh$，乘 $L$ 层（PaLM 附录 B，按不带 causal mask 的满额算）。\n比值 $\\dfrac{12Lhs}{6N} = \\dfrac{2Lhs}{N} = \\dfrac{2 \\times 80 \\times 8192 \\times 32768}{7\\times10^{10}} \\approx 61\\%$。\n标准结构 $N \\approx 12Lh^2$ 时比值化简成 $s/(6h)$：$s$ 到 $6h$ 时 attention 项和 $6N$ 一样大。长上下文阶段一定要算上。',
    ref: '/train/compute-mfu',
  }),
  b({
    id: 'bagu-compute-mfu-from-tps',
    topic: 'MFU',
    q: '8B 模型在 H100 上训练，实测每卡 8000 token/s。H100 bf16 dense 峰值 989 TFLOP/s，忽略 attention 项，MFU 是多少？',
    a: '定义（PaLM）：MFU = 按模型需要的 FLOPs 算出的吞吐 ÷ 峰值。\n$$\\text{MFU} = \\frac{T_{\\text{gpu}} \\cdot 6N}{\\text{peak}}$$\n$T_{\\text{gpu}}$ 每卡 token/s，$6N$ 每 token 训练 FLOPs，peak 单卡峰值。\n代入：$8000 \\times 6 \\times 8\\times10^9 / 9.89\\times10^{14} = 3.84\\times10^{14} / 9.89\\times10^{14} \\approx 38.8\\%$。\npeak 取训练精度下的 dense 峰值；规格表上 1979 TFLOPS 是带 2:4 稀疏的数，不能用。',
    ref: '/train/compute-mfu',
  }),
  b({
    id: 'bagu-compute-mfu-hfu',
    topic: 'MFU',
    q: '开了 full recompute，算出 MFU = 40%。HFU 是多少？为什么 PaLM 主张报 MFU 而不是 HFU？',
    a: '1. MFU 的分子只算模型需要的 FLOPs：每 token $6N$（forward 2N + backward 4N），recompute 不算。\n2. HFU 的分子是硬件实际执行的 FLOPs。full recompute 在 backward 前多跑一次 forward，每 token 实际做 $8N$。\n3. 同样的 token/s，$\\text{HFU} = \\frac{8}{6}\\text{MFU} = 1.33 \\times 40\\% \\approx 53\\%$。\n为什么报 MFU：recompute 是拿算力换显存，目标是 token/s，不是多做 FLOPs。HFU 会因为多重算而变好看；MFU 只取决于 token/s、模型结构和峰值，不同系统能直接比。\n推论：full recompute 下就算硬件 100% 忙，MFU 上限也只有 75%。',
    ref: '/train/compute-mfu',
  }),
  b({
    id: 'bagu-compute-mfu-days',
    topic: '训练算力',
    q: '70B 训 15T token，用 4096 张 H100（bf16 dense 峰值 989 TFLOP/s），MFU 40%，ETTR 0.9。要训多少天？',
    a: '$$t = \\frac{6ND}{n_{\\text{gpu}} \\cdot \\text{peak} \\cdot \\text{MFU} \\cdot \\text{ETTR}}$$\n符号：$N$ 参数量，$D$ token 数，$n_{\\text{gpu}}$ 卡数；MFU 是稳态时的算力利用率；ETTR 是墙钟时间里真正在推进训练的比例（扣掉故障重算、重启、写 checkpoint）。\n1. 分子：$6 \\times 7\\times10^{10} \\times 1.5\\times10^{13} = 6.3\\times10^{24}$。\n2. 分母：$4096 \\times 9.89\\times10^{14} \\times 0.4 \\times 0.9 = 1.46\\times10^{18}$ FLOP/s。\n3. $t = 4.32\\times10^6$ s ≈ **50 天**。',
    ref: '/train/compute-mfu',
  }),
  b({
    id: 'bagu-compute-mfu-chinchilla',
    topic: 'scaling law',
    q: '训练预算 $C = 10^{24}$ FLOPs。按 Chinchilla 的「每参数约 20 个 token」，最优的参数量 $N$ 和 token 数 $D$ 各是多少？',
    a: 'Chinchilla（Hoffmann et al. 2022）拟合出 $N_{\\text{opt}} \\propto C^{0.5}$、$D_{\\text{opt}} \\propto C^{0.5}$：参数和数据同比例放大，所以 $D/N$ 是常数，约 20。\n把 $D = 20N$ 代入 $C = 6ND$：\n$$C = 120N^2 \\Rightarrow N_{\\text{opt}} = \\sqrt{C/120}$$\n1. $N_{\\text{opt}} = \\sqrt{10^{24}/120} = \\sqrt{8.33\\times10^{21}} \\approx 91$ B。\n2. $D_{\\text{opt}} = 20 \\times 91\\text{B} \\approx 1.83$ T token。\n20 不是普适常数，换数据和结构要重新拟合；它也只最小化训练算力，不管推理成本。',
    ref: '/train/compute-mfu',
  }),
  b({
    id: 'bagu-compute-mfu-overtrain',
    topic: 'scaling law',
    q: 'Llama 3 8B 训了约 15T token。它的 token / 参数比是多少，是 Chinchilla 最优（20 token / 参数）的几倍？为什么要这样 over-train？',
    a: '1. 比值：$15\\times10^{12} / 8\\times10^9 = 1875$ token / 参数。\n2. Chinchilla 最优是 $20 \\times 8\\text{B} = 160$ B token，15T 是它的约 94 倍。\n为什么：Chinchilla 只最小化**训练**算力。上线后每生成一个 token 还要约 $2N$ FLOPs，一生的总成本是\n$$6ND + 2N D_{\\text{infer}}$$\n$D_{\\text{infer}}$ 是要服务的 token 总数，往往远大于 $D$。小模型多训些 token，训练多花一点，每次推理都省。Llama 3 论文的说法是小模型训得远超算力最优，在同样推理预算下比算力最优的模型更好。',
    ref: '/train/compute-mfu',
  }),
  b({
    id: 'bagu-compute-mfu-gpu-hours',
    topic: 'MFU',
    q: '一次训练 $N = 405$ B、$D = 15.6$ T token，报告用了 $3.084\\times10^7$ GPU 小时（H100，bf16 dense 峰值 989 TFLOP/s）。反推平均的 MFU × ETTR 是多少？',
    a: '符号：$N$ 参数量，$D$ token 数；MFU 是稳态时按 $6N$ 算的算力利用率，ETTR 是墙钟时间里真正推进训练的比例（扣掉故障重算、重启、写 checkpoint）。\n记 GPU 小时数为 $H$，$H = \\dfrac{6ND}{\\text{peak} \\cdot \\text{MFU} \\cdot \\text{ETTR}} / 3600$，所以\n$$\\text{MFU} \\cdot \\text{ETTR} = \\frac{6ND}{3600\\,H \\cdot \\text{peak}}$$\n1. 分子：$6 \\times 4.05\\times10^{11} \\times 1.56\\times10^{13} = 3.79\\times10^{25}$ FLOPs。\n2. 分母：$3.084\\times10^7 \\times 3600 \\times 9.89\\times10^{14} = 1.10\\times10^{26}$。\n3. 约 **34.5%**。\n这是 Llama 3 405B 的公开数字：论文报告稳态 MFU 38–43%、ETTR >90%，乘起来约 37%。反推值更低，因为长上下文阶段 MFU 只有 38% 且 attention FLOPs 没进 $6ND$，前期 batch 小也更慢。',
    ref: '/train/compute-mfu',
  }),

  // ---------------- 训练 step：AdamW / LR / 裁剪 ----------------
  b({
    id: 'bagu-optimizer-lr-bias-correction',
    topic: 'Adam 偏差修正',
    q: 'Adam 第 1 步：梯度 $g_1 = 0.5$，$m_0 = v_0 = 0$，$\\beta_1 = 0.9$，$\\beta_2 = 0.999$，忽略 $\\epsilon$。更新式 $m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$，$v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2$。不做偏差修正时 $m_1/\\sqrt{v_1}$ 是多少？用 $\\hat m_t = m_t/(1-\\beta_1^t)$、$\\hat v_t = v_t/(1-\\beta_2^t)$ 修正后呢？',
    a: '符号：$m$ 是梯度的滑动平均（一阶动量），$v$ 是梯度平方的滑动平均（二阶动量），$t$ 是 step 序号。Adam 每步走 $\\eta\\,\\hat m/\\sqrt{\\hat v}$，正常情况下这个比值约为 1。\n1. 不修正：$m_1 = 0.1 \\times 0.5 = 0.05$，$v_1 = 0.001 \\times 0.25 = 0.00025$，$\\sqrt{v_1} = 0.0158$，比值 $= 3.16$，第一步走了正常步长的 3 倍多。\n2. 修正：$\\hat m_1 = 0.05/0.1 = 0.5$，$\\hat v_1 = 0.00025/0.001 = 0.25$，比值 $= 0.5/0.5 = 1$。\n原因：$m$、$v$ 从 0 开始，前几步被 0 拉低，而且被低估的倍数不同（$1-\\beta_1^t$ 对 $1-\\beta_2^t$），比值就偏了。$t$ 大了 $\\beta^t \\to 0$，修正项趋于 1。',
    ref: '/train/optimizer-lr',
  }),
  b({
    id: 'bagu-optimizer-lr-beta2',
    topic: 'Adam β2',
    q: 'Adam 单步更新量的上界（Kingma & Ba 2014 §2.1）：当 $1-\\beta_1 > \\sqrt{1-\\beta_2}$ 时 $\\lvert\\Delta\\theta\\rvert \\le \\eta\\,(1-\\beta_1)/\\sqrt{1-\\beta_2}$，否则 $\\lvert\\Delta\\theta\\rvert \\le \\eta$（$\\eta$ 是学习率）。$\\beta_1 = 0.9$ 时，分别算 $\\beta_2 = 0.999$ 和 $\\beta_2 = 0.95$ 的上界。',
    a: '$\\beta_1$、$\\beta_2$ 是一阶、二阶动量的衰减系数。\n1. $\\beta_2 = 0.999$：$\\sqrt{0.001} = 0.032 < 0.1$，落在第一种情况，上界 $\\eta \\times 0.1/0.032 = 3.16\\eta$。\n2. $\\beta_2 = 0.95$：$\\sqrt{0.05} = 0.224 > 0.1$，上界就是 $\\eta$。\n第一种上界在「前面梯度一直是 0、这一步突然有梯度」时取到：$v$ 还停在旧尺度，$\\sqrt v$ 偏小，更新偏大。LLM 里低频 token 的 embedding 行、spike 前后梯度尺度突变都是这种情况。0.95 的 $v$ 记忆约 $1/(1-\\beta_2) = 20$ 步，0.999 约 1000 步。GPT-3、Llama 2 都用 $\\beta_2 = 0.95$。',
    ref: '/train/optimizer-lr',
  }),
  b({
    id: 'bagu-optimizer-lr-adamw-decay',
    topic: 'AdamW vs L2',
    q: 'Adam 下两个参数都是 $\\theta = 1$，weight decay 系数 $\\lambda = 0.1$，学习率 $\\eta$。参数 A 历史梯度大，$\\sqrt{\\hat v} = 10$；参数 B 历史梯度小，$\\sqrt{\\hat v} = 0.01$。L2 正则（把 $\\lambda\\theta$ 加进梯度，之后被 $\\sqrt{\\hat v}$ 除）和 AdamW（直接 $\\theta \\leftarrow \\theta - \\eta\\lambda\\theta$）下，每步衰减量各是多少？（衰减项远小于梯度时近似计算）',
    a: '$\\hat v$ 是偏差修正后的二阶动量，$\\sqrt{\\hat v}$ 是 Adam 给每个参数的尺度。\n1. L2：衰减项随梯度一起被除以 $\\sqrt{\\hat v}$，约为 $\\eta\\lambda\\theta/\\sqrt{\\hat v}$。A：$0.1\\eta/10 = 0.01\\eta$；B：$0.1\\eta/0.01 = 10\\eta$。差 1000 倍，梯度大的参数几乎不被正则。\n2. AdamW：衰减不进梯度，A、B 都是 $\\eta\\lambda\\theta = 0.1\\eta$。\n这是 Loshchilov & Hutter（arXiv 1711.05101）的论点：SGD 下 L2 和 weight decay 等价，Adam 下不等价。PyTorch 的 `Adam(weight_decay=...)` 是 L2，`AdamW` 才是 decoupled。',
    ref: '/train/optimizer-lr',
  }),
  b({
    id: 'bagu-optimizer-lr-accum-ddp',
    topic: 'Gradient accumulation',
    q: 'DDP 共 $W = 2$ 个 rank，all-reduce 对梯度求平均（除以 $W$）。这一步 rank 0 有 400 个计入 loss 的 token、loss 之和 650；rank 1 有 600 个、loss 之和 1350。正确的按 token 平均的 loss 是多少？每个 rank 应该反传什么，DDP 平均后才等于它？如果每个 rank 先按本地 token 数平均，得到多少？',
    a: '设 $N$ 为这一步所有 rank 计入 loss 的 token 总数，$\\ell_i$ 为单个 token 的 loss。\n1. 正确：$N = 1000$，$(650 + 1350)/1000 = 2.0$。\n2. 每个 rank 反传 $\\frac{W}{N}\\sum_{\\text{本 rank}} \\ell_i$：rank 0 是 $650 \\times 2/1000 = 1.3$，rank 1 是 $2.7$，DDP 平均后 $(1.3 + 2.7)/2 = 2.0$。乘 $W$ 是为了抵消 DDP 的除以 $W$。\n3. 按本地 token 数平均：rank 0 $650/400 = 1.625$，rank 1 $1350/600 = 2.25$，平均 $1.94$。token 少的 rank 被放大了权重。\n实现上要在 forward 前先统计 token 数并 all-reduce 求和得到 $N$。梯度和 loss 一起缩放，所以这里的 loss 权重就是每个 token 梯度的权重。',
    ref: '/train/optimizer-lr',
  }),
  b({
    id: 'bagu-optimizer-lr-clip-sharded',
    topic: '梯度裁剪',
    q: '梯度裁剪：$g \\leftarrow g\\cdot\\min(1,\\ c/\\lVert g\\rVert)$，$\\lVert g\\rVert$ 是所有参数梯度拼成一个向量的 L2 norm，阈值 $c = 1.0$。FSDP 下 2 个 rank 各持有一半梯度，本地梯度平方和分别是 9 和 16。全局 norm 和缩放系数是多少？',
    a: '1. 先跨 rank 对**平方和**求和再开根：$\\sqrt{9 + 16} = 5$。不是 $\\sqrt 9 + \\sqrt{16} = 7$，也不是平均。\n2. 缩放系数 $\\min(1, 1/5) = 0.2$，所有 rank 的所有梯度都乘 0.2。\n3. 实现：每个 rank 算本地平方和，`all_reduce(SUM)`，再开根。\n为什么不能各 rank 用本地 norm：缩放系数会不同（$1/3$ 和 $1/4$），等于把梯度向量的方向改了，而且各 rank 结果不一致。同理，TP 下每个 rank 都有一份的参数（如 LayerNorm 权重）只能计一次；纯 DDP 在 all-reduce 后各 rank 梯度相同，本地算就是全局的。阈值 1.0 是经验值（GPT-3、Llama 2、PaLM），目的是只截异常步。',
    ref: '/train/optimizer-lr',
  }),
  b({
    id: 'bagu-optimizer-lr-warmup-cosine',
    topic: 'LR 调度',
    q: 'Llama 3 405B 的调度：峰值 $\\eta_{\\max} = 8\\times10^{-5}$，线性 warmup $T_w = 8000$ 步（$\\eta_t = \\eta_{\\max}\\,t/T_w$），之后 cosine 降到 $\\eta_{\\min} = 8\\times10^{-7}$：$\\eta = \\eta_{\\min} + \\frac{1}{2}(\\eta_{\\max}-\\eta_{\\min})(1+\\cos \\pi p)$，$p$ 是 cosine 段的进度（0 到 1）。算 $t = 2000$ 时和 $p = 0.5$ 时的 lr。',
    a: '1. $t = 2000$ 在 warmup 段：$8\\times10^{-5} \\times 2000/8000 = 2\\times10^{-5}$。\n2. $p = 0.5$：$\\cos(\\pi/2) = 0$，$\\eta = 8\\times10^{-7} + \\frac{1}{2}(8\\times10^{-5} - 8\\times10^{-7}) = 8\\times10^{-7} + 3.96\\times10^{-5} = 4.04\\times10^{-5}$。\n3. 终点 $p = 1$：$\\cos\\pi = -1$，$\\eta = \\eta_{\\min}$。\nwarmup 的原因：前几步 $v$ 只平均了很少几个梯度，$1/\\sqrt{\\hat v}$ 方差很大（RAdam，arXiv 1908.03265），先用小 lr 等 $v$ 稳定。cosine 的缺点是形状绑定总步数，开训前就要定。',
    ref: '/train/optimizer-lr',
  }),
  b({
    id: 'bagu-optimizer-lr-critical-batch',
    topic: 'Critical batch size',
    q: 'McCandlish et al. 2018：固定 batch $B$（按 token 计）时，达到某个 loss 需要的步数 $S = S_{\\min}(1 + B_{\\text{noise}}/B)$，token 数 $E = E_{\\min}(1 + B/B_{\\text{noise}})$。设 $B_{\\text{noise}} = 1$M token，$S_{\\min} = 10^4$ 步，$E_{\\min} = 10^{10}$ token。$B = 1$M 和 $B = 4$M 时 $S$、$E$ 各是多少？',
    a: '符号：$S_{\\min}$ 是 batch 无穷大时的最少步数，$E_{\\min}$ 是 batch 无穷小时的最少数据量；$B_{\\text{noise}} \\approx \\operatorname{tr}(\\Sigma)/\\lVert G\\rVert^2$，梯度噪声方差和梯度信号相当时的 batch。\n1. $B = 1$M：$S = 2\\times10^4$，$E = 2\\times10^{10}$，步数和数据量都是最优值的 2 倍，是时间和算力的折中点，即 critical batch size $B_{\\text{crit}} = E_{\\min}/S_{\\min}$。\n2. $B = 4$M：$S = 10^4 \\times 1.25 = 1.25\\times10^4$，$E = 10^{10} \\times 5 = 5\\times10^{10}$。\n加大 4 倍 batch，步数只少 37.5%，算力多花 2.5 倍。$B_{\\text{noise}}$ 随训练增大（梯度变小），所以 GPT-3、Llama 3 都先用小 batch 再加大。',
    ref: '/train/optimizer-lr',
  }),
  b({
    id: 'bagu-optimizer-lr-wsd',
    topic: 'WSD 调度',
    q: 'WSD（warmup-stable-decay）调度，计划 100k 步：warmup 2k 步，2k–90k 保持峰值 $3\\times10^{-4}$，90k–100k 线性降到 0。95k 步的 lr 是多少？训完想延长到 200k 步，从哪个 checkpoint 续、lr 怎么走？',
    a: '1. 95k 在 decay 段正中间：$3\\times10^{-4} \\times (100 - 95)/10 = 1.5\\times10^{-4}$。\n2. 续训：从 **90k 的 stable checkpoint** 接着用峰值 $3\\times10^{-4}$ 训到 180k，再 decay 2 万步（约总量的 10%）。90k–100k 那段 decay 只是一个分支，可以扔掉。\n3. cosine 不行的原因：cosine 的形状由总步数决定，中途的 checkpoint lr 已经降了一部分，续训要重新拉高 lr。\nMiniCPM（arXiv 2404.06395）的观察：decay 段 loss 快速下降，decay 占总 token 10% 就够、2.5% 不够（作者自测）。它也因此能从一条主线分出多个 decay 分支，用线性代价测不同数据量下的 loss。',
    ref: '/train/optimizer-lr',
  }),
  b({
    id: 'bagu-optimizer-lr-muon-rms',
    topic: 'Muon',
    q: 'Muon 把隐藏层矩阵的动量 $M$ 用 Newton–Schulz 迭代近似正交化成 $O = UV^\\top$（$M = U\\Sigma V^\\top$，非零奇异值都变成 1）。形状 $[A, B]$、$A \\le B$ 时 $O$ 有 $A$ 个奇异值 1，$\\lVert O\\rVert_F^2 = A$。算 $4096\\times11008$ 时 $O$ 的 RMS（元素平方均值开根）。Moonlight 把更新写成 $0.2\\sqrt{\\max(A,B)}\\,O$，更新的 RMS 是多少？',
    a: '1. RMS $= \\sqrt{\\lVert O\\rVert_F^2/(AB)} = \\sqrt{A/(AB)} = 1/\\sqrt{B} = 1/\\sqrt{11008} \\approx 0.0095$。一般地是 $1/\\sqrt{\\max(A,B)}$，矩阵越大更新越小。\n2. 乘 $0.2\\sqrt{11008}$ 后 RMS $= 0.2$。\n为什么：Moonlight（arXiv 2502.16982）观察到 AdamW 的更新 RMS 通常在 0.2–0.4，把 Muon 对齐到这个量级后，不同形状的矩阵步长一致，而且能直接复用 AdamW 调好的 lr 和 weight decay。embedding、输出 head、norm 的 scale 仍用 AdamW。Moonlight 报告 Muon 达到同样 loss 约需 52% 的训练 FLOPs（作者自测）。',
    ref: '/train/optimizer-lr',
  }),

  // ---------------- Pipeline 调度 ----------------
  b({
    id: 'bagu-pp-gpipe-bubble',
    topic: 'Pipeline 调度',
    q: 'GPipe 调度，$p = 8$ 个 stage，一个 global batch 切成 $m = 32$ 个 micro-batch。每个 micro-batch 在一个 stage 上 forward + backward 共耗时 $T$。bubble 占总时间多少？相对理想时间（$mT$）又多花了多少？',
    a: '1. 每个 stage 开头等 forward 灌满 $(p-1)t_f$、结尾等 backward 排空 $(p-1)(t_b+t_w)$，合计空等 $(p-1)T$；总时间 $(m+p-1)T$。\n2. 占总时间：$\\frac{p-1}{m+p-1} = \\frac{7}{39} =$ **17.9%**。\n3. 相对理想：$\\frac{p-1}{m} = \\frac{7}{32} =$ **21.9%**（Megatron 论文的定义）。\n两者关系：记 $x = \\frac{p-1}{m}$，则 $\\frac{p-1}{m+p-1} = \\frac{x}{1+x}$。降 bubble 只能加大 $m$。',
    ref: '/train/pipeline-schedules',
  }),
  b({
    id: 'bagu-pp-1f1b-memory',
    topic: 'Pipeline 调度',
    q: '$p = 8$、$m = 64$，一个 micro-batch 在一个 stage 上的激活是 $M_a = 3$ GB。GPipe 和 1F1B 下 stage 0 的激活峰值各是多少？',
    a: '1. GPipe 先做完全部 forward 再做 backward，每个 stage 同时存 $m$ 份：$64 \\times 3 = $ **192 GB**。\n2. 1F1B（PipeDream-Flush）：stage $i$ 先做 $p-1-i$ 个 forward 预热，之后一个 forward 一个 backward 交替，在途（做了 forward、还没做 backward）最多 $p-i$ 个。stage 0 最多 $p = 8$ 个：$8 \\times 3 =$ **24 GB**。\n两者 bubble 完全一样，都是 $(p-1)T$；1F1B 省的是显存，所以 $m$ 可以放心调大。',
    ref: '/train/pipeline-schedules',
  }),
  b({
    id: 'bagu-pp-first-stage',
    topic: 'Pipeline 调度',
    q: '1F1B 调度下，为什么第一个 PP stage 最容易 OOM？给出 stage $i$（从 0 数，共 $p$ 个）的在途 micro-batch 数。',
    a: 'stage $i$ 预热 $p-1-i$ 个 forward，稳态再加 1 个，在途最多 $p - i$ 个：\n1. stage 0 在途 $p$ 个，最后一个 stage 只有 1 个。\n2. embedding 表也在 stage 0。\n缓解：给第一个 stage 少分层（Megatron `--decoder-first-pipeline-num-layers`）；只给前几个 stage 开 activation checkpointing；减小 micro-batch 大小 $b$。',
    ref: '/train/pipeline-schedules',
  }),
  b({
    id: 'bagu-pp-interleaved',
    topic: 'Pipeline 调度',
    q: 'Interleaved 1F1B：$p = 8$、$m = 32$，每卡 $v = 4$ 个 model chunk。bubble（相对理想时间）是多少？一个 micro-batch 在 forward 里跨几次 stage 边界？不交错时呢？',
    a: '1. 每个 chunk 的计算是原来的 $1/v$，灌满和排空也缩成 $1/v$：$\\frac{1}{v} \\cdot \\frac{p-1}{m} = \\frac{7}{128} \\approx$ **5.5%**（不交错是 21.9%）。\n2. 一个 micro-batch 要绕流水线 $v$ 圈，跨 $pv - 1 =$ **31** 次边界；不交错是 $p - 1 = 7$ 次。通信约 × $v$。\n论文的调度还要求 $m$ 是 $p$ 的整数倍。',
    ref: '/train/pipeline-schedules',
  }),
  b({
    id: 'bagu-pp-zero-bubble',
    topic: 'Pipeline 调度',
    q: 'Zero Bubble 把 backward 拆成 B（对输入）和 W（对权重）。设 $t_f = t_b = t_w = 1$ ms，$p = 8$。按论文 Table 2 的公式，1F1B、ZB-H1、ZB-H2 的 bubble 时间各是多少？',
    a: '公式（$t_f$ forward，$t_b$ 对输入的梯度，$t_w$ 对权重的梯度）：\n1. 1F1B：$(p-1)(t_f+t_b+t_w) = 7 \\times 3 =$ **21 ms**。\n2. ZB-H1：$(p-1)(t_f+t_b-t_w) = 7 \\times 1 =$ **7 ms**，1/3，峰值激活和 1F1B 一样（$pM_B$）。\n3. ZB-H2：$(p-1)(t_f+t_b-2t_w) =$ **0**，激活约 $(2p-1)M_B$，还要去掉优化器 step 的跨 stage 同步（梯度裁剪、NaN 检查改成事后校验、出错回滚）。\nW 能挪是因为 $\\partial W$ 只给本 stage 的优化器用，没有别的 stage 在等它；B 的 $\\partial X$ 上一个 stage 要等。',
    ref: '/train/pipeline-schedules',
  }),
  b({
    id: 'bagu-pp-dualpipe',
    topic: 'Pipeline 调度',
    q: 'DeepSeek-V3 Table 2：1F1B bubble $(p-1)(F+B)$，ZB1P $(p-1)(F+B-2W)$，DualPipe $(\\frac{p}{2}-1)(F\\&B+B-3W)$。取 $p = 8$，$F = 1$，$B = 2$，$W = 1$，且通信完全藏住时 $F\\&B \\approx F + B$。三者各是多少？DualPipe 的代价是什么？',
    a: '符号：$F$ 一个 forward chunk，$B$ 一个完整 backward chunk，$W$ 对权重的 backward，$F\\&B$ 一对互相重叠的 forward 和 backward chunk。\n1. 1F1B：$7 \\times 3 =$ **21**。\n2. ZB1P：$7 \\times (1+2-2) =$ **7**。\n3. DualPipe：$3 \\times (3+2-3) =$ **6**。\n代价：每个 rank 放两个 stage（第 $i$ 和第 $p-1-i$ 个），**参数存两份**；激活从 $p$ 变成 $p+1$。要求 stage 数和 micro-batch 数能被 2 整除。真正的收益在通信：DeepSeek-V3 跨节点 EP 的计算 : 通信约 1 : 1，配对后 all-to-all 藏在另一个 chunk 的计算后面。',
    ref: '/train/pipeline-schedules',
  }),
  b({
    id: 'bagu-pp-p2p-bytes',
    topic: 'Pipeline 调度',
    q: 'PP 相邻 stage 之间，每个 micro-batch forward 传一份激活。70B 模型：micro-batch $b = 1$，$s = 8192$，$h = 8192$，bf16。一次传多少字节？IB 每卡 50 GB/s 要多久？',
    a: '1. 激活形状 $[b, s, h]$，$bsh$ 个元素，bf16 每个 2 字节：$2 \\times 1 \\times 8192 \\times 8192 \\approx$ **134 MB**。\n2. $134\\text{ MB} / 50\\text{ GB/s} \\approx$ **2.7 ms**。\n对比：一个 stage 10 层，每层约 8.56 亿参数，forward $2 \\times 10 \\times 8.56\\times10^8 \\times 8192 \\approx 1.4\\times10^{14}$ FLOP，按 494 TFLOP/s（H100 bf16 989 的 50%）约 284 ms。通信不到 1%，还能和别的 micro-batch 重叠。PP 的问题是 bubble，不是通信。',
    ref: '/train/pipeline-schedules',
  }),
  b({
    id: 'bagu-pp-lm-head',
    topic: 'Pipeline 调度',
    q: '70B 模型 80 层、$h = 8192$、词表 $V = 128{,}000$，每层约 8.56 亿参数。PP $p = 8$ 平均切层时，最后一个 stage 的计算比其他 stage 多多少？怎么调？',
    a: '1. LM head 是一个 $V \\times h$ 的线性层：$128000 \\times 8192 \\approx 1.05$B 参数，计算约等于 $1.05 / 0.856 \\approx 1.2$ 层。\n2. 平均每 stage 10 层，最后一个是 $10 + 1.2 = 11.2$ 层的活（还有 loss），流水线按最慢的 stage 走，约慢 **12%**。\n调法：给最后一个 stage 少分层（Megatron `--decoder-last-pipeline-num-layers`）。Llama 3 405B 从首尾 stage 各拿掉一层，让第一个 chunk 只有 embedding、最后一个 chunk 只有输出投影和 loss。',
    ref: '/train/pipeline-schedules',
  }),

  // ---------------- 并行配置 ----------------
  b({
    id: 'bagu-parallel-config-states',
    topic: '并行配置',
    q: '70B dense 模型，TP $t = 8$、PP $p = 2$、DP $d = 64$，DP 上开 ZeRO-1（bf16 参数和梯度、fp32 Adam）。每张卡的模型状态多少 GB？（GB = $10^9$ 字节）',
    a: '每卡分到 $N/(tp)$ 个参数，每个参数的字节：\n1. bf16 参数 2 + bf16 梯度 2，每卡存完整的一份（对这个 TP×PP 分片而言）。\n2. fp32 主权重、Adam $m$、$v$ 共 12 字节，ZeRO-1 在 $d$ 张卡上切开，每卡 $12/d$。\n$$M_{\\text{states}} = \\frac{N}{tp}\\left(4 + \\frac{12}{d}\\right)$$\n代入：$7\\times10^{10}/16 = 4.375\\times10^9$ 个参数，$\\times (4 + 12/64) = 4.1875$ 字节，约 **18.3 GB**。\nPP 和 ZeRO 切的都是这一项，激活要另算。',
    ref: '/train/parallel-config',
  }),
  b({
    id: 'bagu-parallel-config-pp-act',
    topic: '并行配置',
    q: '1F1B 调度、PP $p = 4$，模型 $L = 80$ 层。每层每个 micro-batch 的激活是 0.285 GB。第一个 stage 要存多少激活？$p$ 改成 8 呢？',
    a: '1F1B 要让流水线跑满，第一个 stage 先做 $p$ 个 micro-batch 的 forward，才轮到第一个 backward，所以同时存着 $p$ 份激活；每份只有它自己那 $L/p$ 层。\n$$p \\times \\frac{L}{p} = L \\text{ 层}$$\n1. $p = 4$：$80 \\times 0.285 = 22.8$ GB。\n2. $p = 8$：还是 22.8 GB，和 $p$ 无关（Korthikanti et al. 2022 第 4.2.3 节）。\ninterleaved（每卡 $v$ 段）再乘 $1 + \\frac{p-1}{pv}$。\n结论：PP 只省模型状态，激活要靠 TP、CP、recompute 或减小 micro-batch。',
    ref: '/train/parallel-config',
  }),
  b({
    id: 'bagu-parallel-config-batch',
    topic: '并行配置',
    q: 'global batch $B = 512$ 条序列，DP $d = 32$，micro-batch $b = 1$，PP $p = 4$，interleaved 每卡 $v = 5$ 段。gradient accumulation 几步？bubble 占多少？',
    a: '符号：$B$ 每次更新的序列数，$d$ DP 度，$b$ 一次 forward 的序列数，$k$ gradient accumulation 步数，也就是每步每条 pipeline 跑的 micro-batch 个数 $m$。\n1. $B = d \\cdot b \\cdot k \\Rightarrow k = m = 512 / (32 \\times 1) = 16$。\n2. bubble $= \\dfrac{p-1}{v\\,m} = \\dfrac{3}{5 \\times 16} = 3.75\\%$；不 interleave（$v = 1$）是 $3/16 = 18.8\\%$。\n三者互相掐：卡数翻倍（$d = 64$）而 $B$ 不变，$m$ 减半，bubble 翻倍；$b$ 加大也一样让 $m$ 变小。',
    ref: '/train/parallel-config',
  }),
  b({
    id: 'bagu-parallel-config-mesh',
    topic: '并行配置',
    q: '1024 张卡，每节点 8 张，按 $\\text{rank} = (i_{\\text{dp}} \\cdot p + i_{\\text{pp}}) \\cdot t + i_{\\text{tp}}$ 排（$t = 8$，$p = 2$，$d = 64$）。rank 21 的 TP 组、PP 组各是哪些 rank？为什么 TP 要放最内层？',
    a: '1. 拆 rank 21：$21 = 2 \\times 8 + 5$，$i_{\\text{tp}} = 5$；$2 = 1 \\times 2 + 0$，$i_{\\text{dp}} = 1$、$i_{\\text{pp}} = 0$。\n2. TP 组：只变 $i_{\\text{tp}}$，rank 16–23，正好是同一台机器的 8 张卡。\n3. PP 组：只变 $i_{\\text{pp}}$，rank 21 和 29，在相邻两台机器上。\n4. DP 组：只变 $i_{\\text{dp}}$，rank 5, 21, 37, ...，跨 64 台机器。\n为什么：相邻 rank 在同一台机器上，最内层的维度就落在 NVLink 域内。TP 每层都有集合通信，而且在关键路径上，必须走 NVLink；DP 每步只同步一次、能和 backward 重叠，放最外层跨节点。Llama 3 的顺序就是 [TP, CP, PP, DP] 从内到外。',
    ref: '/train/parallel-config',
  }),
  b({
    id: 'bagu-parallel-config-zero1-pp',
    topic: '并行配置',
    q: '每卡的参数分片是 $4.375\\times10^9$ 个 bf16 参数，一步跑 $k = 8$ 个 micro-batch。DP 用 ZeRO-3 时，每步要 all-gather 多少参数字节？ZeRO-1 呢？',
    a: '1. ZeRO-3 只存 $1/d$ 的参数，每个 micro-batch 的 forward 和 backward 前各要 all-gather 一次完整分片：$2k$ 次。每次 $4.375\\times10^9 \\times 2$ 字节 $= 8.75$ GB，一步 $16 \\times 8.75 = 140$ GB。\n2. ZeRO-1 每卡存完整分片，只在一步结束时 reduce-scatter 梯度、all-gather 更新后的参数各一次：约 $8.75 + 8.75 = 17.5$ GB。\n所以有 PP 和 gradient accumulation 时 DP 配 ZeRO-1：PP 已经把参数切小了，存完整的一份不贵，通信少 $k$ 倍。',
    ref: '/train/parallel-config',
  }),
  b({
    id: 'bagu-parallel-config-dp-comm',
    topic: '并行配置',
    q: 'ZeRO-1 下每卡每步要 reduce-scatter 梯度、all-gather 参数，各约 8.6 GB；跨节点每卡 50 GB/s。一步计算是每卡 $1.72\\times10^{15}$ FLOPs，H100 bf16 峰值 989 TFLOP/s、MFU 40%。DP 通信占计算的多少？',
    a: '1. 通信：$(8.6 + 8.6)\\ \\text{GB} / 50\\ \\text{GB/s} \\approx 0.34$ s。\n2. 计算：$1.72\\times10^{15} / (9.89\\times10^{14} \\times 0.4) \\approx 4.35$ s。\n3. 比例约 8%。\n而且梯度的 reduce-scatter 可以在 backward 时逐层发出，和还没算完的层重叠（Megatron 的 `--overlap-grad-reduce`），参数 all-gather 可以和下一步的 forward 重叠（`--overlap-param-gather`）。所以 DP 可以放心跨节点。',
    ref: '/train/parallel-config',
  }),
  b({
    id: 'bagu-parallel-config-tp-cap',
    topic: '并行配置',
    q: 'Llama 3 70B 有 64 个 query head、8 个 KV head，H100 每节点 8 卡，NVLink 单向约 450 GB/s，跨节点每卡约 50 GB/s。TP 最大取多少？为什么不能开 16？',
    a: '取 **8**，两条约束同时卡住：\n1. **拓扑**：TP 每层 forward 2 次、backward 2 次集合通信，在关键路径上，下一步要等它。TP 16 要跨节点，带宽从 450 GB/s 掉到 50 GB/s，慢约 9 倍。\n2. **KV head**：GQA 下按 head 切，$t$ 超过 KV head 数（8）时，KV head 要在多张卡上复制。\n还要切更多就加 PP（切模型状态）或 CP（切序列、降激活）。',
    ref: '/train/parallel-config',
  }),
  b({
    id: 'bagu-parallel-config-mfu-first',
    topic: '并行配置',
    q: '70B 训练 MFU 只有 30%。profile 一个 step：5.0 s 里 GEMM 和 attention 3.0 s，其他计算 kernel 0.8 s，NCCL 单独跑（计算 stream 空着）0.9 s，完全空闲 0.3 s。配置里开着 full recompute，峰值显存 45 / 80 GB。先动哪个？',
    a: '先关 full recompute。按 checklist 的顺序（profile → micro-batch → recompute → overlap → 融合 → 并行配置），先做便宜、不改收敛的。\n1. 显存还有约 35 GB 余量，够把激活存回来（或换成 selective）。\n2. full recompute 让每 token 实际做 $8N$、MFU 只记 $6N$；关掉后计算部分约省 1/4，MFU 直接涨。\n3. 接着看 0.9 s 单独跑的 NCCL：打开 overlap 开关（`--overlap-grad-reduce` 等），按原理判断能不能藏。\n4. 0.8 s 的其他 kernel 留给融合 / torch.compile。\n并行配置最后才动，因为它连带显存、batch 和通信全要重算。',
    ref: '/train/parallel-config',
  }),

  // ---------------- FP8 训练 ----------------
  b({
    id: 'bagu-fp8-training-three-gemms',
    topic: 'FP8 训练',
    q: '一个 Linear 层 $Y = XW$，$X$ 是 $[M, K]$，$W$ 是 $[K, N]$。训练一步里它有哪三个 GEMM？Transformer Engine 默认的 `HYBRID` 格式下，每个 GEMM 的两个输入各用 e4m3 还是 e5m2？三个 GEMM 合计多少 FLOPs？',
    a: '1. **Fprop**：$Y = XW$，沿 $K$ 求和。$X$ e4m3，$W$ e4m3。\n2. **Dgrad**：$dX = dY \\cdot W^\\top$，沿 $N$ 求和。$dY$ e5m2，$W$ e4m3。\n3. **Wgrad**：$dW = X^\\top \\cdot dY$，沿 $M$（token）求和。$X$ e4m3，$dY$ e5m2。\n规则：forward 的张量（权重、激活）用 e4m3，精度高；backward 的梯度 $dY$ 用 e5m2，范围大。\n每个 GEMM $2MKN$ FLOPs（乘、加各一次），合计 $6MKN$。每个参数（$KN$ 个）每个 token（$M$ 个）6 FLOPs，就是训练 $6\\Psi$ FLOPs/token 的来源。\nfp8 只出现在输入上，输出都在 fp32 里累加。',
    ref: '/train/fp8-training',
  }),
  b({
    id: 'bagu-fp8-training-amdahl',
    topic: 'FP8 训练',
    q: 'H100 SXM dense 峰值：bf16 989 TFLOP/s，fp8 1979 TFLOP/s。bf16 训练时 GEMM 占一步时间的 70%，其余（attention 核心、norm、通信、优化器）不变。全部 GEMM 换成 fp8 后，端到端加速比上限是多少？实际为什么还更低？',
    a: 'Amdahl：设 GEMM 占比 $f$，GEMM 加速 2 倍，\n$$\\text{加速比} = \\frac{1}{(1 - f) + f/2} = \\frac{1}{0.3 + 0.35} = 1.54$$\n实际更低：\n1. 量化有开销：每个 fp8 输入要求 amax、乘 scale、cast，backward 还要转置版本，都是 memory-bound 的小 kernel。\n2. 小矩阵的 GEMM 本来就到不了峰值。\n参考：PyTorch 团队在 128 张 H100 上训 Llama3-70B，float8 比 bf16 快 1.50×，其中约 20% 来自 FSDP 的 fp8 all-gather（作者自测）。',
    ref: '/train/fp8-training',
  }),
  b({
    id: 'bagu-fp8-training-scale',
    topic: 'FP8 训练',
    q: 'per-tensor scale 的定义是 $s = q_{\\max} / \\text{amax}$，$q = \\text{fp8}(x \\cdot s)$，反量化 $\\hat{x} = q / s$。e4m3 的 $q_{\\max} = 448$，在 $[1, 2)$ 里相邻格点间隔 $2^{-3}$。一个张量 amax = 3.5，求 $s$；$x = 0.01$ 量化后反量化回来是多少、相对误差多大？',
    a: '1. $s = 448 / 3.5 = 128$。\n2. $0.01 \\times 128 = 1.28$。\n3. $[1, 2)$ 里的格点是 1.0, 1.125, 1.25, 1.375, ...，1.28 最近的是 **1.25**。\n4. 反量化 $1.25 / 128 = 0.00977$，相对误差 $(0.01 - 0.00977)/0.01 \\approx$ **2.3%**。\n符号：amax 是这组元素的 $\\max|x|$；乘 $s$ 让最大的数正好落在 448。\ne4m3 正规数的最大相对舍入误差是半个间隔 $2^{-4} = 6.25\\%$，和 $s$ 无关；$s$ 只决定数落不落在正规数范围里。',
    ref: '/train/fp8-training',
  }),
  b({
    id: 'bagu-fp8-training-delayed-overflow',
    topic: 'FP8 训练',
    q: 'delayed scaling 用前若干步的 amax 历史（取最大值）算这一步的 scale：$s = 448 / \\text{amax}_{\\text{hist}}$。历史 amax 是 3.5，这一步突然出现一个值 10。它量化后变成多少，反量化回来误差多大？current scaling 为什么没有这个问题？',
    a: '1. $s = 448 / 3.5 = 128$。\n2. $10 \\times 128 = 1280 > 448$，超出 e4m3 的最大值，只能 clamp 到 448（e4m3 没有 inf，不 clamp 直接 cast 会变 NaN）。\n3. 反量化 $448 / 128 = 3.5$，相对误差 $(10 - 3.5)/10 =$ **65%**。\ncurrent scaling 在 cast 之前先扫一遍当前张量求 amax，scale 跟着当前数据走，最大值总能落在 448 以内。代价是多读一遍张量。\ndelayed 的另一个问题：历史窗口（TE 当前默认 `amax_history_len=1024`）取最大值，数值变小后要等旧的大值滑出窗口，scale 才跟上。',
    ref: '/train/fp8-training',
  }),
  b({
    id: 'bagu-fp8-training-per-tensor-range',
    topic: 'FP8 训练',
    q: 'e4m3 的正规数范围是 $[2^{-6}, 448]$，比它小的进入非正规数区（间隔固定 $2^{-9} \\approx 0.002$）。一个张量用 per-tensor scale，amax = $10^4$，普通元素的大小约 1。乘 scale 后普通元素变成多少？还在正规数范围里吗？',
    a: '1. $s = 448 / 10^4 = 0.0448$。\n2. 普通元素 $1 \\times 0.0448 = 0.0448$，大于 $2^{-6} = 0.0156$，**还在正规数范围**，相对误差仍由 3 位尾数决定（几个百分点）。\n3. 如果 amax = $10^5$：$s = 0.00448$，普通元素变成 0.00448，**跌进非正规数区**。间隔 $2^{-9} \\approx 0.002$，最大舍入误差半个间隔约 0.001，相对它是 $0.001 / 0.0045 \\approx 22\\%$；原值小于约 0.22 的元素（乘 scale 后不到半个间隔）直接变 0。\n规律：一组里最大值和普通值之比超过 $448 / 2^{-6} \\approx 2.9 \\times 10^4$，per-tensor 就不够了，要用更细的块 scale（如 DeepSeek-V3 的 $1 \\times 128$）。在这之前 fp8 作为浮点数对离群值并不敏感，这是它比 int8 更适合 per-tensor 的原因。',
    ref: '/train/fp8-training',
  }),
  b({
    id: 'bagu-fp8-training-dsv3-scales',
    topic: 'FP8 训练',
    q: 'DeepSeek-V3 的 fp8 训练：激活每个 token 每 128 个通道（$1 \\times 128$）一个 fp32 scale，权重每 $128 \\times 128$ 块一个。激活 $X$ 是 $[4096, 7168]$，权重 $W$ 是 $[7168, 2048]$。各有多少个 scale？激活的 scale 每元素摊多少 bit？',
    a: '1. 激活：每行 $7168 / 128 = 56$ 组，共 $4096 \\times 56 =$ **229,376** 个 scale。\n2. 权重：$(7168/128) \\times (2048/128) = 56 \\times 16 =$ **896** 个。\n3. 激活每元素摊 $32 / 128 =$ **0.25 bit**（一个 fp32 scale 管 128 个元素）。\n为什么这样分：\n1. 激活的离群值按 token 出现，按 token 分组能把离群 token 和普通 token 隔开。\n2. 权重用方块，$W$ 和 $W^\\top$ 的分块完全一样，Dgrad 用 $W^\\top$ 时不用重新量化。\n3. scale 每步在线算（current），不用历史。',
    ref: '/train/fp8-training',
  }),
  b({
    id: 'bagu-fp8-training-promotion',
    topic: 'FP8 训练',
    q: 'DeepSeek-V3 测得 H800 上 FP8 GEMM 的 Tensor Core 累加器只保留约 14 bit。它的做法是沿 $K$ 每 $N_C = 128$ 个元素把部分和拷到 CUDA core 上用 fp32 累加。$K = 7168$ 时每个输出元素要提升几次？为什么累加精度不够会出问题？',
    a: '1. $7168 / 128 =$ **56** 次。$N_C = 128$ 对应 4 个 WGMMA。\n2. 为什么：累加和越来越大以后，再加一个小的乘积，低位会被舍掉。累加器只有约 14 bit 时，$K$ 越大丢得越多；论文测 $K = 4096$ 的随机矩阵乘，最大相对误差接近 **2%**（作者自测）。\n3. 每 128 个元素提升一次，Tensor Core 里只累加 128 项，误差有限；之后在 fp32 里累加。\n4. 顺便：$1 \\times 128$ / $128 \\times 128$ 的块 scale 正好在提升时乘上去（反量化），几乎不额外花钱。\n5. 两个 warpgroup 交替：一个做提升时另一个做 MMA，Tensor Core 不闲着。',
    ref: '/train/fp8-training',
  }),
  b({
    id: 'bagu-fp8-training-mx-scale',
    topic: 'FP8 训练',
    q: 'MXFP8：每 32 个元素共享一个 E8M0 scale（8 位指数、0 位尾数，只能是 2 的幂），元素用 e4m3（最大 448）。NVIDIA 的配方取 $X = 2^{\\lceil \\log_2(\\text{amax}/448) \\rceil}$，$q = \\text{fp8}(x / X)$。一块的 amax = 5，求 $X$；如果指数向下取整会怎样？scale 每元素摊几 bit？',
    a: '1. $5 / 448 = 0.0112$，$\\log_2 0.0112 = -6.49$。\n2. 向上取整 $-6$：$X = 2^{-6} = 1/64$，最大元素 $5 \\times 64 = 320 \\le 448$，不溢出。\n3. 向下取整 $-7$：$X = 1/128$，$5 \\times 128 = 640 > 448$，**溢出**。论文（arXiv 2506.08027）认为 OCP v1 的取法相当于向下取，这是 MXFP8 训练出问题的原因之一。\n4. 开销：8 bit / 32 元素 = **0.25 bit**，和 DeepSeek-V3 激活的 fp32 / 128 一样。\n和 DeepSeek-V3 的区别：块更小（32），scale 是 2 的幂（乘 scale 只改指数、无舍入），在 Blackwell 的 Tensor Core 里原生乘，不用回 CUDA core。',
    ref: '/train/fp8-training',
  }),
  b({
    id: 'bagu-fp8-training-memory',
    topic: 'FP8 训练',
    q: 'FlashAttention 下每层激活约 $34sbh$ 字节（$s$ 序列长度、$b$ micro-batch、$h$ hidden），其中 Linear 层的输入是 QKV 输入 $2sbh$、输出投影输入 $2sbh$、MLP 两个输入 $2sbh + 8sbh$。把这些 Linear 输入存成 fp8，每层激活变成多少？模型状态（每参数 16 字节）省多少？',
    a: '1. Linear 输入合计 $2 + 2 + 2 + 8 = 14sbh$，bf16 → fp8 减半，省 $7sbh$。\n2. 每层激活从 $34sbh$ 变成 $27sbh$，**省约 21%**。7B（$L = 32$，$h = 4096$，$s = 4096$，$b = 1$）从 18.3 GB 到 14.5 GB。\n3. 模型状态**一点不省**：bf16 参数 2 + 梯度 2 + fp32 主权重 4 + Adam $m$、$v$ 8 = 16 字节/参数，fp8 没动它们；反而多出 fp8 权重的缓存。\n4. 通信省一半：FSDP all-gather 权重从 $2\\Psi$ 字节变 $\\Psi$，MoE dispatch 也减半。\nfp8 主要买的是计算速度；模型状态还要靠 ZeRO / FSDP 切。',
    ref: '/train/fp8-training',
  }),

  // ---------------- 长上下文训练 ----------------
  b({
    id: 'bagu-lc-attn-ratio',
    topic: '长上下文',
    q: '训练每 token：线性层 $6N$ FLOPs，causal attention $6Lhs$ FLOPs。Llama 3 8B：$N = 8\\times10^9$，$L = 32$，$h = 4096$。$s = 8192$ 和 $s = 131072$ 时 attention 与线性层之比各是多少？每 token 算力涨了几倍？',
    a: '1. 比值 $\\frac{6Lhs}{6N} = \\frac{Lhs}{N}$，$Lh/N = 32 \\times 4096 / 8\\times10^9 = 1.64\\times10^{-5}$。\n2. $s = 8$K：**0.13**；$s = 128$K：**2.15**。\n3. 每 token 算力：$(1 + 2.15) / (1 + 0.13) \\approx$ **2.8 倍**。\n$6Lhs$ 的来由：位置 $i$ 的 query 做 $QK^\\top$ 和 $PV$ 共 $4hi$ FLOPs，causal 下 $i$ 平均 $s/2$，forward 每层 $2hs$；backward 是 forward 的 2 倍，合计 $6hs$，再乘 $L$。GQA 不减少这部分（query head 数没变）。',
    ref: '/train/long-context',
  }),
  b({
    id: 'bagu-lc-act-mem',
    topic: '长上下文',
    q: 'FlashAttention 下每层激活约 $34sbh$ 字节。Llama 3 8B（$L = 32$，$h = 4096$），$s = 131072$，$b = 1$。总激活多少？用 CP 度 $c = 16$ 切序列后每卡多少？',
    a: '1. 每层：$34 \\times 131072 \\times 4096 \\approx 18.3$ GB；32 层 **584 GB**。\n2. CP 把每卡的序列变成 $s/c = 8192$，激活按 $s$ 线性缩：$584 / 16 \\approx$ **36.5 GB**。\n$34sbh$ 是 Korthikanti et al. 2022 数出来的各个 $[s,b,h]$ 级张量之和（FlashAttention 去掉了 $s^2$ 项）。checkpointing、TP+SP 都只是把系数变小，只有 CP 让每卡激活和总长度脱钩。',
    ref: '/train/long-context',
  }),
  b({
    id: 'bagu-lc-zigzag',
    topic: '长上下文',
    q: 'Context Parallel，$c = 4$ 张卡，causal mask。序列平均切 4 段时各卡工作量之比是多少？zigzag 怎么切？证明它均衡。',
    a: '1. 平均切：第 $i$ 段要看前 $i$ 段整块加自己一半，工作量正比于 $i + 0.5$：**0.5 : 1.5 : 2.5 : 3.5**，最慢的卡是平均的 1.75 倍。\n2. zigzag：切成 $2c = 8$ 块，卡 $i$ 拿第 $i$ 块和第 $2c-1-i$ 块：卡 0 拿 {0,7}，卡 1 拿 {1,6}，卡 2 拿 {2,5}，卡 3 拿 {3,4}。\n3. 证明：块 $j$ 的工作量 $j + 0.5$（单位：一个 $(s/2c)^2$ 块）。卡 $i$：$(i+0.5) + (2c-1-i+0.5) = 2c$，与 $i$ 无关，每卡都是 8。\nLlama 3 的 CP 就是这样切的。',
    ref: '/train/long-context',
  }),
  b({
    id: 'bagu-lc-ring-hide',
    topic: '长上下文',
    q: 'Ring Attention 每一步：本地 $s/c$ 个 query 对一块 $s/c$ 个 key 算 attention（$4(s/c)^2h$ FLOPs），同时传一块 KV（$2(s/c)H_{kv}d_h$ 个 bf16 元素）。H100 按 494 TFLOP/s（bf16 989 的 50%），IB 每卡 50 GB/s，$H_{kv}/H = 1/4$。每卡至少要多少 token 才藏得住通信？',
    a: '1. 条件：$\\frac{4(s/c)^2 h}{\\text{peak}\\cdot\\text{MFU}} \\ge \\frac{4(s/c)H_{kv}d_h}{\\text{BW}}$，即 $\\frac{s}{c} \\ge \\frac{H_{kv}}{H} \\cdot \\frac{\\text{peak}\\cdot\\text{MFU}}{\\text{BW}}$（$h = Hd_h$）。\n2. 代入：$\\frac{1}{4} \\times \\frac{494\\times10^{12}}{50\\times10^9} \\approx$ **2470** 个 token。\n3. causal + zigzag 时每步平均只算一半，门槛约翻倍到 4940。\n128K、$c = 16$ 时每卡 8192 个，跨节点也藏得住。GQA 让 KV 变小，门槛按 $H_{kv}/H$ 降低。',
    ref: '/train/long-context',
  }),
  b({
    id: 'bagu-lc-allgather-kv',
    topic: '长上下文',
    q: 'Llama 3 的 CP 先 all-gather 全部 K、V 再算本地 Q。405B：8 个 KV head，$d_h = 128$，$s = 131072$，$c = 16$，bf16。每个 CP rank 每层要收多少字节？IB 50 GB/s 要多久？作者为什么接受它不和计算重叠？',
    a: '1. 完整 $K+V$：$2 \\times 131072 \\times (8 \\times 128) \\times 2 \\approx 537$ MB；收其中 $15/16$：**503 MB**。\n2. $503\\text{ MB} / 50\\text{ GB/s} \\approx$ **10 ms**。\n3. 同一 rank 每层 forward 计算约 $8.8\\times10^{13}$ FLOP（attention $2s^2h/c$ 加线性层），按 494 TFLOP/s 约 178 ms，通信约 6%。\n作者的两个理由：all-gather 后本地有完整 KV，文档掩码等各种 mask 好做；GQA 让 KV 比 Q 小得多，attention 是 $O(s^2)$、all-gather 是 $O(s)$。',
    ref: '/train/long-context',
  }),
  b({
    id: 'bagu-lc-ulysses',
    topic: '长上下文',
    q: 'DeepSpeed-Ulysses 用 all-to-all 把「按序列切」换成「按 head 切」。Llama 3 8B 有 $H = 32$ 个 query head、$H_{kv} = 8$ 个 KV head。Ulysses 度 $c$ 的上限是多少？超过 8 会怎样？',
    a: '1. all-to-all 之后每张卡拿 $H/c$ 个 head、看完整序列，所以 $c \\le H = 32$ 且整除 $H$。\n2. K、V 只有 8 个 head：$c \\le 8$ 时每卡分到整数个 KV head；$c > 8$ 时同一个 KV head 要发给多张卡（**复制 KV**），通信和显存都变多。\n所以 GQA 模型上 Ulysses 实际常限制在 $H_{kv}$ 以内，更长的序列再在外层叠 Ring。优点是本地就是普通 attention，任何 mask 和 kernel 都能直接用。',
    ref: '/train/long-context',
  }),
  b({
    id: 'bagu-lc-rope-wavelength',
    topic: '长上下文',
    q: 'RoPE 第 $j$ 对维度角频率 $\\theta_j = \\Theta^{-2j/d_h}$，波长 $\\lambda_j = 2\\pi/\\theta_j$。$d_h = 128$。$\\Theta = 10{,}000$ 时最低频一对（$j = 63$）的波长是多少？$\\Theta$ 调到 500,000（ABF）后呢？',
    a: '1. $\\Theta = 10^4$：$\\theta_{63} = 10^{4 \\times (-126/128)} \\approx 1.15\\times10^{-4}$，$\\lambda_{63} = 2\\pi / \\theta_{63} \\approx$ **54,410** 个 token。\n2. $\\Theta = 5\\times10^5$：$\\theta_{63} \\approx 2.46\\times10^{-6}$，$\\lambda_{63} \\approx$ **256 万**。\n意义：波长超过训练长度的维度在训练中没转满一圈，超出长度就会遇到没见过的角度。ABF 让 $\\theta_j$ 变成原来的 $(\\Theta/\\Theta\')^{2j/d_h}$：$j = 0$ 不变，越低频降得越多（最后一对约 1/47）。',
    ref: '/train/long-context',
  }),
  b({
    id: 'bagu-lc-pi',
    topic: '长上下文',
    q: 'Position Interpolation 把 4096 训练的模型扩到 16384（$\\kappa = 4$）。位置 $m = 10000$ 的 token 用哪个位置算 RoPE？对第 $j$ 对维度的频率意味着什么？代价是什么？',
    a: '1. $m \\to m/\\kappa = 10000/4 =$ **2500**，落回原训练范围 $[0, 4096)$。\n2. 等价于所有维度 $\\theta_j \\to \\theta_j/\\kappa$：每对维度一起放慢 4 倍，不会出现没见过的角度。\n3. 代价：高频维度也被压缩，相邻 token 的角度差缩小 4 倍，局部位置分辨率下降，需要微调（作者在 LLaMA 上扩到 32768，微调 1000 步以内，作者自测）。\nYaRN 改进的就是这点：高频维度不动，只压低频维度。',
    ref: '/train/long-context',
  }),
  b({
    id: 'bagu-lc-yarn',
    topic: '长上下文',
    q: 'YaRN：$r_j = s_0/\\lambda_j$ 是第 $j$ 对维度在原训练长度 $s_0$ 内转的圈数；$r_j < \\alpha$ 按 PI 除以 $\\kappa$，$r_j > \\beta$ 不动，中间线性过渡，$\\alpha = 1$、$\\beta = 32$。$s_0 = 4096$ 时，波长 $\\lambda = 63$、$628$、$54410$ 的三对维度各怎么处理？$\\kappa = 16$ 时温度 $\\sqrt{1/t}$ 是多少？',
    a: '1. $\\lambda = 63$：$r = 4096/63 = 65 > 32$，**不动**（高频，训练中转过很多圈）。\n2. $\\lambda = 628$：$r = 6.5$，$\\gamma = (6.5-1)/31 = 0.18$，$\\theta\' = 0.82\\,\\theta/16 + 0.18\\,\\theta \\approx 0.23\\,\\theta$，**部分压缩**。\n3. $\\lambda = 54410$：$r = 0.075 < 1$，**除以 16**（和 PI 一样）。\n4. $\\sqrt{1/t} = 0.1\\ln 16 + 1 \\approx$ **1.28**，乘在 $q$、$k$ 上，等于给 attention logits 加温度。',
    ref: '/train/long-context',
  }),
  b({
    id: 'bagu-lc-staged',
    topic: '长上下文',
    q: '405B（$L = 126$，$h = 16384$，$N = 4.05\\times10^{11}$）。每 token 训练算力 $\\propto 1 + \\frac{Lhs}{N}$。全程 15.6T token 用 128K 比用 8K 贵几倍？Llama 3 只在最后约 800B token 扩到 128K，最多多花多少？',
    a: '1. $Lh/N = 126 \\times 16384 / 4.05\\times10^{11} \\approx 5.1\\times10^{-6}$；8K 时 $Lhs/N = 0.042$，128K 时 0.668。\n2. 全程 128K：$1.668 / 1.042 \\approx$ **1.6 倍**。\n3. 分阶段（800B 全按 128K 算，是上界）：$14.8\\text{T} \\times 1.042 + 0.8\\text{T} \\times 1.668 = 16.75\\text{T}$，对比全程 8K 的 16.25T，只多约 **3%**。\nLlama 3 405B 实际分 6 个阶段从 8K 扩到 128K；每阶段训到短任务指标恢复、大海捞针全对再往上扩。',
    ref: '/train/long-context',
  }),

  // ---------------- 训练稳定性 ----------------
  b({
    id: 'bagu-stability-z-loss',
    topic: 'z-loss',
    q: '一个位置的输出 logits $y = [-100, -101, -102, -103]$（词表 $V = 4$），目标是第 0 个。$Z = \\sum_j e^{y_j}$，CE $= \\log Z - y_0$，z-loss $= 10^{-4}\\log^2 Z$（PaLM）。算 $\\log Z$、CE、z-loss。所有 logits 加 100 后这三个数变成多少？',
    a: '1. $\\log Z = -100 + \\log(1 + e^{-1} + e^{-2} + e^{-3}) = -100 + 0.440 = -99.56$。\n2. CE $= -99.56 - (-100) = 0.440$。\n3. z-loss $= 10^{-4} \\times 99.56^2 = 0.991$，比 CE 还大。\n4. 加 100 后：softmax 对整体平移不变，CE 仍是 0.440；$\\log Z = 0.440$，z-loss $= 10^{-4} \\times 0.194 \\approx 1.9\\times10^{-5}$。\n含义：CE 不约束 $\\log Z$ 的绝对位置，logits 可以整体漂走（Wortsman et al. arXiv 2309.14322 复现到 logits 整体变得很负）。z-loss 只消除这个平移自由度，把 $\\log Z$ 拉回 0 附近，不改概率；logits 在 0 附近时 bf16 分辨率才够。',
    ref: '/train/stability',
  }),
  b({
    id: 'bagu-stability-softcap',
    topic: 'Logit soft-capping',
    q: 'Gemma 2 的 logit soft-capping：$\\text{softcap}(x) = c\\tanh(x/c)$，最终输出 logits 用 $c = 30$。算 $x = 10$、$60$、$100$ 时的输出（$\\tanh(1/3) = 0.3215$，$\\tanh 2 = 0.9640$，$\\tanh(10/3) = 0.9975$）。',
    a: '1. $x = 10$：$30 \\times 0.3215 = 9.65$，$\\lvert x\\rvert \\ll c$ 时近似恒等。\n2. $x = 60$：$30 \\times 0.9640 = 28.92$。\n3. $x = 100$：$30 \\times 0.9975 = 29.92$，被压在 $c$ 以下。\nGemma 2 对 attention logits 用 $c = 50$，最终 logits 用 $c = 30$（arXiv 2408.00118）。用在 attention 上时，tanh 要在 $QK^\\top$ 之后、softmax 之前逐元素做，backward 还要乘 tanh 的导数，必须写进融合 kernel：Gemma 2 发布时它和 FlashAttention / SDPA 不兼容（HF 建议微调用 eager attention），FlashAttention 2.6 起加了 softcap 参数。',
    ref: '/train/stability',
  }),
  b({
    id: 'bagu-stability-softmax-collapse',
    topic: 'Attention logit 增长',
    q: '一个 query 对两个 key 的 attention logits 是 2 和 1。之后 $W_Q$、$W_K$ 的 norm 各涨 3 倍（输入不变）。新的 logits 和 softmax 是多少？softmax 对自身 logit 的导数 $p(1-p)$ 从多少变成多少？',
    a: '$z = q^\\top k/\\sqrt{d_h}$，$q = W_Q x$、$k = W_K x$，所以 logit 对参数 norm 是二次的：各涨 3 倍，logit 涨 9 倍。\n1. 原来：softmax$(2, 1) = [0.731, 0.269]$，$p(1-p) = 0.197$。\n2. 之后：logits 变成 18 和 9，差 9，$e^{-9} = 1.2\\times10^{-4}$，softmax $\\approx [0.99988, 0.00012]$，$p(1-p) \\approx 1.2\\times10^{-4}$。\nattention 塌成 one-hot、熵接近 0，梯度几乎消失。ViT-22B（arXiv 2302.05442）在约 8B 参数时遇到这个问题：不做归一化时 logits 超过 50000，loss 发散（作者自测）。对策是 QK-norm：内积前对 $q$、$k$ 各做 LayerNorm / RMSNorm。',
    ref: '/train/stability',
  }),
  b({
    id: 'bagu-stability-qk-norm-bound',
    topic: 'QK-norm',
    q: 'QK-norm：attention logit $z = \\text{RMSNorm}(q)^\\top\\text{RMSNorm}(k)/\\sqrt{d_h}$，RMSNorm 的可学 scale $\\gamma = 1$，head 维度 $d_h = 128$。RMSNorm 后向量每个元素平方的均值为 1。$\\lvert z\\rvert$ 最大是多少？',
    a: '1. RMS 为 1 意味着 $\\lVert q\\rVert^2 = d_h$，即 $\\lVert q\\rVert = \\sqrt{d_h}$，$k$ 同理。\n2. Cauchy–Schwarz：$\\lvert q^\\top k\\rvert \\le \\lVert q\\rVert\\lVert k\\rVert = d_h$。\n3. 除以 $\\sqrt{d_h}$：$\\lvert z\\rvert \\le \\sqrt{d_h} = \\sqrt{128} \\approx 11.3$。\n不加 QK-norm 时 logit 随 $W_Q$、$W_K$ 的 norm 二次增长、没有上界；加了以后只有 $\\gamma$ 能放大它，增长慢得多。QK-norm 每个 head 在 $d_h$ 维上单独做。Qwen3、OLMo 2 都用了它。',
    ref: '/train/stability',
  }),
  b({
    id: 'bagu-stability-adam-eps',
    topic: 'Adam ε',
    q: 'Adam 更新 $\\eta\\,\\hat m/(\\sqrt{\\hat v} + \\epsilon)$。某层梯度的 RMS 稳定在 $g$，于是 $\\lvert\\hat m\\rvert \\approx \\sqrt{\\hat v} \\approx g$，更新是正常步长（$\\epsilon = 0$ 时）的 $g/(g + \\epsilon)$ 倍。$\\epsilon = 10^{-8}$ 时，算 $g = 10^{-6}$、$10^{-8}$、$10^{-9}$ 三种情况。',
    a: '1. $g = 10^{-6}$：$10^{-6}/(1.01\\times10^{-6}) = 0.99$，几乎没影响。\n2. $g = 10^{-8}$：$0.5$，步长减半。\n3. $g = 10^{-9}$：$10^{-9}/(1.1\\times10^{-8}) = 0.09$，几乎不更新。\nWortsman et al.（arXiv 2309.14322 §3.4，作者自测）发现梯度 RMS 随模型变大、lr 变大而下降，最大规模下已接近默认 $\\epsilon = 10^{-8}$，层会学不动；4.8B 模型、lr 0.3 时把 $\\epsilon$ 降到 $10^{-15}$ loss 更好，升到 $10^{-6}$ 则发散。OLMo 2 把 $\\epsilon$ 从 $10^{-5}$ 降到 $10^{-8}$ 后 grad norm 更早稳定（作者自测）。',
    ref: '/train/stability',
  }),
  b({
    id: 'bagu-stability-residual-init',
    topic: '残差初始化',
    q: 'GPT-2 / Megatron 的初始化：所有权重 $\\mathcal{N}(0, \\sigma^2)$，$\\sigma = 0.02$；每个残差分支的最后一个线性层（attention 输出投影、MLP down 投影）的 std 再除以 $\\sqrt{2L}$。$L = 32$ 层时这些层的 std 是多少？为什么是 $2L$？',
    a: '1. $2L = 64$，$0.02/\\sqrt{64} = 0.02/8 = 0.0025$。\n2. 为什么：残差流 $x_{2L} = x_0 + \\sum_{l=1}^{2L} f_l$，每层有 attention 和 MLP 两个分支，共 $2L$ 个。初始化时各分支输出近似独立，方差相加：每个分支方差 $s^2$ 时总和是 $2L s^2$，越深残差流越大。分支输出层 std 除以 $\\sqrt{2L}$，每个分支方差变成 $s^2/(2L)$，总和回到 $s^2$，与深度无关。\n出处：GPT-2 论文 §2.3（$1/\\sqrt N$，$N$ 为残差层数）；Megatron 的 `output_layer_init_method` 是 `init_method_std / sqrt(2 * num_layers)`，`init_method_std` 默认 0.02。',
    ref: '/train/stability',
  }),
  b({
    id: 'bagu-stability-mup-lr',
    topic: 'μP',
    q: 'μP（Tensor Programs V）用 Adam 时，宽度从基准 $n_0$ 放大到 $n$、记 $m = n/n_0$，要改三处：隐藏层矩阵的 lr 除以 $m$；输出 logits 乘 $1/m$；attention 缩放从 $1/\\sqrt{d_h}$ 改成 $1/d_h$。在 $n_0 = 256$ 上调出全局 lr $= 10^{-2}$，迁到 $n = 4096$，隐藏层矩阵和 embedding 的 lr 各用多少？',
    a: '1. $m = 4096/256 = 16$。\n2. 隐藏层矩阵：$10^{-2}/16 = 6.25\\times10^{-4}$（mup 库的 `MuAdam` 就是 `globalLR / width_mult`）。\n3. embedding、bias、norm 的 scale：只有一个维度随宽度变，lr 不缩放，仍是 $10^{-2}$。\n为什么隐藏层要除以 $m$：Adam 的更新每个元素约为 $\\eta$，而 $\\Delta W \\propto g x^\\top$ 和输入 $x$ 对齐，$(\\Delta W x)_i$ 是 $n$ 个同号项相加，约 $\\eta n$。宽度翻倍激活变化就翻倍，最优 $\\eta$ 要减半。μP 把这个 $1/n$ 写进参数化，最优全局 lr 就和宽度无关。论文从 40M 模型迁移超参，超过公开的 GPT-3 6.7B 结果，调参成本为预训练的 7%（作者自测）。',
    ref: '/train/stability',
  }),
  b({
    id: 'bagu-stability-palm-rollback',
    topic: 'Loss spike 处理',
    q: 'PaLM 处理 loss spike 的做法：回到 spike 开始前约 100 步的 checkpoint，跳过约 200–500 个 batch 再训。某次训练在第 30,000 步开始 spike，global batch 为 2048 条序列，决定跳过 300 个 batch。从第几步的 checkpoint 重启？data loader 的全局样本号要往前跳多少条？',
    a: '1. 重启点：$30{,}000 - 100 = 29{,}900$ 步（要求 checkpoint 间隔不超过 100 步，或取之前最近的一个）。\n2. 跳过：$300 \\times 2048 = 614{,}400$ 条序列，即把 `consumed_samples` 加上 614,400。data loader 必须确定、能按全局样本号恢复，否则跳不准。\n3. 为什么有效：PaLM（arXiv 2204.02311 §5.1，作者自测）把 spike 附近的 batch 从更早的另一个 checkpoint 训，不会 spike。所以 spike 是特定数据和特定参数状态的组合，不是单纯的坏数据；换了顺序，参数遇到这批数据时状态不同，组合不再出现。\nPaLM 540B 约有 20 次这样的 spike，开着梯度裁剪也有。',
    ref: '/train/stability',
  }),
  b({
    id: 'bagu-stability-spike-detect',
    topic: 'Spike 检测',
    q: 'spike 检测规则：当前 loss $>$ 前 $w$ 步的均值 $\\mu$ + $k$ 倍标准差 $\\sigma$ 就报警，报警的步不进入统计窗口。$w = 50$ 步的窗口里 $\\mu = 2.50$、$\\sigma = 0.02$，$k = 5$。接下来两步的 loss 是 2.58 和 2.65，哪一步报警？为什么报警的步不能进窗口？',
    a: '1. 阈值 $= 2.50 + 5 \\times 0.02 = 2.60$。\n2. 2.58 < 2.60，不报警，进入窗口；2.65 > 2.60，报警。\n3. 报警的步如果进窗口，会把 $\\sigma$ 拉大、阈值抬高，紧跟着的第二个 spike 就检测不到了。\n只看 loss 不够：OLMo 2（arXiv 2501.00656 §3，作者自测）观察到 grad norm 的 spike 常先于 loss spike，所以同样的检测也该加在裁剪前的 grad norm、每层最大 attention logit 和 $\\log Z$ 上。',
    ref: '/train/stability',
  }),
  b({
    id: 'bagu-stability-bf16-logits',
    topic: 'bf16 精度',
    q: 'bf16 尾数 7 位：数值在 $[2^e, 2^{e+1})$ 区间时相邻两数的间隔是 $2^{e-7}$。一个 logit 是 $-200.3$，存成 bf16 后最近的可表示数是多少？舍入误差最大可到多少，对应的概率比例误差是多少（$e^{\\Delta}$）？',
    a: '1. $\\lvert -200.3\\rvert$ 在 $[128, 256)$，$e = 7$，间隔 $2^{0} = 1.0$。\n2. 最近的可表示数是 $-200$，误差 0.3；最坏情况误差是半个间隔 0.5。\n3. softmax 里 logit 差 $\\Delta$ 对应概率乘 $e^{\\Delta}$：$e^{0.5} = 1.65$，即概率可能偏 65%。\n所以 logits、logsumexp、CE 要在 fp32 里算；也说明 z-loss 把 $\\log Z$ 拉回 0 附近的意义：logits 绝对值小，bf16 的间隔就小（在 $[1, 2)$ 里是 $2^{-7} \\approx 0.0078$）。',
    ref: '/train/stability',
  }),

  // ---------------- 训练容错 ----------------
  b({
    id: 'bagu-fault-tolerance-job-mtbf',
    topic: '训练容错',
    q: 'Llama 3 405B 预训练的一段 54 天里有 419 次非计划中断，作业用了约 16384 张 H100。作业 MTBF 是多少？反推单卡等效 MTBF 是多少？同样的卡跑 1024 卡的作业，MTBF 是多少？',
    a: '公式：同步训练里任何一张卡出事作业就停。各卡独立、故障率恒定（指数分布）时，$n$ 张卡的故障率相加：\n$$M = \\frac{M_{\\text{gpu}}}{n}$$\n$M$ 是作业 MTBF，$M_{\\text{gpu}}$ 是单卡 MTBF，$n$ 是卡数。\n1. 作业 MTBF：$54 \\times 24 / 419 =$ **3.09 小时**。\n2. 单卡等效：$3.09 \\times 16384 \\approx$ **50,700 小时**，约 5.8 年（把 GPU、网络、主机、软件所有原因都摊到每张卡上）。\n3. 1024 卡：$50{,}700 / 1024 =$ **49.5 小时**，约两天一次。\n单卡很可靠，是卡数把故障频率放大了。',
    ref: '/train/fault-tolerance',
  }),
  b({
    id: 'bagu-fault-tolerance-ckpt-size',
    topic: '训练容错',
    q: '混合精度 Adam 训练，checkpoint 存 fp32 主权重和 Adam 的 $m$、$v$。405B 模型的 checkpoint 多大？所有 rank 并行写，文件系统总写带宽 50 GB/s，同步写一次要停多久？',
    a: '1. 每参数：主权重 4 + $m$ 4 + $v$ 4 = **12 字节**。bf16 参数从主权重 cast 得到、梯度一步用完就扔，都不用存（所以比显存里的 16 字节少 4）。\n2. 大小：$405 \\times 10^9 \\times 12 = 4.86 \\times 10^{12}$ 字节 = **4.86 TB**（TB = $10^{12}$ 字节）。\n3. 写入时间 $\\delta = 4.86\\text{ TB} / 50\\text{ GB/s} \\approx$ **97 s**。\n前提是每个 rank 写自己的分片、用满文件系统的总带宽（PyTorch DCP 的做法）。如果 gather 到 rank 0 一个人写，带宽只有单节点的，慢一两个数量级。\n另外还要存 step、学习率调度器、RNG、数据加载器位置，都很小。',
    ref: '/train/fault-tolerance',
  }),
  b({
    id: 'bagu-fault-tolerance-young',
    topic: '训练容错',
    q: '写一次 checkpoint 要停 $\\delta = 100$ s，作业 MTBF $M = 3$ 小时。用 Young 公式算最优 checkpoint 间隔 $\\tau^*$，并说明公式从哪来。',
    a: '单位时间浪费的比例：\n$$W(\\tau) = \\frac{\\delta}{\\tau} + \\frac{\\tau/2}{M} + \\frac{R}{M}$$\n1. $\\delta/\\tau$：每隔 $\\tau$ 的训练写一次，停 $\\delta$。\n2. $\\tau/(2M)$：故障在间隔内均匀发生，平均丢半个间隔；单位时间发生 $1/M$ 次。\n3. $R/M$：每次故障重启 $R$，和 $\\tau$ 无关。\n对 $\\tau$ 求导令其为 0：$\\delta/\\tau^2 = 1/(2M)$，\n$$\\tau^* = \\sqrt{2\\delta M}$$\n代入：$M = 10800$ s，$\\tau^* = \\sqrt{2 \\times 100 \\times 10800} = \\sqrt{2.16 \\times 10^6} \\approx$ **1470 s ≈ 24.5 分钟**。\n前提 $\\delta \\ll \\tau \\ll M$（一阶近似，Young 1974；Daly 2006 有高阶修正）。',
    ref: '/train/fault-tolerance',
  }),
  b({
    id: 'bagu-fault-tolerance-ettr',
    topic: '训练容错',
    q: 'ETTR（有效训练时间比例）= 真正推进训练的时间 / 墙钟时间。在 Young 最优间隔下，ETTR $\\approx 1 - \\sqrt{2\\delta/M} - R/M$。$\\delta = 100$ s，$M = 10800$ s，重启耗时 $R = 600$ s，ETTR 是多少？$\\delta$ 降到 5 s 呢？',
    a: '公式来历：浪费 $W = \\delta/\\tau + \\tau/(2M) + R/M$，在 $\\tau^* = \\sqrt{2\\delta M}$ 处前两项相等、都是 $\\sqrt{\\delta/(2M)}$，加起来 $\\sqrt{2\\delta/M}$。\n1. $\\delta = 100$：$\\sqrt{200/10800} = 0.136$，$R/M = 0.056$，ETTR $\\approx$ **0.81**。\n2. $\\delta = 5$：$\\sqrt{10/10800} = 0.030$，ETTR $\\approx 1 - 0.030 - 0.056 =$ **0.91**。\n3. 剩下的大头是 $R/M$，和间隔无关，只能靠缩短重启（内存 checkpoint、热备节点、快速检测）。\n符号：$\\delta$ 写 checkpoint 让训练停的时间，$M$ 作业 MTBF，$R$ 从故障到恢复训练的时间。',
    ref: '/train/fault-tolerance',
  }),
  b({
    id: 'bagu-fault-tolerance-async',
    topic: '训练容错',
    q: '异步 checkpoint 怎么把训练停顿从「写盘时间」降到「GPU → CPU 拷贝时间」？405B 的 checkpoint 4.86 TB，文件系统总写带宽 50 GB/s，异步之后 checkpoint 间隔最短能多短？',
    a: '两段：\n1. 阻塞段：每个 rank 把自己的分片从 GPU 拷到 CPU 内存（D2H），拷完训练继续。\n2. 后台段：CPU 线程把内存里的副本序列化、写盘。\n训练只停第一段。IBM 实测 7B 从平均 148.8 s 降到 6.3 s（作者自测，PyTorch 博客）。\n间隔下限：后台写盘没完成前不能开始下一次，所以 $\\tau \\ge 4.86\\text{ TB} / 50\\text{ GB/s} \\approx$ **97 s**。\n其他约束：\n1. CPU 内存要多放一份分片。\n2. checkpoint 线程的 collective 要用单独的 process group，否则和训练的 collective 互相等会死锁。\nAPI：`torch.distributed.checkpoint.async_save`。',
    ref: '/train/fault-tolerance',
  }),
  b({
    id: 'bagu-fault-tolerance-dedup',
    topic: '训练容错',
    q: '405B 用 TP 8 × PP 16 × DP 128 训练，ZeRO-1 把优化器状态在 DP 维切开。如果 checkpoint 里还要存 bf16 权重，每个 DP 副本都写一份，总共写多少？去重后呢？优化器状态需要去重吗？',
    a: '1. bf16 权重 $405 \\times 10^9 \\times 2 = 810$ GB。每个 DP 副本（TP × PP 的一组）都持有完整一份。\n2. 不去重：$810 \\text{ GB} \\times 128 =$ **104 TB**。\n3. 去重：只让一个副本写，**810 GB**。\n4. 优化器状态不用：ZeRO-1 已经在 DP 维切开，每个 rank 持有的是不同的片，没有重复。\nPyTorch DCP 生成保存计划时，把多个 rank 都有的同一个张量只分给一个 rank 写（`DefaultSavePlanner` 的去重）。',
    ref: '/train/fault-tolerance',
  }),
  b({
    id: 'bagu-fault-tolerance-gemini',
    topic: '训练容错',
    q: 'Gemini（SOSP 2023）把 checkpoint 存在 CPU 内存里。机器坏了内存里的 checkpoint 也没了，它怎么还能恢复？它主要缩短了 Young 公式里的哪几项？',
    a: '做法：\n1. 每台机器在自己的 CPU 内存里存一份，再在几台对等机器的内存里各存一份副本。\n2. 坏一台时，换上来的机器从对等机器的内存拉 checkpoint，不读远端存储。\n3. 同一组副本所在的机器全坏了，才退回远端持久存储里较旧的 checkpoint。所以持久存储照样要定期写。\n缩短了：\n1. $\\delta$：写 CPU 内存比写远端存储快得多，能做到每个 iteration 都 checkpoint，丢的工作 $\\tau/2$ 几乎为 0。\n2. $R$ 里的加载段：从内存取回。\n效果（作者自测，DeepSpeed ZeRO-3，A100 / V100 实例）：取回快至多 250×，频率高至多 8×，故障恢复快 13× 以上。',
    ref: '/train/fault-tolerance',
  }),
  b({
    id: 'bagu-fault-tolerance-sdc',
    topic: '训练容错',
    q: '什么是 silent data corruption（SDC）？为什么 NCCL timeout 和心跳查不出来？怎么检测和定位？',
    a: 'SDC：硬件算错了但不报错，没有异常、没有超时，只在 loss 或梯度范数上留下痕迹。NCCL timeout 和心跳只能发现「卡住」和「进程没了」，SDC 两样都不是。\n检测和定位（Gemini 2.5 的做法）：\n1. 发现可疑指标（loss、梯度范数尖峰）就立刻**确定性重放**这一步。\n2. 重放结果和原来不同，说明不是数据问题而是硬件问题。\n3. 对比每个设备的中间结果校验和，找出是哪个设备算错，踢出作业。\n数字（作者自测）：Gemini 2.5 约 0.25% 的步被重放，其中 6% 确实是硬件损坏；Llama 3 的 54 天 419 次非计划中断里有 6 次是 SDC。',
    ref: '/train/fault-tolerance',
  }),
  b({
    id: 'bagu-fault-tolerance-reshard',
    topic: '训练容错',
    q: '用 FSDP 64 卡存的 checkpoint（每个参数沿第 0 维切 64 段），要在 128 卡上接着训。PyTorch DCP 怎么加载？每个新 rank 读多少？',
    a: '1. 保存时：每个 rank 只写自己的分片；全局的 `.metadata` 记下每个张量的全局形状，以及每个分片的偏移和所在文件。\n2. 加载时：每个 rank 按**新**配置算出自己要哪一段（第 0 维的 1/128），到 metadata 里查这一段落在哪个旧分片、哪个文件，只读那几段字节。\n3. 64 → 128：每个新 rank 读**一个旧文件的一半**。反过来 TP 8 → 4，每个新 rank 读两个旧分片拼起来。\n4. 模型和优化器状态用 `get_state_dict` / `set_state_dict` 取和设，屏蔽 FSDP / TP 的差别。\n前提：参数名和全局形状不变，只是切法变了。改了张量布局（比如 QKV 拼在一起）要另写转换脚本。',
    ref: '/train/fault-tolerance',
  }),

  // ---------------- 数据管线 ----------------
  b({
    id: 'bagu-data-pipeline-tokens-per-gpu',
    topic: '数据管线',
    q: '训练 70B 模型（$N = 70.6\\times10^9$），H100 bf16 dense 峰值 989 TFLOP/s，MFU $u = 0.4$。每卡每秒要喂多少 token？token id 用 int32 存，每卡每秒读多少字节？',
    a: '公式：$$\\text{tokens/s/GPU} = \\frac{u \\cdot \\text{peak}}{6N}$$\n1. $6N$：训练每 token 的 FLOPs，forward $2N$（每参数一次乘加），backward $4N$（对输入、对权重各一次矩阵乘）。\n2. 代入：$0.4 \\times 989\\times10^{12} / (6 \\times 70.6\\times10^9) \\approx$ **930 token/s**。\n3. 字节：$930 \\times 4 \\approx$ **3.7 KB/s**。\n结论：带宽不是瓶颈，模型越大需要的 token/s 越少。数据管线的难点是确定性和随机访问。',
    ref: '/train/data-pipeline',
  }),
  b({
    id: 'bagu-data-pipeline-token-dtype',
    topic: '数据管线',
    q: 'Megatron 预处理时按词表大小 $V$ 选 token 的存储类型：$V < 65500$ 用 uint16，否则用 int32。Llama 3 的 $V = 128256$，2T token 的语料 `.bin` 文件多大？换成 Llama 2 的词表（$V = 32000$）呢？',
    a: '1. token id 最大是 $V - 1$。uint16 最大 65535，装不下 128255，所以 Llama 3 用 int32，每 token 4 字节：$2\\times10^{12} \\times 4 =$ **8 TB**。\n2. Llama 2：$V = 32000 < 65500$，uint16，每 token 2 字节：**4 TB**。\n大词表让存储和 page cache 压力翻倍。',
    ref: '/train/data-pipeline',
  }),
  b({
    id: 'bagu-data-pipeline-sample-slice',
    topic: '数据管线',
    q: '预训练把所有文档（末尾加 EOS）拼成一条 token 流，切成序列长度 $L = 4096$ 的样本。第 $i = 3$ 条样本（从 0 数）取 token 流的哪一段？共几个 token？input 和 label 怎么取？',
    a: '1. 第 $i$ 条取 $[iL,\\ iL + L]$ 闭区间：$[12288, 16384]$，共 $L + 1 =$ **4097** 个 token。\n2. input 是前 4096 个，label 是后 4096 个（错一位做 next-token 预测）。\n3. 相邻样本共享边界上的一个 token（16384 既是样本 3 的最后一个 label，也是样本 4 的第一个 input），这样每个 token 恰好被当作 label 一次。',
    ref: '/train/data-pipeline',
  }),
  b({
    id: 'bagu-data-pipeline-epochs',
    topic: '数据混合',
    q: '某数据源权重 $w = 0.05$，总训练 token $D = 2$T，这个源只有 $S = 25$B token。它要被过几遍？公式是什么？',
    a: '$$e = \\frac{w D}{S} = \\frac{0.05 \\times 2\\times10^{12}}{25\\times10^9} =$$ **4 遍**。\n$wD$ 是这个源要贡献的 token 数。Megatron 的 GPTDataset 发现请求的样本超过一遍的量时，就把 epoch 数加到够为止，每个 epoch 用不同的随机顺序。Muennighoff et al. 发现重复 4 遍以内 loss 和全新数据相比变化可以忽略（作者自测），再多就要考虑降权重。',
    ref: '/train/data-pipeline',
  }),
  b({
    id: 'bagu-data-pipeline-blend-greedy',
    topic: '数据混合',
    q: 'Megatron 混合两个源，权重 $w = (0.6, 0.4)$。排第 $j$ 个位置时，对每个源算 $\\text{err}_k = w_k \\cdot \\max(j, 1) - c_k$（$c_k$ 是源 $k$ 已经取了几条），选 err 最大的。写出前 5 个位置分别来自哪个源。',
    a: '1. $j=0$：err $= (0.6, 0.4)$，选 0，$c = (1, 0)$。\n2. $j=1$：$(0.6-1, 0.4) = (-0.4, 0.4)$，选 1，$c = (1, 1)$。\n3. $j=2$：$(1.2-1, 0.8-1) = (0.2, -0.2)$，选 0，$c = (2, 1)$。\n4. $j=3$：$(1.8-2, 1.2-1) = (-0.2, 0.2)$，选 1，$c = (2, 2)$。\n5. $j=4$：$(2.4-2, 1.6-2) = (0.4, -0.4)$，选 0。\n结果 **0 1 0 1 0**，3 比 2。err 是「目标条数减已取条数」，每次补最落后的源，任意前缀的比例都贴近权重。',
    ref: '/train/data-pipeline',
  }),
  b({
    id: 'bagu-data-pipeline-dp-slice',
    topic: '分布式采样',
    q: 'Megatron 的 sampler：从 `consumed_samples` 开始按样本号顺序取，每 $B_\\mu \\cdot d$ 条为一组（$B_\\mu$ 是 micro-batch size，$d$ 是 DP 大小），DP rank $r$ 取组内第 $[rB_\\mu, (r+1)B_\\mu)$ 条。consumed = 1000、$B_\\mu = 4$、$d = 8$ 时，rank 3 接下来两步拿哪些样本号？',
    a: '1. 一组 $4 \\times 8 = 32$ 条。第一组从 1000 开始，rank 3 取组内第 12–15 条：**1012–1015**。\n2. 第二组从 1032 开始：**1044–1047**。\n样本号已经被 shuffle index 打乱过，所以顺序取就是随机的。记的是全局样本号，换 DP 大小重启时全局 batch 的内容不变。',
    ref: '/train/data-pipeline',
  }),
  b({
    id: 'bagu-data-pipeline-tp-pp-cp',
    topic: '分布式采样',
    q: 'Megatron 里一个 DP rank 内有 TP、PP、CP 多张卡。谁跑 dataloader？PP 的第一个 stage、中间 stage、最后一个 stage 各需要哪些张量？CP 组的卡怎么分一条序列？',
    a: '1. **TP**：只有 TP rank 0 读数据，`broadcast` 给组内其他卡。保证数据逐位相同，也省掉重复 IO。\n2. **PP**：第一个 stage 要 `tokens`、`position_ids`；最后一个 stage 要 `labels`、`loss_mask`；中间 stage 什么都不要（只收激活）。\n3. **CP**：CP 组的卡属于同一个 DP rank，拿同一条样本；把序列切成 $2c$ 块（$c$ 是 CP 度），rank $i$ 取第 $i$ 块和第 $2c-1-i$ 块，让因果 mask 下每卡计算量均衡。',
    ref: '/train/data-pipeline',
  }),
  b({
    id: 'bagu-data-pipeline-doc-mask-flops',
    topic: '预训练 packing',
    q: '序列长度 $L = 8192$，一条样本里正好有 4 篇各 2048 token 的文档。屏蔽跨文档 attention（varlen，按 `cu_seqlens` 只在文档内算）后，attention 的 FLOPs 是不屏蔽时的几分之几？',
    a: 'attention 的 FLOPs 正比于（参与计算的 query-key 对数）。都是因果 mask，所以只比平方项：\n$$\\frac{\\sum_j l_j^2}{L^2} = \\frac{4 \\times 2048^2}{8192^2} =$$ **1/4**。\nvarlen kernel 直接跳过跨文档的块，所以不只是结果正确，还真的少算。Llama 3 报告这种 mask 在标准预训练里影响有限、在长序列继续预训练里很重要（作者自测）。',
    ref: '/train/data-pipeline',
  }),
  b({
    id: 'bagu-data-pipeline-skip-batches',
    topic: '可确定性恢复',
    q: '第 20000 步出现 loss spike。按 PaLM 的做法：从 spike 前约 100 步的 checkpoint 重启，跳过约 300 个 batch。global batch size 是 1024 条样本，checkpoint 里存的 `consumed_samples` 应该改成多少？checkpoint 在第 19900 步。',
    a: '1. 第 19900 步的 checkpoint：consumed $= 19900 \\times 1024 = 20{,}377{,}600$。\n2. 跳过 300 个 batch：加 $300 \\times 1024 = 307{,}200$，改成 **20,684,800**。\n前提是样本序列由种子确定（Megatron 的 document / sample / shuffle index），这样同一个 consumed 值永远对应同一批数据。否则既不能复现 spike，也不知道「跳过」跳的是什么。',
    ref: '/train/data-pipeline',
  }),

  // ---------------- SFT 提速 ----------------
  b({
    id: 'bagu-efficient-sft-throughput',
    topic: 'SFT 吞吐',
    q: '全参 SFT 8B 模型（$N = 8.03\\times10^9$），8 张 H100（bf16 dense 峰值 989 TFLOP/s），MFU $u = 0.4$。理论每秒多少 token？开 full gradient checkpointing 呢？',
    a: '公式：$$\\text{tokens/s} = \\frac{n_{\\text{gpu}} \\cdot \\text{peak} \\cdot u}{6N}$$\n1. $6N$：forward $2N$，backward $4N$（dgrad、wgrad 各一次同样大小的矩阵乘）。\n2. 代入：$8 \\times 989\\times10^{12} \\times 0.4 / (6 \\times 8.03\\times10^9) \\approx$ **6.6 万 token/s**。\n3. full recompute 多一次 forward，每 token $8N$：约 **4.9 万 token/s**。\n先算这个上限，再和实测比，得出自己的 MFU。',
    ref: '/train/efficient-sft',
  }),
  b({
    id: 'bagu-efficient-sft-eta',
    topic: 'padding',
    q: '一个 micro-batch 有 4 条样本，长度 200、300、500、2000，不 packing，pad 到组内最长。有效 token 比例 $\\eta$ 是多少？',
    a: '$$\\eta = \\frac{\\sum_i l_i}{B \\cdot l_{\\max}} = \\frac{200+300+500+2000}{4 \\times 2000} = \\frac{3000}{8000} =$$ **37.5%**。\n$l_i$ 是真实长度，$B$ 是样本数，$l_{\\max}$ 是组内最长。62.5% 的算力花在 pad 上：padding 位置的线性层照算，attention mask 只是把分数置 $-\\infty$。packing 后 $\\eta$ 能到 99%。',
    ref: '/train/efficient-sft',
  }),
  b({
    id: 'bagu-efficient-sft-logits-mem',
    topic: 'logits 显存',
    q: 'Llama 3 词表 $V = 128256$，一个 micro-batch 有 $s = 8192$ 个 token。一份 fp32 logits 多大？HF 的 loss 路径前向时同时有 bf16 logits（2 字节）、fp32 副本（4 字节）、log_softmax 输出（4 字节），合计多大？',
    a: '1. 元素数 $s \\cdot V = 8192 \\times 128256 = 1.05\\times10^9$。\n2. fp32 一份：$\\times 4 =$ **4.2 GB**。\n3. 三份合计 $10sV$ 字节：**10.5 GB**。\n对比：8B 开 full recompute 后各层激活只剩 $2shL = 2.1$ GB（$h = 4096$，$L = 32$），logits 成了最大的一块。$V$ 是 $h$ 的 31 倍，所以词表越大越严重。',
    ref: '/train/efficient-sft',
  }),
  b({
    id: 'bagu-efficient-sft-chunk-size',
    topic: 'fused CE',
    q: 'Liger 的 fused linear cross-entropy 沿 token 维分块，块大小 $c = 2^{\\lceil \\log_2 \\lceil s / \\lceil V/h \\rceil \\rceil \\rceil}$。$s = 8192$、$V = 128256$、$h = 4096$ 时 $c$ 是多少？一块 fp32 logits 多大？',
    a: '1. $\\lceil V/h \\rceil = \\lceil 31.3 \\rceil = 32$。\n2. $s / 32 = 256$，已是 2 的幂，$c =$ **256**。\n3. 一块 logits：$256 \\times 128256 \\times 4 \\approx$ **131 MB**。\n设计意图：$c \\cdot V \\approx s \\cdot h$，一块 logits 和整个 micro-batch 的 hidden states 一样大，比完整 logits 小 $\\lceil V/h \\rceil$ 倍。',
    ref: '/train/efficient-sft',
  }),
  b({
    id: 'bagu-efficient-sft-ce-grad',
    topic: 'fused CE',
    q: '一个 token 的 logits 是 $z = (2, 1, 0)$，正确 token 是第 0 个。cross-entropy 对 logits 的梯度 $\\partial \\ell / \\partial z$ 是多少？',
    a: '公式：$$\\frac{\\partial \\ell}{\\partial z} = \\text{softmax}(z) - \\text{onehot}(y)$$\n1. $e^z = (7.389, 2.718, 1)$，和 11.107，softmax $= (0.665, 0.245, 0.090)$。\n2. 减 onehot：**$(-0.335, 0.245, 0.090)$**。\n3. 按 token 平均时再除以全局有效 token 数。\nfused CE 就是在 forward 里当场算出它，乘出 $\\partial x = gW$、$\\partial W \\mathrel{+}= g^\\top x$ 后丢掉这块 logits，backward 不用再做 lm_head 的矩阵乘。',
    ref: '/train/efficient-sft',
  }),
  b({
    id: 'bagu-efficient-sft-gemm-ai',
    topic: '显存换吞吐',
    q: '线性层 GEMM $[s, h] \\times [h, k]$，bf16。FLOPs $2shk$，读写字节约 $2(hk + sh + sk)$。$s \\ll h, k$ 时算术强度约多少？$s = 8192$ 时，相对 H100 的 ridge point 295 FLOP/B 是什么状态？这说明加大 micro-batch 有没有用？',
    a: '1. $s \\ll h, k$ 时字节由权重 $2hk$ 主导，算术强度 $\\approx 2shk / 2hk = s$ FLOP/B。\n2. $s = 8192 \\gg 295$，GEMM 已是 **compute-bound**。\n3. 再加大 micro-batch，每 token 不会更快。它只摊薄每个 micro-batch 的固定开销（FSDP 的参数 all-gather 和梯度 reduce-scatter）。\n所以省下的显存**先关 recompute**（确定省 $8N \\to 6N$），再加 micro-batch。',
    ref: '/train/efficient-sft',
  }),
  b({
    id: 'bagu-efficient-sft-recompute-layer',
    topic: '显存换吞吐',
    q: '8B（$h = 4096$，$L = 32$），micro-batch $s = 8192$ token，当前开 full recompute。每少 checkpoint 一层，激活从每层 $2sh$ 字节变成 $34sh$ 字节。每关一层多占多少显存、省多少计算？有 20 GB 余量能关几层？',
    a: '1. 多占：$(34 - 2) sh = 32 \\times 8192 \\times 4096 \\approx$ **1.07 GB**。\n2. 省计算：full recompute 每 token $8N$，其中重算 forward $2N$ 平摊到 $L$ 层，每层占 $\\frac{1}{L} \\cdot \\frac{2}{8} = \\frac{1}{32} \\times 0.25 \\approx$ **0.8%**。\n3. 20 GB：$20 / 1.07 \\approx$ **18 层**，约省 14% 的计算。\n$34sh$ 是 Korthikanti et al. 的每层激活系数（FlashAttention 下去掉了 $s^2$ 项）。',
    ref: '/train/efficient-sft',
  }),
  b({
    id: 'bagu-efficient-sft-zero2-vs-fsdp',
    topic: '并行选择',
    q: '8B（$N = 8.03\\times10^9$）全参，单机 8 卡，混合精度 Adam 每参数 16 字节（bf16 参数 2 + bf16 梯度 2 + fp32 主权重和 Adam 状态 12）。ZeRO-2 和 ZeRO-3 / FSDP2 每卡模型状态各多少？',
    a: '1. ZeRO-2：参数完整，梯度和优化器状态切 8 份：$2 + 14/8 = 3.75$ 字节/参数，**30.1 GB**。\n2. ZeRO-3 / FSDP2：全部切 8 份：$16/8 = 2$ 字节/参数，**16.1 GB**。\n再加 $s = 8192$ 的激活（不 recompute 36.5 GB）和 logits：ZeRO-2 接近 80 GB 只能开 recompute，FSDP2 不开也放得下。FSDP2 多 50% 通信，单机 NVLink 上能被预取盖住，而 recompute 是 33% 的计算，所以选 FSDP2。',
    ref: '/train/efficient-sft',
  }),
  b({
    id: 'bagu-efficient-sft-lora-flops',
    topic: 'LoRA 吞吐',
    q: '全参训练每 token $6N$ FLOPs（forward、dgrad、wgrad 各 $2N$）。LoRA 冻结底座后每 token 多少？理论加速多少倍？开 full recompute 时呢？',
    a: '1. LoRA 省掉的只有冻结权重的 wgrad（$2N$）。forward 照算；dgrad 也要算，因为梯度必须穿过每一层传回下面的 adapter。每 token **$4N$**，理论 **1.5 倍**。\n2. full recompute 多一次 forward：全参 $8N$、LoRA $6N$，只有 **1.33 倍**。\n3. 实测更少：LoRA 论文在 GPT-3 175B 上训练吞吐 +25%（作者自测）。\nLoRA 主要省模型状态显存（16 字节/参数 → 约 2），激活和 logits 一点没少。',
    ref: '/train/efficient-sft',
  }),

  // ---------------- RL 提速 ----------------
  b({
    id: 'bagu-efficient-rl-gen-weights-term',
    topic: 'RL 提速',
    q: '同步 RL 的 generate 阶段，每个 decode 步都要把全部权重从 HBM 读一遍。模型 $N = 7.6\\times10^9$ 参数、bf16，H100 HBM 带宽 3.35 TB/s，回答最大长度 $\\ell_{\\max} = 16000$、平均长度 $\\bar\\ell = 4000$。只算读权重这一项，generate 至少要多久？如果所有回答都恰好是平均长度呢？',
    a: '读权重项 $= \\ell_{\\max} \\cdot 2N / \\text{BW}$：\n1. 每步读 $2N = 15.2$ GB（每参数 2 字节），和 batch 里有几条序列无关。\n2. 同步 RL 要等最长的那条写完，步数至少 $\\ell_{\\max}$：$16000 \\times 15.2\\ \\text{GB} / 3.35\\ \\text{TB/s} \\approx$ **72.6 s**。\n3. 都是 4000 token 时：$4000 \\times 15.2 / 3350 \\approx$ **18.1 s**。\n多出来的约 54 s 都在等最长的几条，这就是长尾。训练侧各段只看总 token 数，不受长尾影响。',
    ref: '/train/efficient-rl',
  }),
  b({
    id: 'bagu-efficient-rl-logprob-time',
    topic: 'RL 提速',
    q: 'GRPO 一步有 $Bn = 1024$ 条回答，每条 prompt 512 + 回答平均 4000 token。模型 $N = 7.6\\times10^9$，8 张 H100（bf16 dense 峰值 989 TFLOP/s），MFU 取 40%。重算 old log-prob（一次 forward）要多久？actor update 呢？',
    a: '1. 总 token 数 $T = 1024 \\times 4512 = 4.62\\times10^6$。\n2. forward 每 token 每参数一次乘加，$2N$ FLOPs：$2NT = 7.0\\times10^{16}$，除以 $8 \\times 989\\times10^{12} \\times 0.4$，约 **22 s**。\n3. update 是 $6N$：forward 2，backward 4（对输入、对权重各一次同样大小的矩阵乘），所以是 forward 的 3 倍，约 **67 s**。开 gradient checkpointing 时 backward 前多一次 forward，变成 $8N$。',
    ref: '/train/efficient-rl',
  }),
  b({
    id: 'bagu-efficient-rl-grpo-vs-ppo',
    topic: 'RL 提速',
    q: 'PPO 的 critic 和 actor 一样大。训练侧每个 step 要做：old log-prob、ref log-prob、critic values（各一次 forward，$2NT$），critic update 和 actor update（各 $6NT$）。GRPO 去掉 critic 后，训练侧计算量少了百分之几？',
    a: '$N$ 是参数量，$T$ 是总 token 数。\n1. PPO：$2+2+2+6+6 = 18NT$。\n2. GRPO：去掉 values 和 critic update，剩 $2+2+6 = 10NT$。\n3. 少 $8/18 \\approx$ **44%**。\n显存上还少一整套 critic 训练状态（每参数 16 字节）。generate 那段不省：GRPO 靠每题采 $n$ 条来算组内基线。',
    ref: '/train/efficient-rl',
  }),
  b({
    id: 'bagu-efficient-rl-prefix-share',
    topic: 'RL 提速',
    q: 'GRPO 一步 $B = 128$ 个 prompt，每个采 $n = 8$ 条回答，prompt 长 $\\ell_p = 512$。推理引擎开 prefix caching 前后，prefill 一共要算多少 token？',
    a: '1. 不共享：每条回答都把 prompt prefill 一遍，$Bn\\ell_p = 128 \\times 8 \\times 512 =$ **524,288**。\n2. 共享：同一 prompt 的第一条算完 KV，其余 $n-1$ 条命中缓存，KV block 按引用计数共用一份，$B\\ell_p =$ **65,536**，少 $n$ 倍。\nprompt 短时 prefill 本来就不是大头（这里按 $2N$ / token、8 卡、MFU 40% 算只从 2.5 s 降到 0.3 s）；长文档或多轮 agent 时才显著。',
    ref: '/train/efficient-rl',
  }),
  b({
    id: 'bagu-efficient-rl-kv-budget',
    topic: 'RL 提速',
    q: 'vLLM 的 `gpu_memory_utilization` $u$ 是它占总显存的比例。80 GB 的卡，$u = 0.5$，bf16 权重 15.2 GB，每 token KV 57,344 字节（$2 \\times 28 \\text{ 层} \\times 4 \\text{ 个 KV head} \\times 128 \\times 2$ 字节）。每条序列上下文 4512 token 时，最多同时装下多少条（忽略激活和 CUDA graph）？',
    a: 'KV 容量 $\\approx (u \\cdot M - 2N) / k_v$，$M$ 是单卡显存，$2N$ 是权重字节，$k_v$ 是每 token KV 字节。\n1. $(0.5 \\times 80 - 15.2)\\ \\text{GB} = 24.8$ GB。\n2. $24.8\\times10^9 / 57344 \\approx 4.3\\times10^5$ token。\n3. 除以 4512：约 **95** 条。\n如果每卡要同时跑 128 条，就会排队或被抢占重算，decode batch 变小、步数变多。$u = 0.7$ 时约 157 条。',
    ref: '/train/efficient-rl',
  }),
  b({
    id: 'bagu-efficient-rl-colocate-util',
    topic: 'RL 提速',
    q: 'colocate 下 verl 默认不 offload，FSDP 切分后的训练状态（每参数 16 字节）在 generate 期间留在卡上。$N = 7.6\\times10^9$，8 卡 FSDP，单卡 80 GB。vLLM 的 `gpu_memory_utilization` 最多能设多少？',
    a: '1. 每卡训练状态 $16N/G = 16 \\times 7.6\\times10^9 / 8 = 15.2$ GB（$G$ 是 GPU 数；16 字节 = fp32 参数、梯度、Adam $m$、$v$ 各 4）。\n2. 要满足 $u \\cdot M + 16N/G \\le M$：$u \\le (80 - 15.2)/80 \\approx$ **0.81**。\n3. 实际还要给 ref 分片、通信 buffer、CUDA graph 留余量。想再大就开 `optimizer_offload`（挪走 $m$、$v$，每参数 8 字节）或 `param_offload`。',
    ref: '/train/efficient-rl',
  }),
  b({
    id: 'bagu-efficient-rl-dynamic-bsz',
    topic: 'RL 提速',
    q: '一张卡上 8 条序列，prompt 都是 0.5k，回答长度 $[1, 1, 2, 2, 3, 4, 8, 16]$k token。(1) 全部补齐到 16.5k，有多少计算花在 padding 上？(2) 开 verl 的 `use_dynamic_bsz`，`ppo_max_token_len_per_gpu = 33024`，要分几个 micro-batch？',
    a: '1. 真实 token：$8 \\times 0.5 + 37 = 41$k；补齐后 $8 \\times 16.5 = 132$k。padding 占 $1 - 41/132 \\approx$ **69%**。去掉 padding（`use_remove_padding`）后计算只和真实 token 数有关。\n2. micro-batch 数 $= \\lceil 41000 / 33024 \\rceil =$ **2**。按固定条数分时只能按最坏情况定：每条都可能 16.5k，一次只能放 2 条，要分 4 次。\nmicro-batch 少，矩阵更大、FSDP 的参数 all-gather 次数更少。',
    ref: '/train/efficient-rl',
  }),
  b({
    id: 'bagu-efficient-rl-dp-imbalance',
    topic: 'RL 提速',
    q: '4 个 DP rank 做 actor update，每个 rank 分到的总 token 数（千）是 $[5, 10, 20, 51]$。update 的时间由谁决定？效率是多少？比完全均衡慢多少倍？',
    a: '各 rank 在梯度同步处互相等，时间由工作量最大的 rank 决定：$t \\propto \\max_r W_r$，$W_r$ 是第 $r$ 个 rank 的工作量。\n1. 均值 $86/4 = 21.5$，最大 51。\n2. 效率 $= \\text{mean}/\\max = 21.5/51 \\approx$ **42%**。\n3. 比均衡慢 $51/21.5 \\approx$ **2.37 倍**。\nGRPO 下同一题的 $n$ 条长度相近、在 batch 里挨着，按顺序切会让长的扎堆。verl 的 `balance_batch` 用 Karmarkar–Karp 重排，这个例子均衡后约 $[22, 22, 21, 21]$。',
    ref: '/train/efficient-rl',
  }),
  b({
    id: 'bagu-efficient-rl-sleep-level',
    topic: 'RL 提速',
    q: 'colocate RL 里 vLLM sleep mode 的 level 1 和 level 2 分别对权重和 KV cache 做什么？全参 RL 和 LoRA（adapter 形式加载）各该用哪个？',
    a: '1. level 1：权重卸到 CPU 内存，KV cache 丢掉。\n2. level 2：权重和 KV cache 都丢掉。\n3. 全参 RL 用 **level 2**：醒来反正要灌新权重，备份旧的白白多搬 $2N$ 字节过 PCIe。\n4. LoRA 只同步 adapter、底座不变，用 **level 1** 把底座留在 CPU，醒来拷回。\n醒来顺序：先 `wake_up(tags=["weights"])`，灌完新权重再 `wake_up(tags=["kv_cache"])`，避免两者同时占满显存。CUDA graph 的显存默认 sleep 时不释放。',
    ref: '/train/efficient-rl',
  }),
]
