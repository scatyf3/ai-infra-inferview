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
    q: 'decode 的算术强度怎么推？batch = 1 和 64 时各是多少，和 H100 的 ridge 比呢？',
    a: '先算 ridge：峰值算力 ÷ 显存带宽。H100 SXM 的 bf16 是 989 TFLOP/s、3.35 TB/s：\n$$\\text{ridge} = \\frac{989 \\times 10^{12}}{3.35 \\times 10^{12}} \\approx 295 \\ \\text{FLOP/B}$$\n再算 decode 的 AI。符号：$P$ 参数量，$B$ batch，$b_w$ 每个参数的字节数（bf16 = 2）。decode 每步每个序列只算 1 个新 token：\n1. FLOPs ≈ $2PB$：每个权重对每个 token 一次乘加。\n2. 访存 ≈ $P b_w$：权重整读一遍，KV 先忽略。\n3. $AI = \\dfrac{2PB}{P b_w} = B$（bf16）。\n代入：B = 1 时 AI = 1，B = 64 时 AI = 64，都远小于 295，**memory-bound**。要 B 接近 295 才到 ridge，算上读 KV 还更难。',
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
    id: 'bagu-how-many-gpus',
    topic: '显存账',
    q: '算一下：Llama-3-70B bf16 跑在 80 GB 的 H100 上，每卡留 4 GB 给激活和固定开销。TP = 2 / 4 / 8 时各能放多少 token 的 KV？要同时服务 32 个 8k 上下文的请求，最少几张卡？',
    a: '公式：KV 能放的 token 数 = (N × (单卡显存 − 开销) − 权重) ÷ KV/token。\n1. 权重 70 × 10⁹ × 2 B = 140 GB；KV/token = 2 × 80 × 8 × 128 × 2 B = 320 KiB（见 KV/token 那张卡）。\n2. TP = 2：2 × 76 − 140 = 12 GB，约 3.7 万 token，只够 4 个 8k 请求。\n3. TP = 4：4 × 76 − 140 = 164 GB，约 50 万 token，61 个。\n4. 32 × 8192 = 26 万 token 的 KV 约 86 GB，加权重 140 GB = 226 GB，÷ 76 ≈ 3 张，TP 要整除 head 数，取 **4 张**。\n规律：权重是常数，激活和开销每卡固定，**只有 KV 随并发 × 长度线性涨**，所以「要几张卡」由 KV 定。代码可以直接改参数跑。',
    code: lines`
      GB = 1e9
      P, b_w = 70e9, 2                   # 参数量, 每参数字节 (bf16)
      L, H_kv, d_h, b = 80, 8, 128, 2    # 层数, KV head 数, head 维度, KV 每元素字节
      kv_per_token = 2 * L * H_kv * d_h * b   # K 和 V 各一份: 327,680 B

      def kv_tokens(n_gpu, mem=80 * GB, overhead=4 * GB):
          free = n_gpu * (mem - overhead) - P * b_w   # 权重按 TP 切到 n_gpu 张卡上
          return int(free // kv_per_token)

      for n in (2, 4, 8):
          t = kv_tokens(n)
          print(f"TP={n}: {t:,} token = {t // 8192} 个 8k 请求")
      # TP=2: 36,621 token = 4 个 8k 请求
      # TP=4: 500,488 token = 61 个 8k 请求
      # TP=8: 1,428,222 token = 174 个 8k 请求

      need = 32 * 8192 * kv_per_token + P * b_w      # 32 个 8k 请求 + 权重
      print(need / GB, need / (76 * GB))             # 226 GB, 2.97 -> 取 TP=4
    `,
    ref: `${MEM}#交互`,
  }),
  b({
    id: 'bagu-layer-params',
    topic: '显存账',
    q: '用 d、H_kv、d_h、d_ff 写出一层 decoder（GQA + SwiGLU）的参数量公式，并用 Llama-3-70B 验算。常见算错点是什么？',
    a: '符号：d hidden 维度，H / H_kv 是 Q / KV head 数，d_h head 维度（H·d_h = d），d_ff MLP 中间维度，V 词表大小。按矩阵一个个数（见图）：\n1. attention：W_q、W_o 各 $d^2$；W_k、W_v 各 $d \\cdot H_{kv} d_h$。\n2. MLP（SwiGLU）：gate、up、down 三个矩阵，各 $d \\cdot d_{ff}$。\n3. 两个 RMSNorm：各 d。\n每层 $= 2d^2 + 2 d H_{kv} d_h + 3 d\\, d_{ff} + 2d$。模型再加 embedding 和 lm_head 各 $Vd$（共享就只算一次）。\n算错点：**SwiGLU 是三个矩阵**，不是两个；GQA 下 K、V 投影比 Q 小。\n验算 Llama-3-70B（d = 8192，H_kv = 8，d_h = 128，d_ff = 28672，80 层，V = 128256）：每层约 0.86 B，80 层 68.4 B，加 embedding 和 lm_head 2.1 B，共约 **70.6 B**。',
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
    q: 'MLA 缓存什么？decode 时怎么避免把 K 展开？',
    a: '符号（DeepSeek-V3）：$h$ 是 token 的 hidden（7168 维），128 个 head，每个 head 128 维；latent 维度 $d_c = 512$，RoPE 部分 $d_r = 64$，61 层。\n缓存什么：\n1. 下投影 $c_{kv} = W_{DKV} h$，7168 → 512 维，**缓存它**。\n2. 位置部分 $k^R = \\text{RoPE}(W_{KR} h)$，64 维，所有 head 共享，**也缓存**。\n3. 用的时候再上投影：第 i 个 head 的 $k_i = W_{UK,i}\\, c_{kv}$，$v_i = W_{UV,i}\\, c_{kv}$。\n每 token 每层 512 + 64 = 576 个数，61 层、bf16 共 **68.6 KiB**；同样 61 层、8 个 KV head 的 GQA 是 244 KiB。\n怎么不展开 K（weight absorption）：\n$$q_i^\\top k_i = q_i^\\top W_{UK,i}\\, c_{kv} = (W_{UK,i}^\\top q_i)^\\top c_{kv}$$\n先把 q 投到 512 维的 latent 空间，直接和缓存的 $c_{kv}$ 点积。V 同理：先对 $c_{kv}$ 加权求和，最后再乘 $W_{UV,i}$，它还能并进 $W_O$。\n效果：decode 时相当于 128 个 Q head 共用**一个 576 维的 KV head**，像 MQA。点积维度从 192（128 + 64）变成 576，算得更多，但 decode 是 memory-bound，拿算力换带宽划算。prefill 是 compute-bound，反而展开成每 head 128 维更快。',
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
    q: 'FlashAttention v2、v3 分别改了什么？',
    a: '**v2**：调换循环顺序，外层遍历 Q 块（每个 warp 负责一段 Q 行），K/V 在内层流过；减少 shared memory 同步和非 matmul 的 FLOPs，约快 2 倍。\n**v3**：针对 Hopper，用 TMA 异步搬数据、wgmma、warp specialization 让 softmax 和 GEMM 流水重叠，并支持 FP8。',
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
    q: '投机采样怎么保证输出分布和 target 单独采样完全一样？',
    a: '符号：draft 模型给 token x 的概率 $q(x)$，target 给的 $p(x)$。对 draft 猜的每个位置依次：\n1. 以概率 $\\min\\left(1, \\frac{p(x)}{q(x)}\\right)$ 接受 draft 的 x。\n2. 拒绝时从残差分布重新采样，后面的 draft token 全部作废：\n$$\\tilde x \\sim \\frac{\\max(0,\\ p - q)}{\\sum_y \\max(0,\\ p(y) - q(y))}$$\n3. k 个全接受时，target 在第 k+1 个位置的分布已经算出来了，顺手再采一个。\n为什么严格等于 p：拒绝的总概率正好是 $\\sum_y \\max(0, p(y) - q(y))$，代进去\n$$P(x) = \\min(p(x), q(x)) + \\max(0,\\ p(x) - q(x)) = p(x)$$\n前一项是 draft 猜到 x 且被接受，后一项是拒绝后补采到 x。\n例（见图）：draft 猜 A 的概率 0.6，接受率 0.3 / 0.6 = 0.5；拒绝的总概率 0.3，按 B : C = 0.2 : 0.1 重采。每个 token 最后的概率都等于 p。',
    fig: lines`
      token          A     B     C
      q (draft)     0.6   0.3   0.1
      p (target)    0.3   0.5   0.2
      accepted      0.3   0.3   0.1   min(p,q)
      resampled     0     0.2   0.1   max(0,p-q)
      total         0.3   0.5   0.2   = p
    `,
    ref: SPEC,
  }),
  b({
    id: 'bagu-spec-expected',
    topic: '投机解码',
    q: '接受率 α、猜 k 个，一轮平均产出几个 token？k 越大越好吗？',
    a: '假设每个 draft token 独立地以概率 α 被接受。前 i 个都被接受的概率是 $\\alpha^i$；不管停在哪，target 都会再补 1 个 token（拒绝位置的重采样，或全接受后的第 k+1 个）：\n$$E[\\text{tokens}] = \\sum_{i=0}^{k} \\alpha^i = \\frac{1 - \\alpha^{k+1}}{1 - \\alpha}$$\n算一下：α = 0.8、k = 4 时 $(1 - 0.8^5) / 0.2 \\approx 3.4$；α = 0.5 时约 1.9。\n加速比还要除以一轮的成本 $ck + 1$（c 是 draft 一步和 target 一步的耗时比）：\n$$\\text{speedup} = \\frac{1 - \\alpha^{k+1}}{(1 - \\alpha)(ck + 1)}$$\nk 大了 $\\alpha^k$ 越来越小、分母线性涨，典型最优 k 在 **3–5**；α 低、draft 又不够便宜时净收益为负。',
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
    q: 'TTFT、TPOT、ITL、E2E 分别是什么？各自主要受什么影响？',
    a: '**TTFT**：第一个 token 返回的时间，受排队和 prefill 影响。\n**TPOT**：后续每个 token 的平均间隔，受 decode 每步时间（batch 大小、读多少字节）影响。\n**ITL**：单个 token 间隔，看 P99 抖动，受混进来的 prefill 影响。\n**E2E** = TTFT + TPOT × 输出长度。',
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
    q: '什么是 warp divergence？causal attention 为什么不怎么受影响？',
    a: '同一 warp 的 32 个线程共用一条指令流。线程走不同分支时，硬件先让走 if 的线程执行（其余空等），再让走 else 的执行，两条路径**串行**，吞吐减半（例子见代码）。\ncausal attention 影响小，因为 mask 按 tile 处理（见图）：大多数 tile 要么全可见、要么全被 mask（整块跳过），这个判断按 tile 做，整个 warp 走同一路。只有对角线上的 tile 要逐元素 mask，而且用 `where(mask, s, -inf)` 这样的选择指令，不用分支。',
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
    code: lines`
      // divergence：同一 warp 里奇数、偶数线程走不同分支，两条路径串行
      if (threadIdx.x % 2 == 0) a(); else b();

      // 没有 divergence：按 warp 分界，同一 warp 的线程走同一路
      if ((threadIdx.x / 32) % 2 == 0) a(); else b();
    `,
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
    q: 'torch.compile 的三段分别做什么？什么是 graph break？',
    a: '**Dynamo**：在 CPython 的 frame evaluation 钩子上符号执行字节码，抓出 FX 图，记下 guard，下次 guard 命中直接跑编译好的图。\n**AOTAutograd**：把前向和反向一起 trace 成 ATen 算子图。\n**Inductor**：融合 pointwise / reduction，生成 Triton（GPU）或 C++（CPU）kernel。\n**graph break**：遇到数据相关的控制流（如 `if x.sum() > 0`）、print、不支持的调用时切断图，前后各编一段；太多就退化成 eager。',
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
    q: '混合精度 Adam 训练，每个参数占多少显存？7B 全参微调要多少？',
    a: '每个参数要存 5 样东西（见图）：\n1. bf16 参数 w（2 B）：前向、反向用的权重。\n2. bf16 梯度 g（2 B）：反向算出的 $\\partial L / \\partial w$。\n3. fp32 主权重（4 B）：优化器在它上面更新，更新完再 cast 成 bf16 的 w（为什么见 bf16 那张卡）。\n4. Adam 的 m 和 v（各 4 B）：每个参数各有一个，更新公式：\n$$m \\leftarrow \\beta_1 m + (1 - \\beta_1) g,\\quad v \\leftarrow \\beta_2 v + (1 - \\beta_2) g^2,\\quad w \\leftarrow w - \\eta \\frac{m}{\\sqrt{v} + \\epsilon}$$\n合计 **16 字节 / 参数**。7B：7 × 10⁹ × 16 = 112 GB，超过一张 80 GB 的 H100，这还没算激活。',
    fig: lines`
      per parameter   bytes   dtype
      w               2       bf16
      grad            2       bf16
      master w        4       fp32
      Adam m          4       fp32
      Adam v          4       fp32
      total           16
    `,
    ref: TMEM,
  }),
  b({
    id: 'bagu-float-formats',
    topic: 'Post-train',
    q: 'fp32、fp16、bf16、fp8（e4m3、e5m2）各几位指数、几位尾数？指数位和尾数位各决定什么？',
    a: '位数见图。\n1. **指数位定范围**：fp16 只有 5 位，最大 65504，最小的正规数约 6 × 10⁻⁵；bf16 和 fp32 都是 8 位，范围约 10⁻³⁸ 到 3 × 10³⁸。\n2. **尾数位定精度**：相邻两个数的相对间隔约 $2^{-m}$（m 是尾数位数）。fp16 约 0.001，bf16 约 0.008，只有两三位有效数字。\n3. fp8：e4m3 精度高、范围小（最大 448），用于前向的权重和激活；e5m2 范围大（最大 57344），用于梯度。',
    fig: lines`
              sign   exp   mantissa
      fp32     1      8      23
      fp16     1      5      10
      bf16     1      8       7
      e4m3     1      4       3
      e5m2     1      5       2
    `,
    ref: TMEM,
  }),
  b({
    id: 'bagu-bf16-fp16',
    topic: 'Post-train',
    q: '为什么 fp16 训练要 loss scaling，bf16 不用？那 bf16 为什么还要 fp32 主权重？',
    a: '位数见「浮点格式」那张卡：fp16 是 5 位指数 + 10 位尾数，bf16 是 8 位指数 + 7 位尾数。\n1. fp16 要 loss scaling：指数只有 5 位，小于约 6 × 10⁻⁸ 的数直接变成 0，小梯度会下溢。先把 loss 乘一个大数 S，梯度跟着放大 S 倍，更新前再除回去。\n2. bf16 不用：8 位指数，范围和 fp32 一样，梯度不会下溢。\n3. bf16 还要 fp32 主权重：尾数只有 7 位，1 附近相邻两个数差 $2^{-7} \\approx 0.008$。权重 w = 1.0、更新量 $\\eta g = 10^{-4}$，直接在 bf16 里算 $1.0 - 10^{-4}$ 会被舍入回 1.0，这次更新就丢了。所以在 fp32 主权重上累加更新，前向时再 cast 成 bf16。',
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
    a: 'packing 把多条短样本拼成一条定长序列，GPU 利用率从三到五成提到九成以上。要注意三件事：\n1. **attention 不能跨样本**：用 block-diagonal 的因果 mask（见图），或者 varlen attention 传 `cu_seqlens`，否则后面的样本会 attend 到前面样本的 token。\n2. **位置编码按样本重置**：position_ids 每个样本从 0 开始。\n3. **loss mask**：prompt、system、用户轮的 label 设成 −100（PyTorch 的 cross entropy 默认忽略它），只在 assistant 的回答上算 loss。',
    fig: lines`
      3 samples packed:  a a a | b b | c c c
      1 = can attend

            a a a b b c c c
         a  1 . . . . . . .
         a  1 1 . . . . . .
         a  1 1 1 . . . . .
         b  . . . 1 . . . .
         b  . . . 1 1 . . .
         c  . . . . . 1 . .
         c  . . . . . 1 1 .
         c  . . . . . 1 1 1

      position_ids = 0 1 2 0 1 0 1 2
      cu_seqlens   = [0, 3, 5, 8]
    `,
    code: lines`
      seg = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])   # 每个 token 属于第几个样本
      i = torch.arange(8)[:, None]                   # (8, 1) query 下标
      j = torch.arange(8)[None, :]                   # (1, 8) key 下标
      allowed = (seg[:, None] == seg[None, :]) & (j <= i)   # 同一样本 且 因果
      attn = attn.masked_fill(~allowed, float('-inf'))
    `,
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
