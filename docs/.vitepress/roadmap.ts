/**
 * 学习路线：按阶段把分层图（layers.ts）的小主题串起来。
 * 每个小主题恰好属于一个阶段（单测会检查），articles 是该阶段建议读 / 手撕的文章（不含 base 的绝对路径）。
 */
export interface Stage {
  id: number
  title: string
  subtitle: string
  goal: string
  /** layers.ts 里的小主题 id */
  topics: string[]
  articles: string[]
  /** 过关标准：能做到这些就可以进下一阶段 */
  checks: string[]
}

export const stages: Stage[] = [
  {
    id: 1,
    title: '一次前向怎么算',
    subtitle: '模型定义 + roofline',
    goal: '搞清楚一个 token 从 embedding 到 logits 经过了哪些计算：每一步的 shape、FLOPs 和访存量，以及为什么 decode 是 memory-bound。kernel 只到概念层，手写 kernel 放到 Stage 6。',
    topics: ['f-model', 'o-lmhead', 'k-attn', 'hw-mem'],
    articles: [
      '/handson/mha-gqa-forward',
      '/inference/attention-variants',
      '/handson/stable-softmax',
      '/handson/online-softmax',
      '/handson/rmsnorm',
      '/inference/flash-attention',
      '/inference/prefill-decode-roofline',
    ],
    checks: [
      '白板写出 GQA forward，讲清每一步的 shape',
      '写出数值稳定的 softmax 和 online softmax，讲清 FlashAttention 为什么省的是 HBM 访问',
      '口算 prefill 和 decode 的算术强度，并放到 roofline 上',
    ],
  },
  {
    id: 2,
    title: '单卡推理引擎',
    subtitle: 'KV cache + 调度',
    goal: '理解引擎怎么同时服务很多请求：KV cache 怎么存，每一步跑哪些请求，一次 forward 的输入怎么组织。',
    topics: ['kv-paged', 'kv-prefix', 's-cb', 's-chunked', 's-preempt', 'f-runner', 'f-graph', 'o-sampling'],
    articles: [
      '/handson/decode-step-kv-cache',
      '/inference/kv-cache-paged-attention',
      '/inference/batching-scheduling',
      '/framework/request-lifecycle',
      '/framework/vllm-v1-architecture',
      '/gpu/cuda-graph-fusion',
      '/handson/top-p-sampling',
    ],
    checks: [
      '手写带 KV cache 的 decode step',
      '讲清 PagedAttention 解决了什么问题、block table 怎么用',
      '讲清 continuous batching 和 chunked prefill 各自改善了哪个指标',
      '从 HTTP 请求进来，一路讲到第一个 token 出去',
    ],
  },
  {
    id: 3,
    title: '省显存与提速',
    subtitle: '量化 · KV 压缩 · 投机解码',
    goal: '显存容量和带宽是 decode 的硬约束。这一阶段学怎么用更少的字节、更少的步数做同样的事。',
    topics: ['ld-dtype', 'ld-format', 'ld-repack', 'kv-quant', 'kv-offload'],
    articles: [
      '/inference/memory-accounting',
      '/inference/quantization',
      '/inference/speculative-decoding',
      '/posttrain/distill-prune-sparse',
    ],
    checks: [
      '口算 70B 模型在给定 context 和 batch 下要几张卡',
      '比较 W4A16 和 W8A8 分别适合什么场景',
      '讲清 speculative decoding 为什么能保证输出分布不变',
    ],
  },
  {
    id: 4,
    title: '多卡与集群',
    subtitle: '并行 · 通信 · PD 分离',
    goal: '模型放不下一张卡、或者一个实例扛不住流量时怎么办：切模型的几种方式和各自的代价，以及多个实例怎么分工。',
    topics: ['hw-link', 'd-intra', 'd-comm', 'd-pd', 'd-route', 'd-scale', 'ld-load'],
    articles: [
      '/parallel/parallelism-overview',
      '/parallel/megatron-tp',
      '/parallel/collective-comm',
      '/parallel/comm-overlap',
      '/parallel/moe-ep',
    ],
    checks: [
      '推导 Megatron TP 为什么每层只需要两次 all-reduce',
      '算出 ring all-reduce 每张卡的通信量',
      '讲清 MoE 的 EP 通信模式和专家负载不均的问题',
      '说明 PD 分离的收益和 KV 传输的代价',
    ],
  },
  {
    id: 5,
    title: '服务化',
    subtitle: '输出 · 接口 · SLA',
    goal: '把引擎变成可靠的服务：输出怎么处理，接口怎么设计，延迟目标怎么定义和度量。最后收在系统设计题上。',
    topics: ['o-logits', 'o-structured', 'o-detok', 's-priority', 'sv-api', 'sv-tokenize', 'sv-stream', 'sv-sla', 'sv-ft'],
    articles: [
      '/framework/serving-layer',
      '/inference/metrics-benchmark',
      '/handson/beam-search',
      '/basics/python',
      '/basics/system-design-llm-serving',
    ],
    checks: [
      '定义 TTFT / TPOT / goodput，并设计一次压测',
      '讲清 structured output 怎么在每一步约束 token',
      '完整回答「设计一个 LLM 推理服务」',
    ],
  },
  {
    id: 6,
    title: 'Kernel 加深',
    subtitle: '选修 · 投 kernel 岗再刷',
    goal: '够用即可，别装深。前五个阶段过完、或者目标岗位明确要写 kernel 时再来：GPU 执行模型、GEMM tiling，以及 Triton / CUDA 手写。',
    topics: ['hw-gpu', 'k-gemm', 'k-fused', 'k-lang'],
    articles: [
      '/gpu/gpu-architecture',
      '/gpu/tensor-core-gemm',
      '/gpu/triton',
      '/handson/triton-softmax',
      '/handson/triton-fused-layernorm',
      '/handson/cuda-reduce',
      '/handson/cuda-tiled-matmul',
      '/gpu/profiling',
    ],
    checks: [
      '讲清 coalescing、bank conflict、occupancy，以及 GEMV 为什么打不满 Tensor Core',
      '用 Triton 写一个 fused softmax 或 RMSNorm',
      '手写 CUDA reduce 和 tiled matmul',
    ],
  },
]
