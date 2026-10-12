/**
 * 推理栈分层：0 硬件在最底，8 服务在最顶，每层再拆成几个小主题。
 * 文章用 frontmatter 的 `stack: [..]` 挂上来：
 *   - 字符串 = 小主题 id（如 kv-paged），文章出现在那个格子里
 *   - 数字   = 整层（如 4），用于讲一整层的综述文章
 * 一篇都没挂的小主题在图上显示成占位格。
 */
export interface StackTopic {
  id: string
  name: string
  /** 同层内的分组，如分布式层的「实例内 / 实例间」 */
  group?: string
}

export interface Layer {
  id: number
  name: string
  /** 首页论文风格插图上用的英文名 */
  en: string
  short: string
  scope: string
  /** 新拆出来 / 新增的层，图上会标出来 */
  tag?: string
  /** 图上的强调色：从底层的冷色渐变到顶层的暖色 */
  color: string
  topics: StackTopic[]
}

export const layers: Layer[] = [
  {
    id: 0, name: '硬件', en: 'Hardware', short: '硬件', color: '#0ea5e9',
    scope: 'kernel 层的地基',
    topics: [
      { id: 'hw-gpu', name: 'GPU 架构' },
      { id: 'hw-mem', name: '显存层次' },
      { id: 'hw-link', name: '互联' },
    ],
  },
  {
    id: 1, name: 'bytes → tensor', en: 'Weight Loading', short: '加载', color: '#06b6d4',
    scope: '权重从磁盘到显存',
    topics: [
      { id: 'ld-format', name: '权重格式' },
      { id: 'ld-load', name: '加载（mmap / 分片 / 流式）' },
      { id: 'ld-dtype', name: 'dtype' },
      { id: 'ld-repack', name: '为 kernel 重排权重' },
    ],
  },
  {
    id: 2, name: 'tensor → kernel', en: 'Kernels', short: 'Kernel', color: '#14b8a6',
    scope: '算子实现',
    topics: [
      { id: 'k-gemm', name: 'GEMM' },
      { id: 'k-attn', name: 'attention kernel' },
      { id: 'k-fused', name: 'fused ops' },
      { id: 'k-lang', name: 'Triton / CUDA' },
    ],
  },
  {
    id: 3, name: 'kernel → forward', en: 'Model Execution', short: 'Forward', color: '#22c55e',
    scope: '把 kernel 串成一次前向',
    topics: [
      { id: 'f-model', name: '模型定义' },
      { id: 'f-runner', name: 'model runner' },
      { id: 'f-graph', name: 'CUDA graph / torch.compile' },
    ],
  },
  {
    id: 4, name: '状态与显存', en: 'KV Cache & Memory', short: 'KV/显存', color: '#84cc16', tag: '新增',
    scope: 'KV cache 管理',
    topics: [
      { id: 'kv-paged', name: 'paged' },
      { id: 'kv-prefix', name: 'prefix caching' },
      { id: 'kv-offload', name: 'offload' },
      { id: 'kv-quant', name: 'KV 量化' },
    ],
  },
  {
    id: 5, name: '调度', en: 'Scheduling', short: '调度', color: '#eab308',
    scope: '每一步跑哪些请求',
    topics: [
      { id: 's-cb', name: 'continuous batching' },
      { id: 's-chunked', name: 'chunked prefill' },
      { id: 's-preempt', name: 'preemption' },
      { id: 's-priority', name: '优先级' },
    ],
  },
  {
    id: 6, name: '分布式', en: 'Distributed', short: '分布式', color: '#f97316',
    scope: '一个模型拆到多卡，一群实例管起来',
    topics: [
      { id: 'd-intra', name: 'TP / PP / EP / DP', group: '实例内' },
      { id: 'd-comm', name: '通信', group: '实例内' },
      { id: 'd-pd', name: 'PD 分离', group: '实例间' },
      { id: 'd-route', name: '路由', group: '实例间' },
      { id: 'd-scale', name: '扩缩容', group: '实例间' },
    ],
  },
  {
    id: 7, name: '输出', en: 'Output', short: '输出', color: '#ef4444',
    scope: 'hidden state 到 token',
    topics: [
      { id: 'o-lmhead', name: 'LM head' },
      { id: 'o-logits', name: 'logits processor' },
      { id: 'o-sampling', name: 'sampling' },
      { id: 'o-structured', name: 'structured output' },
      { id: 'o-detok', name: 'detokenize' },
    ],
  },
  {
    id: 8, name: '服务', en: 'Serving', short: '服务', color: '#ec4899', tag: '拆出',
    scope: '请求进出的边界',
    topics: [
      { id: 'sv-api', name: 'API server' },
      { id: 'sv-tokenize', name: 'tokenize / chat template' },
      { id: 'sv-stream', name: '流式' },
      { id: 'sv-sla', name: 'SLA' },
      { id: 'sv-ft', name: '容错' },
    ],
  },
]

export const LAYER_IDS = new Set(layers.map((l) => l.id))
/** 小主题 id → 所在层号 */
export const layerOfTopic = new Map(layers.flatMap((l) => l.topics.map((t) => [t.id, l.id] as const)))

/**
 * 训练栈分层：0 硬件在最底，8 训练阶段在最顶。小主题 id 统一以 tr- 开头，和推理栈不重名。
 * 文章用 frontmatter 的 `train: [..]` 挂上来（不用 `stack`，免得影响侧边栏按推理栈归组），
 * 规则同上：字符串 = 小主题 id，数字 = 整层。
 */
export const trainLayers: Layer[] = [
  {
    id: 0, name: '硬件', en: 'Hardware', short: '硬件', color: '#0ea5e9',
    scope: '算力、互联、存储',
    topics: [
      { id: 'tr-hw-gpu', name: 'GPU 算力与精度' },
      { id: 'tr-hw-link', name: '互联（NVLink / IB）' },
      { id: 'tr-hw-io', name: '存储与 IO' },
    ],
  },
  {
    id: 1, name: 'tensor → kernel', en: 'Kernels', short: 'Kernel', color: '#06b6d4',
    scope: 'forward 和 backward 的算子',
    topics: [
      { id: 'tr-k-gemm', name: 'GEMM / FP8 GEMM' },
      { id: 'tr-k-attn', name: 'attention fwd / bwd' },
      { id: 'tr-k-fused', name: '融合算子 / loss kernel' },
      { id: 'tr-k-compile', name: 'torch.compile' },
    ],
  },
  {
    id: 2, name: '一个 step', en: 'Training Step', short: 'Step', color: '#14b8a6',
    scope: 'forward → backward → 更新',
    topics: [
      { id: 'tr-st-autograd', name: 'autograd' },
      { id: 'tr-st-opt', name: '优化器' },
      { id: 'tr-st-lr', name: 'LR / batch 调度' },
      { id: 'tr-st-accum', name: '梯度累积与裁剪' },
    ],
  },
  {
    id: 3, name: '显存与精度', en: 'Memory & Precision', short: '显存', color: '#22c55e',
    scope: '一张卡上放什么、用什么精度',
    topics: [
      { id: 'tr-m-account', name: '显存账' },
      { id: 'tr-m-recompute', name: 'recompute' },
      { id: 'tr-m-precision', name: '混合精度 / FP8' },
      { id: 'tr-m-offload', name: 'offload' },
    ],
  },
  {
    id: 4, name: '并行切分', en: 'Parallelism', short: '并行', color: '#84cc16',
    scope: '模型和数据怎么切到多卡',
    topics: [
      { id: 'tr-p-dp', name: 'DP / ZeRO / FSDP' },
      { id: 'tr-p-tp', name: 'TP / SP' },
      { id: 'tr-p-pp', name: 'PP' },
      { id: 'tr-p-cp', name: 'CP' },
      { id: 'tr-p-ep', name: 'EP' },
    ],
  },
  {
    id: 5, name: '通信与配置', en: 'Communication', short: '通信', color: '#eab308',
    scope: '切完以后怎么通信、怎么组合',
    topics: [
      { id: 'tr-c-coll', name: '集合通信' },
      { id: 'tr-c-overlap', name: '计算通信 overlap' },
      { id: 'tr-c-config', name: '并行配置 / device mesh' },
    ],
  },
  {
    id: 6, name: '数据', en: 'Data', short: '数据', color: '#f97316',
    scope: '从语料到每个 rank 的 batch',
    topics: [
      { id: 'tr-d-format', name: 'token 存储格式' },
      { id: 'tr-d-mix', name: '混合与采样' },
      { id: 'tr-d-pack', name: 'packing / mask' },
      { id: 'tr-d-loader', name: 'dataloader 与恢复' },
    ],
  },
  {
    id: 7, name: '作业', en: 'Job', short: '作业', color: '#ef4444',
    scope: '跑多久、跑得稳、坏了怎么办',
    topics: [
      { id: 'tr-j-plan', name: '算力规划 / MFU' },
      { id: 'tr-j-stab', name: '稳定性' },
      { id: 'tr-j-ft', name: '容错 / checkpoint' },
      { id: 'tr-j-debug', name: '排障' },
    ],
  },
  {
    id: 8, name: '训练阶段', en: 'Recipes', short: '阶段', color: '#ec4899',
    scope: '预训练和后训练各自的特殊问题',
    topics: [
      { id: 'tr-r-long', name: '长上下文', group: '预训练' },
      { id: 'tr-r-moe', name: 'MoE', group: '预训练' },
      { id: 'tr-r-sft', name: 'SFT / LoRA', group: '后训练' },
      { id: 'tr-r-rlalgo', name: 'RL 算法', group: '后训练' },
      { id: 'tr-r-rlsys', name: 'RL 系统', group: '后训练' },
    ],
  },
]

export const TRAIN_LAYER_IDS = new Set(trainLayers.map((l) => l.id))
export const trainLayerOfTopic = new Map(trainLayers.flatMap((l) => l.topics.map((t) => [t.id, l.id] as const)))

/** 两张分层图：推理栈挂在 frontmatter 的 `stack`，训练栈挂在 `train` */
export type StackId = 'infer' | 'train'
export interface StackDef {
  id: StackId
  name: string
  /** 图所在的页面，独立页的面包屑链回这里 */
  home: string
  layers: Layer[]
  /** 整层综述的 key，也是 docs/stack/ 下层总述页的文件名 */
  layerKey: (n: number) => string
  /** 层号前缀：推理栈 L4，训练栈 T4 */
  prefix: string
}
export const stacks: Record<StackId, StackDef> = {
  infer: { id: 'infer', name: '推理栈', home: '/', layers, layerKey: (n) => `layer-${n}`, prefix: 'L' },
  train: { id: 'train', name: '训练栈', home: '/train/', layers: trainLayers, layerKey: (n) => `train-layer-${n}`, prefix: 'T' },
}

/**
 * 分层图上的一个格子 key（小主题 id，或 layer-N / train-layer-N）→ 它在哪张图、哪一层。
 * 两张图的小主题 id 不重名，所以 key 在全站唯一。
 */
export function locateKey(key: string): { stack: StackDef; layer: Layer; topic?: StackTopic } | undefined {
  for (const s of Object.values(stacks)) {
    for (const l of s.layers) {
      if (s.layerKey(l.id) === key) return { stack: s, layer: l }
      const t = l.topics.find((x) => x.id === key)
      if (t) return { stack: s, layer: l, topic: t }
    }
  }
  return undefined
}

/** 一篇文章的 frontmatter（stack / train 两个字段）→ 它挂在哪些格子上 */
export function keysOfArticle(t: { stack: (string | number)[]; train: (string | number)[] }): string[] {
  return [
    ...t.stack.map((s) => (typeof s === 'number' ? stacks.infer.layerKey(s) : s)),
    ...t.train.map((s) => (typeof s === 'number' ? stacks.train.layerKey(s) : s)),
  ]
}
