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
