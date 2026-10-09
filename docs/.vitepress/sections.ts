/**
 * 侧边栏分组：按首页的推理栈分层组织。只影响侧边栏，文件不动、URL 不变。
 *
 * 归组规则（见 sidebar.ts）：
 *   1. overrides 里写了的，按它来（跨层的文章在这里指定）；
 *   2. 非推理栈目录（posttrain / basics / handson / leetgpu）整个目录一组；
 *   3. 其余按 frontmatter `stack` 的第一项归到对应的层（小主题 id 或层号）；
 *   4. 没有 stack 的放进「系统全景」。
 * 新写的文章只要 `stack` 标对，就会自动出现在对应的层里。
 */
import { layers } from './layers'

export interface Section {
  id: string
  label: string
  /** 组标题的链接：层链到 /stack/layer-N 总述页，目录组链到目录首页 */
  link?: string
}

export const sections: Section[] = [
  { id: 'account', label: '先算账：roofline / 显存 / 指标' },
  ...layers.map((l) => ({ id: `L${l.id}`, label: `L${l.id} ${l.name}`, link: `/stack/layer-${l.id}` })),
  { id: 'overview', label: '系统全景：把各层串起来' },
  { id: 'train', label: '训练与 Post-train', link: '/posttrain/' },
  { id: 'basics', label: '基础不能挂', link: '/basics/' },
  { id: 'handson', label: '手撕高频', link: '/handson/' },
  { id: 'leetgpu', label: 'LeetGPU 题解', link: '/leetgpu/' },
]

/** 不属于推理栈的目录：整个目录一组 */
export const dirSection: Record<string, string> = {
  posttrain: 'train',
  basics: 'basics',
  handson: 'handson',
  leetgpu: 'leetgpu',
}

/** 不按 `stack` 第一项归组的文章（路径相对 docs/，不带 .md） */
export const overrides: Record<string, string> = {
  'inference/prefill-decode-roofline': 'account',
  'inference/memory-accounting': 'account',
  'inference/metrics-benchmark': 'account',
  'inference/quantization-gptq': 'L1', // 跟着量化
  'inference/speculative-sampling-math': 'L3', // 跟着 speculative decoding
  'inference/omni-serving': 'overview',
  'gpu/cuda-graph-fusion': 'L2', // 主体是 kernel 融合
  'framework/add-model-vllm-sglang': 'L3',
  'framework/request-lifecycle': 'overview',
  'framework/vllm-v1-architecture': 'overview',
  'framework/vllm-release-history': 'overview',
  'parallel/zero-fsdp': 'train', // ZeRO / FSDP 是训练侧
  'basics/system-design-llm-serving': 'overview',
}

/**
 * 组内顺序：列出的按这里排（从下往上、由浅入深），没列出的排在后面（按目录、frontmatter order）。
 * 也可以列入不在任何目录里的页面（比如 /stack/ 下展开写过的小主题页）。
 */
export const sectionOrder: Record<string, string[]> = {
  account: ['inference/prefill-decode-roofline', 'inference/memory-accounting', 'inference/metrics-benchmark'],
  L0: ['gpu/gpu-architecture', 'stack/hw-mem'],
  L1: ['inference/quantization', 'inference/quantization-gptq'],
  L2: ['gpu/tensor-core-gemm', 'inference/flash-attention', 'gpu/cuda-graph-fusion', 'gpu/triton', 'gpu/profiling'],
  L3: [
    'framework/pytorch-internals',
    'framework/torch-compile',
    'framework/cuda-graph',
    'inference/attention-variants',
    'inference/speculative-decoding',
    'inference/speculative-sampling-math',
    'framework/add-model-vllm-sglang',
  ],
  L6: ['parallel/parallelism-overview', 'parallel/collective-comm', 'parallel/megatron-tp', 'parallel/moe-ep', 'parallel/comm-overlap'],
  overview: [
    'framework/request-lifecycle',
    'framework/vllm-v1-architecture',
    'basics/system-design-llm-serving',
    'inference/omni-serving',
    'framework/vllm-release-history',
  ],
}
