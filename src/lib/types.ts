export type DType = 'fp32' | 'bf16' | 'fp16' | 'fp8' | 'int8' | 'int4'
export type AttentionKind = 'mha' | 'mqa' | 'gqa' | 'mla'
export type Phase = 'prefill' | 'decode'

export interface ModelConfig {
  id?: string
  name?: string
  /** 总参数量；缺省时用 estimateParams 估算 */
  params?: number
  /** decode 每 token 实际参与计算的参数量（MoE）；缺省按 topK 比例估算 */
  activeParams?: number
  layers: number
  hidden: number
  heads: number
  kvHeads: number
  /** 缺省 hidden / heads */
  headDim?: number
  /** MLP 中间维度（MoE 时为单个 expert 的中间维度） */
  ffn: number
  vocab: number
  attention: AttentionKind
  tiedEmbeddings?: boolean
  /** MLA：压缩后的 KV latent 维度与 decoupled RoPE 维度 */
  mlaLatentDim?: number
  mlaRopeDim?: number
  /** MoE */
  numExperts?: number
  topK?: number
  sharedExperts?: number
}

export interface Workload {
  batch: number
  /** 每个序列的 token 数（prompt + generated） */
  context: number
  weightDtype: DType
  kvDtype: DType
  activationDtype?: DType
}

export interface GpuSpec {
  id: string
  name: string
  memoryGiB: number
  bandwidthTBs: number
  bf16TFLOPS: number
  fp8TFLOPS: number | null
  interconnect: 'NVLink' | 'PCIe'
  interconnectGBs: number
}

export const GiB = 1024 ** 3
export const MiB = 1024 ** 2
export const KiB = 1024
