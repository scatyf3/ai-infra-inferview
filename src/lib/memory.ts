import { bytesOf } from './dtypes'
import { GiB, KiB, MiB } from './types'
import type { DType, GpuSpec, ModelConfig, Workload } from './types'

export function headDim(cfg: ModelConfig): number {
  return cfg.headDim ?? cfg.hidden / cfg.heads
}

/** 参数量拆分：attention / mlp / embedding。MoE 的 mlp 包含全部 expert。 */
export function paramBreakdown(cfg: ModelConfig) {
  const d = cfg.hidden
  const hd = headDim(cfg)
  const qo = 2 * d * cfg.heads * hd
  const kv = 2 * d * cfg.kvHeads * hd
  const attnPerLayer = qo + kv + 2 * d // + norms
  const mlpDense = 3 * d * cfg.ffn // SwiGLU: gate, up, down
  const experts = (cfg.numExperts ?? 0) + (cfg.sharedExperts ?? 0)
  const mlpPerLayer = experts > 0 ? experts * mlpDense : mlpDense
  const embedding = cfg.vocab * d * (cfg.tiedEmbeddings ? 1 : 2)
  const attention = attnPerLayer * cfg.layers
  const mlp = mlpPerLayer * cfg.layers
  return { attention, mlp, embedding, total: attention + mlp + embedding }
}

export function estimateParams(cfg: ModelConfig): number {
  return cfg.params ?? paramBreakdown(cfg).total
}

/** decode 一步实际参与计算的参数（MoE 只算激活的 expert） */
export function activeParams(cfg: ModelConfig): number {
  if (cfg.activeParams) return cfg.activeParams
  if (cfg.params) {
    if (!cfg.numExperts) return cfg.params
    // 按比例缩放 mlp 部分
    const b = paramBreakdown(cfg)
    const activeFrac = ((cfg.topK ?? 1) + (cfg.sharedExperts ?? 0)) / (cfg.numExperts + (cfg.sharedExperts ?? 0))
    const scaled = b.attention + b.embedding + b.mlp * activeFrac
    return (cfg.params * scaled) / b.total
  }
  const b = paramBreakdown(cfg)
  if (!cfg.numExperts) return b.total
  const activeFrac = ((cfg.topK ?? 1) + (cfg.sharedExperts ?? 0)) / (cfg.numExperts + (cfg.sharedExperts ?? 0))
  return b.attention + b.embedding + b.mlp * activeFrac
}

export function weightBytes(cfg: ModelConfig, dtype: DType): number {
  return estimateParams(cfg) * bytesOf(dtype)
}

/** 每个 token 的 KV cache 字节数（所有层） */
export function kvBytesPerToken(cfg: ModelConfig, dtype: DType): number {
  const b = bytesOf(dtype)
  if (cfg.attention === 'mla') {
    const latent = (cfg.mlaLatentDim ?? 512) + (cfg.mlaRopeDim ?? 64)
    return cfg.layers * latent * b
  }
  const kvHeads = cfg.attention === 'mqa' ? 1 : cfg.kvHeads
  return 2 * cfg.layers * kvHeads * headDim(cfg) * b
}

export function kvBytes(cfg: ModelConfig, dtype: DType, tokens: number): number {
  return kvBytesPerToken(cfg, dtype) * tokens
}

/**
 * 推理激活显存（近似）：单层活跃张量 B·S_chunk·(4d + 2·d_ff)·b（FlashAttention 无 S² 项），
 * 加上最后一个 token 的 fp32 logits B·V·4。
 */
export function activationBytes(cfg: ModelConfig, batch: number, chunkTokens: number, dtype: DType = 'bf16'): number {
  const b = bytesOf(dtype)
  const perLayer = batch * chunkTokens * (4 * cfg.hidden + 2 * cfg.ffn) * b
  const logits = batch * cfg.vocab * 4
  return perLayer + logits
}

export interface MemoryOptions {
  gpu: GpuSpec
  /** vLLM 语义的 gpu_memory_utilization */
  utilization?: number
  /** CUDA graph / workspace / NCCL 等固定开销 */
  overheadBytes?: number
  /** prefill chunk 大小，用于激活估算 */
  chunkTokens?: number
}

export interface MemoryBreakdown {
  params: number
  weights: number
  kvPerToken: number
  kv: number
  activations: number
  overhead: number
  total: number
  perGpuBudget: number
  gpusNeeded: number
  minTP: number
  perGpuAtMinTP: number
  warnings: string[]
}

export function memoryBreakdown(cfg: ModelConfig, wl: Workload, opts: MemoryOptions): MemoryBreakdown {
  const utilization = opts.utilization ?? 0.9
  const overhead = opts.overheadBytes ?? 2 * GiB
  const chunk = Math.min(opts.chunkTokens ?? 2048, wl.context)
  const params = estimateParams(cfg)
  const weights = weightBytes(cfg, wl.weightDtype)
  const kvPerToken = kvBytesPerToken(cfg, wl.kvDtype)
  const kv = kvPerToken * wl.batch * wl.context
  const activations = activationBytes(cfg, wl.batch, chunk, wl.activationDtype ?? 'bf16')
  const total = weights + kv + activations + overhead
  const perGpuBudget = opts.gpu.memoryGiB * GiB * utilization
  const gpusNeeded = Math.ceil(total / perGpuBudget)

  // 最小的 2 的幂 TP，使得 per-GPU 占用 <= budget。激活和 overhead 不随 TP 缩减（保守）。
  let minTP = 1
  let perGpuAtMinTP = total
  while (minTP <= 64) {
    perGpuAtMinTP = weights / minTP + kv / minTP + activations + overhead
    if (perGpuAtMinTP <= perGpuBudget) break
    minTP *= 2
  }
  const warnings: string[] = []
  if (minTP > 64) warnings.push('TP=64 仍放不下，需要 PP 或更大显存')
  const kvHeads = cfg.attention === 'mqa' ? 1 : cfg.kvHeads
  if (minTP > 1 && cfg.attention !== 'mla' && kvHeads % minTP !== 0) {
    warnings.push(`kv_heads=${kvHeads} 不能被 TP=${minTP} 整除，KV heads 会被复制，每卡 KV 不再是 1/TP`)
  }
  if (kv > weights) warnings.push('KV cache 已超过权重大小，decode 访存主要来自 KV，考虑 GQA/MLA/KV 量化或缩短 context')

  return { params, weights, kvPerToken, kv, activations, overhead, total, perGpuBudget, gpusNeeded, minTP, perGpuAtMinTP, warnings }
}

export function formatBytes(n: number, digits = 1): string {
  if (n >= GiB) return `${(n / GiB).toFixed(digits)} GiB`
  if (n >= MiB) return `${(n / MiB).toFixed(digits)} MiB`
  if (n >= KiB) return `${(n / KiB).toFixed(digits)} KiB`
  return `${n.toFixed(0)} B`
}

export function formatNumber(n: number, digits = 1): string {
  const abs = Math.abs(n)
  if (abs >= 1e12) return `${(n / 1e12).toFixed(digits)}T`
  if (abs >= 1e9) return `${(n / 1e9).toFixed(digits)}B`
  if (abs >= 1e6) return `${(n / 1e6).toFixed(digits)}M`
  if (abs >= 1e3) return `${(n / 1e3).toFixed(digits)}K`
  return n.toFixed(digits)
}
