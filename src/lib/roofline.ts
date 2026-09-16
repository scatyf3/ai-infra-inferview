import { bytesOf } from './dtypes'
import { activationBytes, activeParams, estimateParams, kvBytes } from './memory'
import type { DType, GpuSpec, ModelConfig, Phase, Workload } from './types'

export type BoundSide = 'compute' | 'memory'

/** GPU 峰值算力（FLOP/s），按权重 dtype 选 bf16 或 fp8 */
export function peakFlops(gpu: GpuSpec, dtype: DType): number {
  const useFp8 = (dtype === 'fp8' || dtype === 'int8') && gpu.fp8TFLOPS
  return (useFp8 ? gpu.fp8TFLOPS! : gpu.bf16TFLOPS) * 1e12
}

export function bandwidth(gpu: GpuSpec): number {
  return gpu.bandwidthTBs * 1e12
}

/** ridge point = peak / bandwidth，单位 FLOP/Byte */
export function ridgePoint(gpu: GpuSpec, dtype: DType = 'bf16'): number {
  return peakFlops(gpu, dtype) / bandwidth(gpu)
}

/** 每层 attention 的 FLOPs：QK^T 与 PV 各 2·S_q·S_kv·d */
function attentionFlops(cfg: ModelConfig, batch: number, sq: number, skv: number): number {
  return 4 * cfg.layers * cfg.hidden * sq * skv * batch
}

/** prefill 总 FLOPs ≈ 2·P·B·S（+ 可选 attention 项 4·L·d·S²·B） */
export function prefillFlops(cfg: ModelConfig, batch: number, tokens: number, includeAttention = true): number {
  const linear = 2 * activeParams(cfg) * batch * tokens
  return linear + (includeAttention ? attentionFlops(cfg, batch, tokens, tokens) : 0)
}

/** decode 一步 FLOPs ≈ 2·P·B（+ 4·L·d·S_ctx·B） */
export function decodeFlopsPerStep(cfg: ModelConfig, batch: number, context: number, includeAttention = true): number {
  const linear = 2 * activeParams(cfg) * batch
  return linear + (includeAttention ? attentionFlops(cfg, batch, 1, context) : 0)
}

/** prefill 访存 ≈ 权重 + 激活（一次读写） */
export function bytesMovedPrefill(cfg: ModelConfig, wl: Workload): number {
  const w = estimateParams(cfg) * bytesOf(wl.weightDtype)
  const act = activationBytes(cfg, wl.batch, wl.context, wl.activationDtype ?? 'bf16') * cfg.layers
  return w + act
}

/** decode 一步访存 ≈ 权重 + 全部 KV cache */
export function bytesMovedDecode(cfg: ModelConfig, wl: Workload): number {
  const w = activeParams(cfg) * bytesOf(wl.weightDtype)
  return w + kvBytes(cfg, wl.kvDtype, wl.batch * wl.context)
}

export function arithmeticIntensity(flops: number, bytes: number): number {
  return flops / bytes
}

export function boundSide(ai: number, ridge: number): BoundSide {
  return ai >= ridge ? 'compute' : 'memory'
}

/** T = max(bytes / BW, FLOPs / (peak · MFU)) */
export function estimateTime(flops: number, bytes: number, gpu: GpuSpec, dtype: DType, mfu = 0.5): number {
  return Math.max(bytes / bandwidth(gpu), flops / (peakFlops(gpu, dtype) * mfu))
}

export interface PhaseAnalysis {
  phase: Phase
  flops: number
  bytes: number
  ai: number
  side: BoundSide
  /** 秒 */
  time: number
  /** 该 phase 达到的有效 FLOP/s */
  attainedFlops: number
}

export interface RooflineAnalysis {
  ridge: number
  peak: number
  bw: number
  prefill: PhaseAnalysis
  decode: PhaseAnalysis
}

export interface RooflineOptions {
  mfu?: number
  includeAttention?: boolean
  /** 参与计算的 GPU 数（TP）：权重/KV/FLOPs 均按 1/n 分摊，AI 不变 */
  tp?: number
}

export function rooflineAnalysis(cfg: ModelConfig, wl: Workload, gpu: GpuSpec, opts: RooflineOptions = {}): RooflineAnalysis {
  const mfu = opts.mfu ?? 0.5
  const inc = opts.includeAttention ?? true
  const tp = opts.tp ?? 1
  const ridge = ridgePoint(gpu, wl.weightDtype)
  const peak = peakFlops(gpu, wl.weightDtype)
  const bw = bandwidth(gpu)

  const mk = (phase: Phase, flops: number, bytes: number): PhaseAnalysis => {
    const f = flops / tp
    const b = bytes / tp
    const ai = arithmeticIntensity(f, b)
    const time = estimateTime(f, b, gpu, wl.weightDtype, mfu)
    return { phase, flops: f, bytes: b, ai, side: boundSide(ai, ridge), time, attainedFlops: f / time }
  }

  return {
    ridge,
    peak,
    bw,
    prefill: mk('prefill', prefillFlops(cfg, wl.batch, wl.context, inc), bytesMovedPrefill(cfg, wl)),
    decode: mk('decode', decodeFlopsPerStep(cfg, wl.batch, wl.context, inc), bytesMovedDecode(cfg, wl)),
  }
}
