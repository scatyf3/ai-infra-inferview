import type { AttentionKind, Phase } from './types'

/** 符号维度；数值代入时用 Dims 映射 */
export type Dim = 'B' | 'S' | 'S_kv' | 'd' | 'H' | 'H_kv' | 'h_d' | 'd_ff' | 'd_c' | 'd_r' | '1'
export type Shape = Dim[]

export interface Dims {
  B: number
  S: number
  S_kv: number
  d: number
  H: number
  H_kv: number
  h_d: number
  d_ff: number
  d_c: number
  d_r: number
}

export interface ShapeStep {
  id: string
  op: string
  inputs: Shape[]
  output: Shape
  note: string
  /** 该步是否读写 KV cache */
  cache?: 'read' | 'write'
  /** 分组：attention / mlp */
  group: 'attn' | 'mlp'
}

export interface ShapeFlowInput {
  variant: AttentionKind
  phase: Phase
}

export function numel(shape: Shape, dims: Dims): number {
  return shape.reduce((acc, d) => acc * (d === '1' ? 1 : dims[d]), 1)
}

export function formatShape(shape: Shape, dims?: Dims): string {
  return `[${shape.map((d) => (dims ? (d === '1' ? 1 : dims[d]) : d)).join(', ')}]`
}

export function shapeFlow({ variant, phase }: ShapeFlowInput): ShapeStep[] {
  const S: Dim = phase === 'prefill' ? 'S' : '1'
  const Skv: Dim = phase === 'prefill' ? 'S' : 'S_kv'
  const kvH: Dim = variant === 'mha' ? 'H' : 'H_kv'
  const steps: ShapeStep[] = []
  const attn = (s: Omit<ShapeStep, 'group'>) => steps.push({ ...s, group: 'attn' })
  const mlp = (s: Omit<ShapeStep, 'group'>) => steps.push({ ...s, group: 'mlp' })

  attn({ id: 'x', op: 'input x', inputs: [], output: ['B', S, 'd'], note: phase === 'prefill' ? '整段 prompt 一起进' : '只有当前一个 token' })
  attn({ id: 'q', op: 'Q = x·Wq', inputs: [['B', S, 'd'], ['d', 'H', 'h_d']], output: ['B', 'H', S, 'h_d'], note: 'reshape + transpose 到 head 维' })

  if (variant === 'mla') {
    attn({ id: 'ckv', op: 'c_kv = x·W_dkv', inputs: [['B', S, 'd'], ['d', 'd_c']], output: ['B', S, 'd_c'], note: '压缩 latent，这是唯一要缓存的东西（外加 k_pe）', cache: 'write' })
    attn({ id: 'kpe', op: 'k_pe = RoPE(x·W_kr)', inputs: [['B', S, 'd'], ['d', 'd_r']], output: ['B', S, 'd_r'], note: 'decoupled RoPE key，所有 head 共享', cache: 'write' })
    if (phase === 'decode') {
      attn({ id: 'ckv-read', op: 'read c_kv cache', inputs: [], output: ['B', 'S_kv', 'd_c'], note: '每 token 只有 d_c + d_r 个元素，比 GQA 的 2·H_kv·h_d 小得多', cache: 'read' })
      attn({ id: 'qabs', op: 'q_c = Q·W_uk^T (weight absorption)', inputs: [['B', 'H', '1', 'h_d'], ['H', 'h_d', 'd_c']], output: ['B', 'H', '1', 'd_c'], note: '把 W_uk 吸收进 Q，直接在 latent 空间算 score，避免展开 K' })
      attn({ id: 'scores', op: 'scores = q_c·c_kvᵀ + q_pe·k_peᵀ', inputs: [['B', 'H', '1', 'd_c'], ['B', 'S_kv', 'd_c']], output: ['B', 'H', '1', 'S_kv'], note: 'GEMV-ish：每 head 一个 [1, d_c]×[d_c, S_kv]，memory-bound' })
      attn({ id: 'softmax', op: 'softmax', inputs: [['B', 'H', '1', 'S_kv']], output: ['B', 'H', '1', 'S_kv'], note: '沿 S_kv 归一化，数值稳定要减 max' })
      attn({ id: 'pv', op: 'o_c = P·c_kv', inputs: [['B', 'H', '1', 'S_kv'], ['B', 'S_kv', 'd_c']], output: ['B', 'H', '1', 'd_c'], note: '仍在 latent 空间' })
      attn({ id: 'up', op: 'o = o_c·W_uv', inputs: [['B', 'H', '1', 'd_c'], ['H', 'd_c', 'h_d']], output: ['B', 'H', '1', 'h_d'], note: '升回 head 维' })
    } else {
      attn({ id: 'k', op: 'K = c_kv·W_uk', inputs: [['B', 'S', 'd_c'], ['d_c', 'H', 'h_d']], output: ['B', 'H', 'S', 'h_d'], note: 'prefill 时直接展开 K/V 走标准 attention（compute-bound，展开无妨）' })
      attn({ id: 'v', op: 'V = c_kv·W_uv', inputs: [['B', 'S', 'd_c'], ['d_c', 'H', 'h_d']], output: ['B', 'H', 'S', 'h_d'], note: '' })
      attn({ id: 'scores', op: 'scores = Q·Kᵀ / √h_d', inputs: [['B', 'H', 'S', 'h_d'], ['B', 'H', 'S', 'h_d']], output: ['B', 'H', 'S', 'S'], note: 'FlashAttention 不会物化这个 S×S 矩阵' })
      attn({ id: 'softmax', op: 'causal mask + softmax', inputs: [['B', 'H', 'S', 'S']], output: ['B', 'H', 'S', 'S'], note: '' })
      attn({ id: 'pv', op: 'O = P·V', inputs: [['B', 'H', 'S', 'S'], ['B', 'H', 'S', 'h_d']], output: ['B', 'H', 'S', 'h_d'], note: '' })
    }
  } else {
    attn({ id: 'k', op: 'K = x·Wk', inputs: [['B', S, 'd'], ['d', kvH, 'h_d']], output: ['B', kvH, S, 'h_d'], note: variant === 'mha' ? 'H_kv = H' : variant === 'mqa' ? 'H_kv = 1，所有 Q head 共享一份 K' : `H_kv < H，每 H/H_kv 个 Q head 共享一组 KV`, cache: 'write' })
    attn({ id: 'v', op: 'V = x·Wv', inputs: [['B', S, 'd'], ['d', kvH, 'h_d']], output: ['B', kvH, S, 'h_d'], note: '', cache: 'write' })
    if (phase === 'decode') {
      attn({ id: 'kv-read', op: 'concat KV cache', inputs: [['B', kvH, 'S_kv', 'h_d']], output: ['B', kvH, 'S_kv', 'h_d'], note: 'decode 每步都要把整个 KV cache 读一遍：这就是 decode memory-bound 的来源', cache: 'read' })
    }
    if (variant !== 'mha') {
      attn({ id: 'repeat', op: 'repeat_interleave K,V → H heads', inputs: [['B', kvH, Skv, 'h_d']], output: ['B', 'H', Skv, 'h_d'], note: '逻辑上广播；好的 kernel 不真的复制，而是让 H/H_kv 个 Q head 读同一块' })
    }
    attn({ id: 'scores', op: 'scores = Q·Kᵀ / √h_d', inputs: [['B', 'H', S, 'h_d'], ['B', 'H', Skv, 'h_d']], output: ['B', 'H', S, Skv], note: phase === 'prefill' ? 'S×S，FlashAttention 分块不物化' : '每 head 是 [1, h_d]×[h_d, S_kv] 的 GEMV，算力打不满' })
    attn({ id: 'softmax', op: (phase === 'prefill' ? 'causal mask + ' : '') + 'softmax', inputs: [['B', 'H', S, Skv]], output: ['B', 'H', S, Skv], note: '沿最后一维' })
    attn({ id: 'pv', op: 'O = P·V', inputs: [['B', 'H', S, Skv], ['B', 'H', Skv, 'h_d']], output: ['B', 'H', S, 'h_d'], note: '' })
  }

  attn({ id: 'merge', op: 'merge heads', inputs: [['B', 'H', S, 'h_d']], output: ['B', S, 'd'], note: 'transpose + reshape，H·h_d = d' })
  attn({ id: 'o', op: 'out = O·Wo', inputs: [['B', S, 'd'], ['d', 'd']], output: ['B', S, 'd'], note: 'TP 下 Wo 是 row-parallel，这一步之后 all-reduce' })
  mlp({ id: 'norm', op: 'RMSNorm(x + attn)', inputs: [['B', S, 'd']], output: ['B', S, 'd'], note: 'residual + pre-norm' })
  mlp({ id: 'gate', op: 'g = x·W_gate', inputs: [['B', S, 'd'], ['d', 'd_ff']], output: ['B', S, 'd_ff'], note: 'TP 下 column-parallel，无通信' })
  mlp({ id: 'up', op: 'u = x·W_up', inputs: [['B', S, 'd'], ['d', 'd_ff']], output: ['B', S, 'd_ff'], note: '通常和 gate 融成一个 GEMM [d, 2·d_ff]' })
  mlp({ id: 'act', op: 'h = SiLU(g) ⊙ u', inputs: [['B', S, 'd_ff'], ['B', S, 'd_ff']], output: ['B', S, 'd_ff'], note: 'elementwise，典型 fusion 对象' })
  mlp({ id: 'down', op: 'y = h·W_down', inputs: [['B', S, 'd_ff'], ['d_ff', 'd']], output: ['B', S, 'd'], note: 'TP 下 row-parallel，之后第二次 all-reduce' })
  return steps
}

/** 该配置下 KV cache 每 token 每层的元素数（用于对比 variant） */
export function kvElemsPerTokenPerLayer(variant: AttentionKind, dims: Dims): number {
  if (variant === 'mla') return dims.d_c + dims.d_r
  if (variant === 'mha') return 2 * dims.H * dims.h_d
  if (variant === 'mqa') return 2 * dims.h_d
  return 2 * dims.H_kv * dims.h_d
}
