import { bytesOf } from './dtypes'
import { kvBytesPerToken, paramBreakdown } from './memory'
import type { DType, ModelConfig } from './types'

export interface ParallelConfig {
  tp: number
  pp: number
  dp: number
  ep: number
  /** PP 的 micro-batch 数，用于 bubble 估算 */
  microBatches?: number
}

export type Primitive = 'all-reduce' | 'all-gather' | 'reduce-scatter' | 'all-to-all' | 'p2p' | 'none'

export interface CommItem {
  kind: 'TP' | 'PP' | 'DP' | 'EP' | 'SP'
  primitive: Primitive
  /** 每次通信的消息大小（字节，逻辑大小） */
  messageBytes: number
  /** 每层（或每个 stage 边界）发生次数 */
  timesPerLayer: number
  /** 参与的 GPU 数 */
  groupSize: number
  /** ring 算法下每个 GPU 实际发送字节 = 2(N-1)/N · size（all-reduce）等 */
  perGpuBytesPerLayer: number
  where: string
}

export interface ShardingResult {
  /** 每卡权重字节 */
  perGpuWeightBytes: number
  perGpuAttentionBytes: number
  perGpuMlpBytes: number
  perGpuEmbeddingBytes: number
  /** 每卡 KV cache 字节（给定 tokens） */
  perGpuKvBytes: number
  /** 每卡负责的层数 */
  layersPerGpu: number
  totalGpus: number
  /** PP bubble 比例 (pp-1)/m */
  ppBubble: number
  comm: CommItem[]
  warnings: string[]
}

export interface ShardingInput {
  cfg: ModelConfig
  par: ParallelConfig
  batch: number
  /** 每序列 token 数（用于激活通信量与 KV） */
  tokens: number
  weightDtype: DType
  kvDtype: DType
  activationDtype?: DType
}

/** ring all-reduce 每 GPU 发送量 */
export function ringAllReduceBytes(size: number, n: number): number {
  return n <= 1 ? 0 : (2 * (n - 1) / n) * size
}
export function ringAllGatherBytes(size: number, n: number): number {
  return n <= 1 ? 0 : ((n - 1) / n) * size
}

export function sharding(input: ShardingInput): ShardingResult {
  const { cfg, par, batch, tokens } = input
  const { tp, pp, dp, ep } = par
  const wb = bytesOf(input.weightDtype)
  const ab = bytesOf(input.activationDtype ?? 'bf16')
  const b = paramBreakdown(cfg)
  const warnings: string[] = []
  const isMoe = (cfg.numExperts ?? 0) > 0

  // 权重切分：attention 按 TP·PP；MLP dense 按 TP·PP，MoE expert 按 EP·PP（EP 组内再按 TP 切 expert 内部矩阵）
  const perGpuAttentionBytes = (b.attention * wb) / (tp * pp)
  const perGpuMlpBytes = isMoe ? (b.mlp * wb) / (ep * tp * pp) : (b.mlp * wb) / (tp * pp)
  const perGpuEmbeddingBytes = (b.embedding * wb) / tp // embedding 一般在首尾 stage，按 TP 切 vocab
  const perGpuWeightBytes = perGpuAttentionBytes + perGpuMlpBytes + perGpuEmbeddingBytes

  const kvHeads = cfg.attention === 'mqa' ? 1 : cfg.kvHeads
  let kvShardFactor = tp
  if (cfg.attention !== 'mla' && kvHeads < tp) {
    kvShardFactor = kvHeads
    warnings.push(`kv_heads=${kvHeads} < TP=${tp}，KV heads 会跨卡复制，KV 只能切 ${kvHeads} 份`)
  } else if (cfg.attention !== 'mla' && kvHeads % tp !== 0) {
    warnings.push(`kv_heads=${kvHeads} 不能被 TP=${tp} 整除`)
  }
  const perGpuKvBytes = (kvBytesPerToken(cfg, input.kvDtype) * batch * tokens) / (kvShardFactor * pp)

  if (cfg.layers % pp !== 0) warnings.push(`layers=${cfg.layers} 不能被 PP=${pp} 整除，stage 不均衡`)
  if (isMoe && (cfg.numExperts ?? 0) % ep !== 0) warnings.push(`experts=${cfg.numExperts} 不能被 EP=${ep} 整除`)
  if (!isMoe && ep > 1) warnings.push('dense 模型没有 expert，EP 无意义')

  const actBytes = batch * tokens * cfg.hidden * ab // 一层激活 [B, S, d]
  const comm: CommItem[] = []

  if (tp > 1) {
    comm.push({
      kind: 'TP',
      primitive: 'all-reduce',
      messageBytes: actBytes,
      timesPerLayer: 2,
      groupSize: tp,
      perGpuBytesPerLayer: 2 * ringAllReduceBytes(actBytes, tp),
      where: 'attention 输出 (row-parallel Wo 之后) 与 MLP 输出 (down-proj 之后)，各一次',
    })
  }
  if (pp > 1) {
    comm.push({
      kind: 'PP',
      primitive: 'p2p',
      messageBytes: actBytes / (par.microBatches ?? 1),
      timesPerLayer: 1,
      groupSize: 2,
      perGpuBytesPerLayer: actBytes / (par.microBatches ?? 1),
      where: '每个 stage 边界，把 micro-batch 的激活 [B_micro, S, d] 发给下一 stage（只在边界层，不是每层）',
    })
  }
  if (isMoe && ep > 1) {
    const routed = batch * tokens * (cfg.topK ?? 1) * cfg.hidden * ab
    comm.push({
      kind: 'EP',
      primitive: 'all-to-all',
      messageBytes: routed,
      timesPerLayer: 2,
      groupSize: ep,
      perGpuBytesPerLayer: 2 * routed * ((ep - 1) / ep),
      where: 'MoE 层 dispatch（token 发往 expert 所在卡）与 combine（结果收回），各一次',
    })
  }
  if (dp > 1) {
    comm.push({
      kind: 'DP',
      primitive: 'none',
      messageBytes: 0,
      timesPerLayer: 0,
      groupSize: dp,
      perGpuBytesPerLayer: 0,
      where: '推理时 DP 副本之间无通信（训练时每 step 梯度 all-reduce = 每卡权重大小）',
    })
  }

  const layersPerGpu = cfg.layers / pp
  const m = par.microBatches ?? 1
  return {
    perGpuWeightBytes,
    perGpuAttentionBytes,
    perGpuMlpBytes,
    perGpuEmbeddingBytes,
    perGpuKvBytes,
    layersPerGpu,
    totalGpus: tp * pp * dp * (isMoe ? ep : 1),
    ppBubble: pp > 1 ? (pp - 1) / m : 0,
    comm,
    warnings,
  }
}

/** ZeRO / FSDP 训练态显存：每参数字节数 (权重 + 梯度 + 优化器状态) 按 stage 切分 */
export interface ZeroResult {
  stage: 0 | 1 | 2 | 3
  bytesPerParamPerGpu: number
  perGpuBytes: number
  commPerStep: string
}
export function zeroMemory(params: number, dp: number, stage: 0 | 1 | 2 | 3): ZeroResult {
  // 混合精度：bf16 权重 2 + bf16 梯度 2 + fp32 master 4 + Adam m/v 8 = 16 B/param
  const w = 2, g = 2, os = 12
  let per: number
  let comm: string
  switch (stage) {
    case 0: per = w + g + os; comm = '梯度 all-reduce（2·(N-1)/N · grad）'; break
    case 1: per = w + g + os / dp; comm = '梯度 reduce-scatter + 权重 all-gather（总量同 all-reduce）'; break
    case 2: per = w + g / dp + os / dp; comm = '同 stage 1'; break
    case 3: per = (w + g + os) / dp; comm = '每层前向/反向各 all-gather 权重一次，反向 reduce-scatter 梯度（约 1.5× all-reduce）'; break
  }
  return { stage, bytesPerParamPerGpu: per, perGpuBytes: per * params, commPerStep: comm }
}
