import { describe, expect, it } from 'vitest'
import { findModel } from '@lib/gpus'
import { ringAllReduceBytes, sharding, zeroMemory } from '@lib/parallel'
import { weightBytes } from '@lib/memory'

const llama70b = findModel('llama3-70b')
const base = { batch: 8, tokens: 1024, weightDtype: 'bf16' as const, kvDtype: 'bf16' as const }

describe('parallel', () => {
  it('ring all-reduce per-GPU bytes = 2(N-1)/N · size', () => {
    expect(ringAllReduceBytes(1000, 4)).toBeCloseTo(1500)
    expect(ringAllReduceBytes(1000, 1)).toBe(0)
  })

  it('TP=8 splits weights ~8x and emits 2 all-reduces per layer', () => {
    const r = sharding({ cfg: llama70b, par: { tp: 8, pp: 1, dp: 1, ep: 1 }, ...base })
    expect(r.perGpuWeightBytes).toBeCloseTo(weightBytes(llama70b, 'bf16') / 8, -6)
    const tp = r.comm.find((c) => c.kind === 'TP')!
    expect(tp.primitive).toBe('all-reduce')
    expect(tp.timesPerLayer).toBe(2)
    expect(tp.messageBytes).toBe(8 * 1024 * 8192 * 2)
  })

  it('PP=4 gives 20 layers per GPU and bubble (p-1)/m', () => {
    const r = sharding({ cfg: llama70b, par: { tp: 1, pp: 4, dp: 1, ep: 1, microBatches: 8 }, ...base })
    expect(r.layersPerGpu).toBe(20)
    expect(r.ppBubble).toBeCloseTo(3 / 8)
  })

  it('warns when TP exceeds kv_heads', () => {
    const r = sharding({ cfg: llama70b, par: { tp: 16, pp: 1, dp: 1, ep: 1 }, ...base })
    expect(r.warnings.length).toBeGreaterThan(0)
  })

  it('EP on MoE emits 2 all-to-alls per layer', () => {
    const ds = findModel('deepseek-v3')
    const r = sharding({ cfg: ds, par: { tp: 1, pp: 1, dp: 1, ep: 8 }, ...base })
    const ep = r.comm.find((c) => c.kind === 'EP')!
    expect(ep.primitive).toBe('all-to-all')
    expect(ep.timesPerLayer).toBe(2)
  })

  it('ZeRO-3 divides all 16 B/param by dp', () => {
    expect(zeroMemory(1e9, 8, 3).bytesPerParamPerGpu).toBe(2)
    expect(zeroMemory(1e9, 8, 0).bytesPerParamPerGpu).toBe(16)
    expect(zeroMemory(1e9, 8, 1).bytesPerParamPerGpu).toBe(2 + 2 + 12 / 8)
  })
})
