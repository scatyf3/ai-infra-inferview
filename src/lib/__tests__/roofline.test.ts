import { describe, expect, it } from 'vitest'
import { findGpu, findModel } from '@lib/gpus'
import { decodeFlopsPerStep, prefillFlops, ridgePoint, rooflineAnalysis } from '@lib/roofline'

const llama70b = findModel('llama3-70b')
const h100 = findGpu('h100-sxm')

describe('roofline', () => {
  it('H100 bf16 ridge point ≈ 295 FLOP/B', () => {
    expect(ridgePoint(h100, 'bf16')).toBeCloseTo(295.2, 0)
  })

  it('prefill FLOPs ≈ 2·P·tokens without attention term', () => {
    const f = prefillFlops(llama70b, 1, 1000, false)
    expect(f).toBeCloseTo(2 * 70.55e9 * 1000, -12)
  })

  it('decode FLOPs per step scales with batch', () => {
    expect(decodeFlopsPerStep(llama70b, 8, 1024, false)).toBe(8 * decodeFlopsPerStep(llama70b, 1, 1024, false))
  })

  it('prefill is compute-bound, decode at small batch is memory-bound', () => {
    const r = rooflineAnalysis(llama70b, { batch: 1, context: 4096, weightDtype: 'bf16', kvDtype: 'bf16' }, h100)
    expect(r.prefill.side).toBe('compute')
    expect(r.decode.side).toBe('memory')
    expect(r.decode.ai).toBeLessThan(2)
  })

  it('decode AI grows roughly with batch until KV dominates', () => {
    const wl = (b: number) => ({ batch: b, context: 512, weightDtype: 'bf16' as const, kvDtype: 'bf16' as const })
    const a1 = rooflineAnalysis(llama70b, wl(1), h100).decode.ai
    const a64 = rooflineAnalysis(llama70b, wl(64), h100).decode.ai
    expect(a64).toBeGreaterThan(a1 * 30)
    expect(a64).toBeLessThan(64)
  })

  it('TP does not change AI', () => {
    const wl = { batch: 16, context: 2048, weightDtype: 'bf16' as const, kvDtype: 'bf16' as const }
    const a = rooflineAnalysis(llama70b, wl, h100, { tp: 1 })
    const b = rooflineAnalysis(llama70b, wl, h100, { tp: 8 })
    expect(b.decode.ai).toBeCloseTo(a.decode.ai, 6)
    expect(b.decode.time).toBeCloseTo(a.decode.time / 8, 9)
  })
})
