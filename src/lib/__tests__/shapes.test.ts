import { describe, expect, it } from 'vitest'
import { formatShape, kvElemsPerTokenPerLayer, numel, shapeFlow, type Dims } from '@lib/shapes'

const dims: Dims = { B: 2, S: 16, S_kv: 128, d: 4096, H: 32, H_kv: 8, h_d: 128, d_ff: 14336, d_c: 512, d_r: 64 }

describe('shapes', () => {
  it('prefill MHA scores are [B,H,S,S]', () => {
    const steps = shapeFlow({ variant: 'mha', phase: 'prefill' })
    const scores = steps.find((s) => s.id === 'scores')!
    expect(formatShape(scores.output)).toBe('[B, H, S, S]')
    expect(numel(scores.output, dims)).toBe(2 * 32 * 16 * 16)
  })

  it('decode GQA reads KV cache with H_kv heads and repeats to H', () => {
    const steps = shapeFlow({ variant: 'gqa', phase: 'decode' })
    const read = steps.find((s) => s.cache === 'read')!
    expect(formatShape(read.output)).toBe('[B, H_kv, S_kv, h_d]')
    expect(steps.some((s) => s.id === 'repeat')).toBe(true)
    const scores = steps.find((s) => s.id === 'scores')!
    expect(formatShape(scores.output)).toBe('[B, H, 1, S_kv]')
  })

  it('MHA has no repeat step; MLA decode caches latent', () => {
    expect(shapeFlow({ variant: 'mha', phase: 'decode' }).some((s) => s.id === 'repeat')).toBe(false)
    const mla = shapeFlow({ variant: 'mla', phase: 'decode' })
    expect(mla.find((s) => s.cache === 'read')!.output).toEqual(['B', 'S_kv', 'd_c'])
  })

  it('KV elements per token per layer: MHA > GQA > MLA', () => {
    const mha = kvElemsPerTokenPerLayer('mha', dims)
    const gqa = kvElemsPerTokenPerLayer('gqa', dims)
    const mla = kvElemsPerTokenPerLayer('mla', dims)
    expect(mha).toBe(8192)
    expect(gqa).toBe(2048)
    expect(mla).toBe(576)
  })

  it('every step ends in MLP down-proj with [B,S,d]', () => {
    for (const variant of ['mha', 'gqa', 'mla'] as const) {
      for (const phase of ['prefill', 'decode'] as const) {
        const steps = shapeFlow({ variant, phase })
        const last = steps[steps.length - 1]
        expect(last.id).toBe('down')
        expect(last.output[2]).toBe('d')
      }
    }
  })
})
