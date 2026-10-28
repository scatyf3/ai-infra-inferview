import { describe, expect, it } from 'vitest'
import { blockSmemReads, regTileCost } from '@lib/regtile'

describe('register tile instruction counts', () => {
  it('matches the table in the GEMM tiling page', () => {
    expect(regTileCost(1, 1)).toMatchObject({ lds: 2, fma: 1 })
    expect(Math.round(regTileCost(1, 1).fmaShare * 100)).toBe(33)
    expect(regTileCost(8, 8)).toMatchObject({ lds: 16, fma: 64 })
    expect(Math.round(regTileCost(8, 8).fmaShare * 100)).toBe(80)
    expect(regTileCost(8, 8, 4)).toMatchObject({ lds: 4, fma: 64 })
    expect(Math.round(regTileCost(8, 8, 4).fmaShare * 100)).toBe(94)
  })

  it('counts accumulator plus operand registers', () => {
    expect(regTileCost(8, 8).regs).toBe(64 + 8 + 8)
  })

  it('bigger register tiles cut redundant shared memory reads', () => {
    expect(blockSmemReads(16, 16, 1, 1)).toEqual({ threads: 256, reads: 512, distinct: 32 })
    expect(blockSmemReads(16, 16, 8, 8)).toEqual({ threads: 4, reads: 64, distinct: 32 })
  })
})
