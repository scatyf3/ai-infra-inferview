import { describe, expect, it } from 'vitest'
import { blockSmemReads, regTileCost, tiledGemmCost } from '@lib/regtile'

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

  it('tiledGemmCost: 1x1 is the inner product, M x N is the outer product', () => {
    expect(tiledGemmCost(8, 8, 8, 1, 1)).toMatchObject({ loads: 1024, macs: 512, accum: 1, macPerLoad: 0.5 })
    expect(tiledGemmCost(8, 8, 8, 8, 8)).toMatchObject({ tiles: 1, loads: 128, accum: 64, macPerLoad: 4 })
    // 中间每一档：每读 1 个数做 TM·TN / (TM + TN) 次 MAC
    expect(tiledGemmCost(4096, 4096, 4096, 128, 128).macPerLoad).toBe(64)
    expect(tiledGemmCost(16, 16, 8, 8, 8).macPerLoad).toBe(regTileCost(8, 8).fma / regTileCost(8, 8).lds)
  })

  it('bigger register tiles cut redundant shared memory reads', () => {
    expect(blockSmemReads(16, 16, 1, 1)).toEqual({ threads: 256, reads: 512, distinct: 32 })
    expect(blockSmemReads(16, 16, 8, 8)).toEqual({ threads: 4, reads: 64, distinct: 32 })
  })
})
