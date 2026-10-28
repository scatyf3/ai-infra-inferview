import { describe, expect, it } from 'vitest'
import { bankLoad, sectorsOf, transposeSteps, transposeTotals, type Cell } from '@lib/transpose'

const row = (r: number, c0: number, len: number): Cell[] => Array.from({ length: len }, (_, t) => [r, c0 + t])
const col = (c: number, r0: number, len: number): Cell[] => Array.from({ length: len }, (_, t) => [r0 + t, c])

describe('sectors', () => {
  it('8 contiguous aligned floats fit one 32 B sector; a column takes one sector per element', () => {
    expect(sectorsOf(row(3, 8, 8), 16)).toHaveLength(1)
    expect(sectorsOf(col(5, 0, 8), 16)).toHaveLength(8)
  })

  it('real scale: 32 threads reading a row = 4 sectors, a column = 32', () => {
    expect(sectorsOf(row(0, 0, 32), 64)).toHaveLength(4)
    expect(sectorsOf(col(0, 0, 32), 64)).toHaveLength(32)
  })

  it('misaligned row spills into one more sector', () => {
    expect(sectorsOf(row(0, 4, 8), 16)).toHaveLength(2)
  })
})

describe('bank conflicts', () => {
  it('column of tile[32][32] hits one bank 32 times; tile[32][33] spreads over 32 banks', () => {
    expect(Math.max(...bankLoad(col(5, 0, 32), 32, 32))).toBe(32)
    expect(bankLoad(col(5, 0, 32), 33, 32)).toEqual(Array(32).fill(1))
  })

  it('same address is a broadcast, not a conflict', () => {
    expect(bankLoad([[0, 0], [0, 0], [0, 0]], 8, 8)[0]).toBe(1)
  })
})

describe('transpose steps', () => {
  it('naive: rows read in 1 sector, columns written in 8', () => {
    const s = transposeSteps('naive', 16, 8, 1, 0)
    expect(s).toHaveLength(16)
    expect(transposeTotals(s).sectors).toBe(8 * 1 + 8 * 8)
  })

  it('smem: both sides coalesced, but loading a tile column is an 8-way conflict', () => {
    const s = transposeSteps('smem', 16, 8, 1, 0)
    expect(s.find((x) => x.kind === 'sync')).toBeTruthy()
    expect(transposeTotals(s).sectors).toBe(16)
    expect(s.filter((x) => x.smem?.op === 'store').every((x) => x.ways === 1)).toBe(true)
    expect(s.filter((x) => x.smem?.op === 'load').every((x) => x.ways === 8)).toBe(true)
  })

  it('padding removes the conflict and keeps HBM traffic', () => {
    const s = transposeSteps('padded', 16, 8, 1, 0)
    expect(transposeTotals(s)).toEqual({ sectors: 16, smemPasses: 16, smemAccesses: 16 })
  })

  it('every mode writes exactly the transposed block, and smem lanes carry in[r][c] to out[c][r]', () => {
    for (const mode of ['naive', 'smem', 'padded'] as const) {
      const bx = 1, by = 0, T = 8
      const s = transposeSteps(mode, 16, T, bx, by)
      const written = new Set(s.filter((x) => x.hbm?.mat === 'out').flatMap((x) => x.hbm!.cells.map(([r, c]) => `${r},${c}`)))
      const expected = new Set<string>()
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) expected.add(`${bx * T + j},${by * T + i}`)
      expect(written).toEqual(expected)
    }
    // 阶段 2：lane t 从 tile[t][w] 取值写到 out[x0+w][y0+t]，tile[t][w] 存的是 in[y0+t][x0+w]
    const w = transposeSteps('smem', 16, 8, 1, 0).filter((x) => x.kind === 'write')[3]
    w.smem!.cells.forEach(([i, j], t) => {
      const [R, C] = w.hbm!.cells[t]
      expect([C, R]).toEqual([0 * 8 + i, 1 * 8 + j])
    })
  })
})
