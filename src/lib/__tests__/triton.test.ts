import { describe, expect, it } from 'vitest'
import { cdiv, launchOrder, pidToTile, tileRange, vectorPrograms, waveFootprint } from '@lib/triton'

describe('triton 1D programs', () => {
  it('grid = cdiv(n, BLOCK) and only the last program is masked', () => {
    const ps = vectorPrograms(50, 16)
    expect(ps.length).toBe(cdiv(50, 16))
    expect(ps.length).toBe(4)
    expect(ps[2].offs[0]).toBe(32)
    expect(ps.slice(0, 3).every((p) => p.valid === 16)).toBe(true)
    expect(ps[3].valid).toBe(2)
    expect(ps[3].mask.slice(0, 3)).toEqual([true, true, false])
  })

  it('exact multiple has no masked lanes', () => {
    expect(vectorPrograms(64, 16).every((p) => p.valid === 16)).toBe(true)
  })
})

describe('triton matmul launch order', () => {
  it('groupM = 1 is row-major', () => {
    expect(pidToTile(5, 3, 4, 1)).toEqual({ pid: 5, pidM: 1, pidN: 1 })
  })

  it('grouped ordering walks down GROUP_M rows before moving right', () => {
    const order = launchOrder(4, 4, 2)
    expect(order.slice(0, 4).map((t) => [t.pidM, t.pidN])).toEqual([[0, 0], [1, 0], [0, 1], [1, 1]])
  })

  it('is a bijection even when numPidM is not divisible by groupM', () => {
    const order = launchOrder(3, 5, 2)
    const keys = new Set(order.map((t) => `${t.pidM},${t.pidN}`))
    expect(keys.size).toBe(15)
    // 最后一组只剩 1 行，退化为沿 N 走
    expect(order.slice(10).map((t) => t.pidM)).toEqual([2, 2, 2, 2, 2])
  })

  it('grouping shrinks the strips one wave has to read', () => {
    expect(waveFootprint(launchOrder(4, 4, 1), 0, 4)).toMatchObject({ rowsA: 1, colsB: 4 })
    expect(waveFootprint(launchOrder(4, 4, 2), 0, 4)).toMatchObject({ rowsA: 2, colsB: 2 })
  })

  it('tileRange clamps the edge tile', () => {
    expect(tileRange(2, 32, 80)).toEqual({ start: 64, end: 80, masked: 16 })
    expect(tileRange(1, 32, 80)).toEqual({ start: 32, end: 64, masked: 0 })
  })
})
