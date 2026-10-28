import { describe, expect, it } from 'vitest'
import { canSwizzle, gcd, laneCells, warpAccess, wordOf, type BankLayout } from '@lib/bank'

const float = (pitch: number, swizzle = false): BankLayout => ({ elemBytes: 4, pitch, swizzle })
const half = (pitch: number, swizzle = false): BankLayout => ({ elemBytes: 2, pitch, swizzle })
const ways = (l: BankLayout, access: 'row' | 'col', k = 0) => warpAccess(l, laneCells(access, k)).ways

describe('float tile', () => {
  it('row read never conflicts; column read of [32][32] is 32-way, [32][33] is conflict-free', () => {
    expect(ways(float(32), 'row')).toBe(1)
    expect(ways(float(33), 'row', 7)).toBe(1)
    expect(ways(float(32), 'col', 5)).toBe(32)
    expect(warpAccess(float(33), laneCells('col', 5)).banks).toEqual(Array(32).fill(1))
  })

  it('[32][48] column read only touches banks 0 and 16, 16 lanes each', () => {
    const { banks } = warpAccess(float(48), laneCells('col', 0))
    expect(banks[0]).toBe(16)
    expect(banks[16]).toBe(16)
    expect(banks.filter((n) => n > 0)).toHaveLength(2)
  })

  it('column-read conflict degree = gcd(row stride in words, 32)', () => {
    for (let p = 1; p <= 96; p++) expect(ways(float(p), 'col')).toBe(gcd(p, 32))
  })
})

describe('half tile', () => {
  it('two lanes sharing a word is a broadcast', () => {
    expect(ways(half(64), 'row')).toBe(1)
  })

  it('[32][64] column read is 32-way; +1 half still 2-way; +2 half (one word) is clean', () => {
    expect(ways(half(64), 'col')).toBe(32)
    expect(ways(half(65), 'col')).toBe(2)
    expect(ways(half(66), 'col')).toBe(1)
  })
})

describe('xor swizzle', () => {
  it('only applies when a row is a multiple of 128 B', () => {
    expect(canSwizzle(float(32))).toBe(true)
    expect(canSwizzle(float(33))).toBe(false)
    expect(canSwizzle(half(64))).toBe(true)
    expect(canSwizzle(half(96))).toBe(false)
  })

  it('permutes words within each row', () => {
    for (const l of [float(32, true), float(64, true), half(64, true)]) {
      const rowWords = (l.pitch * l.elemBytes) / 4
      for (let r = 0; r < 32; r++) {
        const ws = new Set(Array.from({ length: l.pitch }, (_, c) => wordOf(l, r, c)))
        const lo = r * rowWords
        expect([...ws].every((w) => w >= lo && w < lo + rowWords)).toBe(true)
        expect(ws.size).toBe(l.elemBytes === 4 ? l.pitch : l.pitch / 2)
      }
    }
  })

  it('removes the conflict for every row and every column without padding', () => {
    for (const l of [float(32, true), float(64, true), half(64, true)])
      for (let k = 0; k < 32; k++) {
        expect(ways(l, 'row', k)).toBe(1)
        expect(ways(l, 'col', k)).toBe(1)
      }
  })
})
