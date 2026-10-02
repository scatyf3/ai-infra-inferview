import { describe, expect, it } from 'vitest'
import { FAM_LEVELS, countByFam, famCls, famOf, isFamiliarity, isPassing } from '@lib/fam'
import { challengeById, challenges, slugOf, urlOf } from '@lib/leetgpu'

describe('familiarity', () => {
  it('未评是 null，不是 0', () => {
    expect(famOf({})).toBeNull()
    expect(famOf({ familiarity: null })).toBeNull()
    expect(famOf({ familiarity: '' })).toBeNull()
    expect(famOf({ familiarity: 0 })).toBe(0)
    expect(isPassing(null)).toBe(false)
    expect(isPassing(0)).toBe(true)
  })

  it('半档和类名', () => {
    expect(famOf({ familiarity: 1.5 })).toBe(1.5)
    expect(famCls(1.5)).toBe('f1_5')
    expect(famCls(null)).toBe('fnone')
    expect(isPassing(2)).toBe(true)
    expect(isPassing(3)).toBe(false)
  })

  it('只接受阶梯上的值', () => {
    expect(isFamiliarity(undefined)).toBe(true)
    expect(isFamiliarity(3.5)).toBe(true)
    expect(isFamiliarity(5)).toBe(false)
    expect(isFamiliarity(2.5)).toBe(false)
    expect(isFamiliarity('2')).toBe(false)
  })

  it('计数覆盖每一档', () => {
    const c = countByFam([{ familiarity: 2 }, { familiarity: 2 }, {}, { familiarity: 1.5 }])
    expect(Object.keys(c)).toHaveLength(FAM_LEVELS.length)
    expect(c['2']).toBe(2)
    expect(c['1.5']).toBe(1)
    expect(c.none).toBe(1)
  })
})

describe('leetgpu', () => {
  it('slug 规则与站点路由一致', () => {
    expect(slugOf('Top-p Sampling')).toBe('top-p-sampling')
    expect(slugOf('General Matrix Multiplication (GEMM)')).toBe('general-matrix-multiplication-gemm')
    expect(slugOf('Fused QKV Projection with RoPE and KV Cache Update')).toBe(
      'fused-qkv-projection-with-rope-and-kv-cache-update',
    )
  })

  it('清单 id 唯一、slug 唯一', () => {
    expect(challengeById.size).toBe(challenges.length)
    expect(new Set(challenges.map((c) => slugOf(c.title))).size).toBe(challenges.length)
  })

  it('urlOf', () => {
    expect(urlOf(challengeById.get(50)!)).toBe('https://leetgpu.com/challenges/rms-normalization')
  })
})

describe('leetgpu roadmap', () => {
  it('题号都在清单里、组内不重复、impl 非空', async () => {
    const { roadmap } = await import('@data/leetgpu-roadmap')
    for (const stage of roadmap)
      for (const g of stage.groups) {
        const ids = g.items.map((i) => i.id)
        expect(new Set(ids).size, g.key).toBe(ids.length)
        for (const it of g.items) {
          expect(challengeById.has(it.id), `${g.key} #${it.id}`).toBe(true)
          expect(it.impl.length, `${g.key} #${it.id}`).toBeGreaterThan(0)
        }
      }
  })
})
