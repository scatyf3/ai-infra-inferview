import { describe, expect, it } from 'vitest'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { buildMatcher, inGlossaryScope, pageOf, splitTerms, type GlossaryEntry } from '@lib/glossary'
import { glossary } from '@data/glossary'

const e = (term: string, ...match: string[]): GlossaryEntry => ({ term, match: match.length ? match : [term], def: term })
const terms = (segs: ReturnType<typeof splitTerms>) => segs.flatMap((s) => (typeof s === 'string' ? [] : [s.text]))

describe('splitTerms', () => {
  const fsdp = e('FSDP')
  const fsdp2 = e('FSDP2')
  const zero = e('ZeRO')
  const m = buildMatcher([fsdp, fsdp2, zero, e('TP'), e('FP8', 'FP8', 'fp8'), e('MXFP8')])

  it('每个术语每页只标第一次', () => {
    const seen = new Set<GlossaryEntry>()
    expect(terms(splitTerms('TP 和 TP', m, seen))).toEqual(['TP'])
    expect(terms(splitTerms('又是 TP', m, seen))).toEqual([])
  })

  it('长写法优先，FSDP2 不会被拆成 FSDP', () => {
    const segs = splitTerms('FSDP2 比 FSDP 好', m, new Set())
    expect(segs).toEqual(['', { entry: fsdp2, text: 'FSDP2' }, ' 比 ', { entry: fsdp, text: 'FSDP' }, ' 好'].filter((s) => s !== ''))
  })

  it('按 ASCII 单词边界匹配：中文相邻可以，字母数字相邻不行', () => {
    expect(terms(splitTerms('用TP切', m, new Set()))).toEqual(['TP'])
    expect(terms(splitTerms('STP、TPU、MXFP8', m, new Set()))).toEqual(['MXFP8'])
    expect(terms(splitTerms('ZeRO-3', m, new Set()))).toEqual(['ZeRO'])
  })

  it('同一个术语的不同写法算同一个', () => {
    expect(terms(splitTerms('fp8 和 FP8', m, new Set()))).toEqual(['fp8'])
  })

  it('skip 的术语不标，也不占掉后面的位置', () => {
    const segs = splitTerms('TP 与 FSDP', m, new Set(), (x) => x === fsdp)
    expect(terms(segs)).toEqual(['TP'])
    expect(segs.map((s) => (typeof s === 'string' ? s : s.text)).join('')).toBe('TP 与 FSDP')
  })
})

describe('范围与页面路径', () => {
  it('只在训练栈页面生效', () => {
    expect(inGlossaryScope('train/compute-mfu.md')).toBe(true)
    expect(inGlossaryScope('stack/tr-hw-gpu.md')).toBe(true)
    expect(inGlossaryScope('stack/train-layer-0.md')).toBe(true)
    expect(inGlossaryScope('stack/hw-gpu.md')).toBe(false)
    expect(inGlossaryScope('inference/flash-attention.md')).toBe(false)
  })

  it('pageOf 和术语表 link 的写法一致', () => {
    expect(pageOf('train/compute-mfu.md')).toBe('/train/compute-mfu')
    expect(pageOf('train/index.md')).toBe('/train/')
  })
})

describe('术语表', () => {
  it('写法不重复', () => {
    const all = glossary.flatMap((g) => g.match)
    expect(new Set(all).size).toBe(all.length)
  })

  it('详解页都存在', () => {
    for (const g of glossary) {
      if (!g.link) continue
      const file = fileURLToPath(new URL(`../../../docs${g.link}.md`, import.meta.url))
      expect(existsSync(file), `${g.term} → ${g.link}`).toBe(true)
    }
  })
})
