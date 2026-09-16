import { describe, expect, it } from 'vitest'
import { addSeq, createPool, finish, freeBlockCount, preempt, resume, step } from '@lib/paged'

const prompt = (n: number, seed = 0) => Array.from({ length: n }, (_, i) => seed * 10000 + i)

describe('paged block manager', () => {
  it('allocates ceil(tokens / blockSize) blocks', () => {
    let s = createPool(8, 4)
    s = addSeq(s, 'A', prompt(10))
    expect(s.seqs[0].blockIds.length).toBe(3)
    expect(freeBlockCount(s)).toBe(5)
    expect(s.blocks[s.seqs[0].blockIds[2]].filled).toBe(2)
  })

  it('shares full prefix blocks between sequences with refcount', () => {
    let s = createPool(8, 4)
    const shared = prompt(8) // 2 full blocks
    s = addSeq(s, 'A', [...shared, 99])
    s = addSeq(s, 'B', [...shared, 77])
    const a = s.seqs[0], b = s.seqs[1]
    expect(a.blockIds.slice(0, 2)).toEqual(b.blockIds.slice(0, 2))
    expect(b.cachedBlocks).toBe(2)
    expect(s.blocks[a.blockIds[0]].refCount).toBe(2)
    expect(a.blockIds[2]).not.toBe(b.blockIds[2])
    expect(freeBlockCount(s)).toBe(4)
  })

  it('decode step fills last block then allocates a new one', () => {
    let s = createPool(4, 2)
    s = addSeq(s, 'A', prompt(3)) // blocks: [full, 1/2]
    s = step(s)
    expect(s.seqs[0].blockIds.length).toBe(2)
    expect(s.blocks[s.seqs[0].blockIds[1]].filled).toBe(2)
    s = step(s)
    expect(s.seqs[0].blockIds.length).toBe(3)
  })

  it('swap preemption frees GPU blocks and resume reallocates', () => {
    let s = createPool(4, 2)
    s = addSeq(s, 'A', prompt(4))
    s = preempt(s, 'A', 'swap')
    expect(freeBlockCount(s)).toBe(4)
    expect(s.seqs[0].state).toBe('swapped')
    expect(s.cpuBlocks['A']).toBe(2)
    s = resume(s, 'A')
    expect(s.seqs[0].state).toBe('running')
    expect(s.seqs[0].blockIds.length).toBe(2)
    expect(s.cpuBlocks['A']).toBeUndefined()
  })

  it('recompute preemption keeps tokens and re-prefills with prefix hits', () => {
    let s = createPool(6, 2)
    s = addSeq(s, 'A', prompt(4))
    s = addSeq(s, 'B', prompt(4)) // shares both blocks
    s = preempt(s, 'A', 'recompute')
    expect(s.seqs[0].state).toBe('preempted')
    expect(s.seqs[0].tokens.length).toBe(4)
    s = resume(s, 'A')
    expect(s.seqs[0].cachedBlocks).toBe(2) // B 还持有这些 block
  })

  it('reports failure when out of blocks instead of throwing', () => {
    let s = createPool(2, 2)
    s = addSeq(s, 'A', prompt(4))
    s = addSeq(s, 'B', prompt(4, 1))
    expect(s.seqs.length).toBe(1)
    expect(s.log[s.log.length - 1]).toContain('抢占')
  })

  it('finish releases blocks', () => {
    let s = createPool(4, 2)
    s = addSeq(s, 'A', prompt(4))
    s = finish(s, 'A')
    expect(freeBlockCount(s)).toBe(4)
  })
})
