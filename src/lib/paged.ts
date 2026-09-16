/**
 * PagedAttention block manager 的最小状态机（纯函数，返回新状态）。
 * 用于可视化：block 网格、每序列 block table、prefix 共享、swap / recompute 抢占。
 */

export interface Block {
  id: number
  /** 引用计数；0 = free，>1 = 被 prefix 共享 */
  refCount: number
  /** 满 block 的内容 hash（prefix caching 用），未满时为 null */
  hash: string | null
  /** 已填 token 数 */
  filled: number
  /** 第一个 owner（展示用） */
  owner: string | null
}

export type SeqState = 'running' | 'swapped' | 'preempted' | 'finished'

export interface Seq {
  id: string
  /** 全部 token（prompt + generated） */
  tokens: number[]
  promptLen: number
  blockIds: number[]
  state: SeqState
  /** 命中 prefix cache 的 block 数 */
  cachedBlocks: number
}

export interface PagedState {
  blockSize: number
  blocks: Block[]
  seqs: Seq[]
  /** hash → block id（prefix cache 表） */
  hashTable: Record<string, number>
  /** swap 到 CPU 的序列：id → 被换出的 block 数 */
  cpuBlocks: Record<string, number>
  log: string[]
  /** 记录最近一次操作影响的 block，动画高亮用 */
  touched: number[]
}

export function createPool(numBlocks: number, blockSize: number): PagedState {
  return {
    blockSize,
    blocks: Array.from({ length: numBlocks }, (_, i) => ({ id: i, refCount: 0, hash: null, filled: 0, owner: null })),
    seqs: [],
    hashTable: {},
    cpuBlocks: {},
    log: [`创建 ${numBlocks} 个 block，每个 ${blockSize} token`],
    touched: [],
  }
}

export function freeBlockCount(s: PagedState): number {
  return s.blocks.filter((b) => b.refCount === 0).length
}

function clone(s: PagedState): PagedState {
  return {
    ...s,
    blocks: s.blocks.map((b) => ({ ...b })),
    seqs: s.seqs.map((q) => ({ ...q, tokens: [...q.tokens], blockIds: [...q.blockIds] })),
    hashTable: { ...s.hashTable },
    cpuBlocks: { ...s.cpuBlocks },
    log: [...s.log],
    touched: [],
  }
}

function prefixHash(tokens: number[], endExclusive: number): string {
  return tokens.slice(0, endExclusive).join(',')
}

function allocFree(s: PagedState): Block | null {
  return s.blocks.find((b) => b.refCount === 0) ?? null
}

function releaseBlock(s: PagedState, id: number) {
  const b = s.blocks[id]
  b.refCount = Math.max(0, b.refCount - 1)
  if (b.refCount === 0) {
    // 保留 hash 供后续命中（vLLM 用 evictor / LRU；这里简化为：free 但 hash 仍在表里，直到被重新分配）
    b.owner = null
  }
}

/** 为一个序列分配 blocks，尽量命中 prefix cache */
function allocateForTokens(s: PagedState, seq: Seq, tokens: number[], usePrefixCache: boolean): boolean {
  const bs = s.blockSize
  const fullBlocks = Math.floor(tokens.length / bs)
  const remainder = tokens.length % bs
  let cached = 0
  for (let i = 0; i < fullBlocks; i++) {
    const h = prefixHash(tokens, (i + 1) * bs)
    const hit = usePrefixCache ? s.hashTable[h] : undefined
    if (hit !== undefined && s.blocks[hit].hash === h) {
      const b = s.blocks[hit]
      b.refCount += 1
      if (!b.owner) b.owner = seq.id
      seq.blockIds.push(b.id)
      s.touched.push(b.id)
      cached += 1
      continue
    }
    const b = allocFree(s)
    if (!b) return false
    b.refCount = 1
    b.hash = h
    b.filled = bs
    b.owner = seq.id
    s.hashTable[h] = b.id
    seq.blockIds.push(b.id)
    s.touched.push(b.id)
  }
  if (remainder > 0) {
    const b = allocFree(s)
    if (!b) return false
    b.refCount = 1
    b.hash = null
    b.filled = remainder
    b.owner = seq.id
    seq.blockIds.push(b.id)
    s.touched.push(b.id)
  }
  seq.cachedBlocks = cached
  return true
}

function rollback(s: PagedState, seq: Seq) {
  for (const id of seq.blockIds) releaseBlock(s, id)
  seq.blockIds = []
}

export function addSeq(state: PagedState, id: string, promptTokens: number[], usePrefixCache = true): PagedState {
  const s = clone(state)
  if (s.seqs.some((q) => q.id === id)) {
    s.log.push(`序列 ${id} 已存在`)
    return s
  }
  const seq: Seq = { id, tokens: [...promptTokens], promptLen: promptTokens.length, blockIds: [], state: 'running', cachedBlocks: 0 }
  if (!allocateForTokens(s, seq, promptTokens, usePrefixCache)) {
    rollback(s, seq)
    s.log.push(`序列 ${id} 分配失败：free block 不足（需要 ${Math.ceil(promptTokens.length / s.blockSize)}，剩 ${freeBlockCount(s)}）→ 进入等待，需要抢占`)
    return s
  }
  s.seqs.push(seq)
  s.log.push(`序列 ${id} 加入：${promptTokens.length} token → ${seq.blockIds.length} block（prefix 命中 ${seq.cachedBlocks}）`)
  return s
}

/** 所有 running 序列各 decode 一个 token */
export function step(state: PagedState, nextToken: (seq: Seq) => number = (q) => 1000 + q.tokens.length): PagedState {
  const s = clone(state)
  for (const seq of s.seqs) {
    if (seq.state !== 'running') continue
    const last = seq.blockIds[seq.blockIds.length - 1]
    const lastBlock = last !== undefined ? s.blocks[last] : null
    const tok = nextToken(seq)
    if (lastBlock && lastBlock.filled < s.blockSize && lastBlock.refCount === 1) {
      lastBlock.filled += 1
      seq.tokens.push(tok)
      s.touched.push(lastBlock.id)
      if (lastBlock.filled === s.blockSize) {
        lastBlock.hash = prefixHash(seq.tokens, seq.tokens.length)
        s.hashTable[lastBlock.hash] = lastBlock.id
      }
      continue
    }
    if (lastBlock && lastBlock.filled < s.blockSize && lastBlock.refCount > 1) {
      // copy-on-write：共享的未满 block 不能直接写
      const nb = allocFree(s)
      if (!nb) {
        s.log.push(`序列 ${seq.id} COW 失败：无 free block`)
        continue
      }
      releaseBlock(s, lastBlock.id)
      nb.refCount = 1
      nb.filled = lastBlock.filled + 1
      nb.owner = seq.id
      seq.blockIds[seq.blockIds.length - 1] = nb.id
      seq.tokens.push(tok)
      s.touched.push(nb.id)
      s.log.push(`序列 ${seq.id} copy-on-write：block ${lastBlock.id} → ${nb.id}`)
      continue
    }
    const nb = allocFree(s)
    if (!nb) {
      s.log.push(`序列 ${seq.id} 需要新 block 但已无 free block → 调度器必须抢占（swap 或 recompute）`)
      continue
    }
    nb.refCount = 1
    nb.filled = 1
    nb.hash = null
    nb.owner = seq.id
    seq.blockIds.push(nb.id)
    seq.tokens.push(tok)
    s.touched.push(nb.id)
  }
  s.log.push(`decode 一步：free block 剩 ${freeBlockCount(s)}`)
  return s
}

export function preempt(state: PagedState, id: string, mode: 'swap' | 'recompute'): PagedState {
  const s = clone(state)
  const seq = s.seqs.find((q) => q.id === id)
  if (!seq || seq.state !== 'running') {
    s.log.push(`序列 ${id} 不在 running，无法抢占`)
    return s
  }
  const n = seq.blockIds.length
  s.touched.push(...seq.blockIds)
  for (const bid of seq.blockIds) releaseBlock(s, bid)
  seq.blockIds = []
  if (mode === 'swap') {
    s.cpuBlocks[id] = n
    seq.state = 'swapped'
    s.log.push(`抢占 ${id}（swap）：${n} 个 block 拷到 CPU，PCIe 传 ${n} × block_bytes；GPU 释放 ${n} block`)
  } else {
    seq.state = 'preempted'
    s.log.push(`抢占 ${id}（recompute）：直接丢弃 ${n} 个 block，恢复时把已生成 token 当 prompt 重新 prefill（有 prefix cache 时能省一部分）`)
  }
  return s
}

export function resume(state: PagedState, id: string): PagedState {
  const s = clone(state)
  const seq = s.seqs.find((q) => q.id === id)
  if (!seq || seq.state === 'running' || seq.state === 'finished') {
    s.log.push(`序列 ${id} 无需恢复`)
    return s
  }
  const wasSwapped = seq.state === 'swapped'
  seq.blockIds = []
  if (!allocateForTokens(s, seq, seq.tokens, !wasSwapped)) {
    rollback(s, seq)
    s.log.push(`恢复 ${id} 失败：free block 不足`)
    return s
  }
  if (wasSwapped) {
    delete s.cpuBlocks[id]
    s.log.push(`恢复 ${id}（swap in）：从 CPU 拷回 ${seq.blockIds.length} 个 block`)
  } else {
    s.log.push(`恢复 ${id}（recompute）：重新 prefill ${seq.tokens.length} token，prefix cache 命中 ${seq.cachedBlocks} 个 block`)
  }
  seq.state = 'running'
  return s
}

export function finish(state: PagedState, id: string): PagedState {
  const s = clone(state)
  const seq = s.seqs.find((q) => q.id === id)
  if (!seq) return s
  s.touched.push(...seq.blockIds)
  for (const bid of seq.blockIds) releaseBlock(s, bid)
  seq.blockIds = []
  seq.state = 'finished'
  delete s.cpuBlocks[id]
  s.log.push(`序列 ${id} 结束，释放 block（满 block 的 hash 仍保留供 prefix 命中）`)
  return s
}

export function removeSeq(state: PagedState, id: string): PagedState {
  const s = finish(state, id)
  s.seqs = s.seqs.filter((q) => q.id !== id)
  return s
}
