/**
 * shared memory bank conflict 的模型（纯函数），给 BankConflictViz 用。真实规模：32 个 bank、每个 bank 一个周期一个 4 B 的字，一个 warp 32 个线程。
 * tile 是行主序数组，每行 pitch 个元素（float 4 B / half 2 B）。
 * swizzle：第 r 行里的字下标 j 存到 j ^ r（只动低 5 位），要求一行正好是 128 B 的整数倍。
 */

export const BANKS = 32
export const WORD_BYTES = 4

export interface BankLayout {
  elemBytes: 2 | 4
  /** 每行实际占几个元素（padding 时比逻辑宽度多） */
  pitch: number
  swizzle?: boolean
}
/** row：lane t 读 tile[k][t]；col：lane t 读 tile[t][k] */
export type Access = 'row' | 'col'
/** [row, col] */
export type Cell = [number, number]

export const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b))

/** 一行是 128 B 的整数倍时才能按行做 XOR swizzle */
export const canSwizzle = (l: BankLayout) => (l.pitch * l.elemBytes) % (BANKS * WORD_BYTES) === 0

/** tile[r][c] 所在的字（4 B）下标；bank 是它 mod 32 */
export function wordOf(l: BankLayout, r: number, c: number): number {
  const w = Math.floor(((r * l.pitch + c) * l.elemBytes) / WORD_BYTES)
  if (!l.swizzle) return w
  const rowStart = (r * l.pitch * l.elemBytes) / WORD_BYTES
  const j = w - rowStart
  return rowStart + (j & ~(BANKS - 1)) + ((j & (BANKS - 1)) ^ (r & (BANKS - 1)))
}

/** 一个 warp 的 32 个 lane 各访问哪个元素 */
export const laneCells = (access: Access, k: number): Cell[] =>
  Array.from({ length: BANKS }, (_, t): Cell => (access === 'row' ? [k, t] : [t, k]))

/**
 * 一个 warp 的一次访问：words[t] 是 lane t 的字下标，banks[b] 是 bank b 上的不同字数，
 * 最大值就是几路冲突。多个 lane 落在同一个字上算广播，只记一次。
 */
export function warpAccess(l: BankLayout, cells: Cell[]) {
  const words = cells.map(([r, c]) => wordOf(l, r, c))
  const per = Array.from({ length: BANKS }, () => new Set<number>())
  for (const w of words) per[w % BANKS].add(w)
  const banks = per.map((s) => s.size)
  return { words, banks, ways: Math.max(...banks) }
}
