// 熟练度 familiarity：和隔壁 leetcode 看板同一条阶梯、同一套颜色，两边读起来一致。
// 0 最熟 → 4 完全不会；frontmatter 里不写 = 未评（null）。
// 未评**不是** 0 —— 0 是阶梯顶端，千万别写成 `Number(x) || 0`，那会把没评过的题静默变成最熟。
// 半档 1.5 / 3.5 直接存小数，排序照样按数值比。

export type Familiarity = number | null

export const FAM: Record<number, { short: string; label: string }> = {
  0: { short: 'L0', label: '英语讲得清' },
  1: { short: 'L1', label: '盲写一次过' },
  1.5: { short: 'L1.5', label: '写得对 · 但不是最优实现' },
  2: { short: 'L2', label: '思路会 · 细节易写错' },
  3: { short: 'L3', label: '思路大概知道 · 不熟' },
  3.5: { short: 'L3.5', label: '方向对 · 但只是直觉' },
  4: { short: 'L4', label: '思路都不知道' },
}
export const FAM_NONE = { short: '—', label: '未评' }
/** 最熟 → 最生 → 未评；进度条从左到右就按这个顺序画 */
export const FAM_LEVELS: Familiarity[] = [0, 1, 1.5, 2, 3, 3.5, 4, null]

export function isFamiliarity(v: unknown): boolean {
  return v === null || v === undefined || (typeof v === 'number' && v in FAM)
}

export const famOf = (p: { familiarity?: unknown }): Familiarity => {
  const v = p.familiarity
  return v === null || v === undefined || v === '' ? null : Number(v)
}
export const famInfo = (f: Familiarity) => (f === null ? FAM_NONE : FAM[f])
export const famKey = (f: Familiarity): string => (f === null ? 'none' : String(f))
/** CSS 类名不能带小数点（`.f1.5` 会被当成两个类），所以 1.5 → f1_5 */
export const famCls = (f: Familiarity) => 'f' + famKey(f).replace('.', '_')

/** 「下一题」顺序：越生越先，未评排在 L2 之后（已经摸过但不熟的债先还） */
export const FAM_ORDER: Record<string, number> = { 4: 0, 3.5: 1, 3: 2, 2: 3, none: 4, 1.5: 5, 1: 6, 0: 7 }

/** L2 及以上 = 写得对，算「过了 OA 门槛」 */
export const isPassing = (f: Familiarity) => f !== null && f <= 2

export function countByFam<T extends { familiarity?: unknown }>(items: T[]): Record<string, number> {
  const c: Record<string, number> = Object.fromEntries(FAM_LEVELS.map((f) => [famKey(f), 0]))
  for (const it of items) c[famKey(famOf(it))]++
  return c
}
