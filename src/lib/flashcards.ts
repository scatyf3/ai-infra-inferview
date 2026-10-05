/**
 * 原语闪卡：FSRS 调度（ts-fsrs）、出题顺序、记录的合并与校验、卡面的行内格式（纯函数）。
 * 自评用 FSRS 原生的 Again / Hard / Good / Easy；每张卡存一份 FSRS 状态，日期用 ISO 字符串，方便写进 json。
 */
import katex from 'katex'
import { createEmptyCard, fsrs, generatorParameters, Rating, State, TypeConvert, type Card as FsrsCard, type Grade } from 'ts-fsrs'

export type Deck = 'torch' | 'triton' | 'bagu'

/** 牌组：按钮上的名字和卡面徽章的颜色；新加牌组在这里登记 */
export const DECKS: { id: Deck; label: string; color: string }[] = [
  { id: 'torch', label: 'torch', color: '#ee4c2c' },
  { id: 'triton', label: 'triton', color: '#6d28d9' },
  { id: 'bagu', label: '八股', color: '#0e7490' },
]

export interface Card {
  /** 稳定 id，复习记录按它存；改题面不要改 id */
  id: string
  deck: Deck
  /** 小标题，比如 stride、mask、online softmax */
  topic: string
  q: string
  /** 答案，支持行内 `code`、**加粗**、$行内公式$ 和 $$单独一行的公式$$（KaTeX） */
  a: string
  /** 可选的示意图（等宽字符画），放在答案下面；手机上宽度别超过 40 列 */
  fig?: string
  /** 可选的代码块，放在示意图下面 */
  code?: string
  /** 出处：站内链接（不含 base） */
  ref?: string
}

/** 一张卡的 FSRS 状态，字段和 ts-fsrs 的 Card 一致，只是日期换成 ISO 字符串 */
export interface Sched {
  due: string
  stability: number
  difficulty: number
  elapsed_days: number
  scheduled_days: number
  learning_steps: number
  reps: number
  lapses: number
  state: State
  last_review?: string
}

export type Progress = Record<string, Sched>

export { Rating, State }
export type { Grade }

export const GRADES: { grade: Grade; name: string; hint: string; key: string }[] = [
  { grade: Rating.Again, name: 'Again', hint: '没想起来', key: '1' },
  { grade: Rating.Hard, name: 'Hard', hint: '想起来了，但很吃力', key: '2' },
  { grade: Rating.Good, name: 'Good', hint: '想了一下，答对了', key: '3' },
  { grade: Rating.Easy, name: 'Easy', hint: '脱口而出', key: '4' },
]

export const STATE_LABEL: Record<State, string> = {
  [State.New]: '新卡',
  [State.Learning]: '学习中',
  [State.Review]: '复习',
  [State.Relearning]: '重学',
}

// 不加 fuzz：同样的评分得到同样的间隔，记录文件的 diff 和单测都是确定的
const scheduler = fsrs(generatorParameters({ enable_fuzz: false }))

const toFsrs = (s: Sched | undefined, now: Date): FsrsCard => (s ? TypeConvert.card(s) : createEmptyCard(now))
const fromFsrs = (c: FsrsCard): Sched => {
  const { due, last_review, ...rest } = c
  return { ...rest, due: due.toISOString(), ...(last_review ? { last_review: last_review.toISOString() } : {}) }
}

/** 四个按钮各自会把这张卡排到什么时候 */
export function preview(s: Sched | undefined, now: Date): Record<Grade, Sched> {
  const log = scheduler.repeat(toFsrs(s, now), now)
  return Object.fromEntries(GRADES.map(({ grade }) => [grade, fromFsrs(log[grade].card)])) as Record<Grade, Sched>
}

export function review(p: Progress, id: string, grade: Grade, now: Date): Progress {
  return { ...p, [id]: fromFsrs(scheduler.next(toFsrs(p[id], now), now, grade).card) }
}

const isLearning = (s: Sched) => s.state === State.Learning || s.state === State.Relearning

export type Bucket = 'new' | 'learning' | 'due' | 'later' | 'suspended'

export function bucketOf(s: Sched | undefined, now: Date, suspended = false): Bucket {
  if (suspended) return 'suspended'
  if (!s || s.state === State.New) return 'new'
  if (isLearning(s)) return 'learning'
  return Date.parse(s.due) <= now.getTime() ? 'due' : 'later'
}

export function countBuckets(cards: Card[], p: Progress, now: Date, flags: Flags = {}): Record<Bucket, number> {
  const c: Record<Bucket, number> = { new: 0, learning: 0, due: 0, later: 0, suspended: 0 }
  for (const card of cards) c[bucketOf(p[card.id], now, isSuspended(flags, card.id))]++
  return c
}

export interface QueueOpts {
  /** 这一轮还能出几张新卡；不传就不限 */
  newLeft?: number
  /** 这一轮跳过的卡，按跳过的先后；它们排在所有别的卡后面 */
  deferred?: string[]
  flags?: Flags
}

/**
 * 下一张：先出已到期的（学习中 + 复习，按到期时间），再出新卡（按文件顺序，给了 newLeft 就受这个额度限制）。
 * 没到期的不提前出：学习中的卡评完 Good 要 10 分钟后才到期，提前出会让人以为评分没生效。暂停的卡不出。
 * 跳过的卡不是丢掉，而是排到最后：别的都出完了，再按跳过的先后出它们。
 * 它们这一轮已经出过一次，所以不再受新卡额度限制，否则额度被别的新卡用完后，跳过的新卡就回不来了。
 */
export function nextCard(cards: Card[], p: Progress, now: Date, { newLeft = Infinity, deferred = [], flags = {} }: QueueOpts = {}): Card | undefined {
  const t = now.getTime()
  const byDue = (a: Card, b: Card) => Date.parse(p[a.id].due) - Date.parse(p[b.id].due)
  const pick = (pool: Card[], allowNew: boolean): Card | undefined => {
    const due = pool.filter((c) => p[c.id] && p[c.id].state !== State.New && Date.parse(p[c.id].due) <= t).sort(byDue)
    if (due.length) return due[0]
    return allowNew ? pool.find((c) => bucketOf(p[c.id], now) === 'new') : undefined
  }

  const active = cards.filter((c) => !isSuspended(flags, c.id))
  const later = new Set(deferred)
  const first = pick(active.filter((c) => !later.has(c.id)), newLeft > 0)
  if (first) return first
  for (const id of deferred) {
    const c = active.find((x) => x.id === id)
    if (c && pick([c], true)) return c
  }
  return undefined
}

/** 学过、没暂停的卡里，最早到期的那张什么时候到期（给「复习完了」的提示用） */
export function nextDueAt(cards: Card[], p: Progress, flags: Flags = {}): number | null {
  const ts = cards
    .filter((c) => !isSuspended(flags, c.id))
    .map((c) => p[c.id])
    .filter((s): s is Sched => !!s && s.state !== State.New)
    .map((s) => Date.parse(s.due))
  return ts.length ? Math.min(...ts) : null
}

/** 间隔的简写：10分钟、3小时、2天、1.5个月、1.2年 */
export function fmtInterval(ms: number): string {
  const min = Math.max(1, Math.round(ms / 60_000))
  if (min < 60) return `${min}分钟`
  const h = Math.round(min / 60)
  if (h < 24) return `${h}小时`
  const d = Math.round(h / 24)
  if (d < 30) return `${d}天`
  if (d < 365) return `${+(d / 30).toFixed(1)}个月`
  return `${+(d / 365).toFixed(1)}年`
}

/** 两份记录按卡片合并，最近复习过的（last_review 更晚）胜出 */
export function mergeProgress(a: Progress, b: Progress): Progress {
  const at = (s: Sched) => (s.last_review ? Date.parse(s.last_review) : 0)
  const out: Progress = { ...a }
  for (const [id, s] of Object.entries(b)) if (!out[id] || at(s) > at(out[id])) out[id] = s
  return out
}

const isNum = (v: unknown) => typeof v === 'number' && Number.isFinite(v)
const isDate = (v: unknown) => typeof v === 'string' && !Number.isNaN(Date.parse(v))

export function isSched(v: unknown): v is Sched {
  if (!v || typeof v !== 'object') return false
  const s = v as Sched
  return (
    isDate(s.due) &&
    (s.last_review === undefined || isDate(s.last_review)) &&
    [s.stability, s.difficulty, s.elapsed_days, s.scheduled_days, s.learning_steps, s.reps, s.lapses].every(isNum) &&
    [State.New, State.Learning, State.Review, State.Relearning].includes(s.state)
  )
}

export function isProgress(v: unknown): v is Progress {
  return !!v && typeof v === 'object' && !Array.isArray(v) && Object.values(v).every(isSched)
}

/** 按 id 排序后输出，保证 diff 稳定 */
export function serializeProgress(p: Progress): string {
  const out: Progress = {}
  for (const id of Object.keys(p).sort()) out[id] = p[id]
  return JSON.stringify(out, null, 2) + '\n'
}

// ---------- 批注 ----------

export interface CardNote {
  text: string
  /** 最近一次修改的时间戳（ms） */
  at: number
}

/** 卡片 id → 批注。删除也记成一条空批注，这样合并时「删掉」能盖过旧的那份 */
export type Notes = Record<string, CardNote>

export const noteOf = (n: Notes, id: string): string => n[id]?.text ?? ''

export function setNote(n: Notes, id: string, text: string, now = Date.now()): Notes {
  return { ...n, [id]: { text: text.trim(), at: now } }
}

/** 按卡片合并，较新的修改（at 更大）胜出；批注和暂停标记都用它 */
function mergeByAt<T extends { at: number }>(a: Record<string, T>, b: Record<string, T>): Record<string, T> {
  const out = { ...a }
  for (const [id, x] of Object.entries(b)) if (!out[id] || x.at > out[id].at) out[id] = x
  return out
}

/** 按 id 排序后输出，保证 diff 稳定 */
function serializeById(v: Record<string, unknown>): string {
  const out: Record<string, unknown> = {}
  for (const id of Object.keys(v).sort()) out[id] = v[id]
  return JSON.stringify(out, null, 2) + '\n'
}

export const mergeNotes: (a: Notes, b: Notes) => Notes = mergeByAt

export function isNotes(v: unknown): v is Notes {
  return (
    !!v &&
    typeof v === 'object' &&
    !Array.isArray(v) &&
    Object.values(v).every((x) => !!x && typeof x.text === 'string' && isNum(x.at))
  )
}

/**
 * 写文件时按 id 排序。空批注（删除记录）也要留着：多台设备合并时，
 * 它靠更新的 at 盖过别的设备内存里的旧批注，删掉的话旧批注会被合并回来。
 */
export const serializeNotes: (n: Notes) => string = serializeById

// ---------- 暂停 ----------

export interface Flag {
  /** 暂停的卡不出现在复习里，直到恢复 */
  suspended: boolean
  /** 最近一次切换的时间戳（ms）；恢复也留一条 suspended: false，理由同空批注 */
  at: number
}

export type Flags = Record<string, Flag>

export const isSuspended = (f: Flags, id: string): boolean => !!f[id]?.suspended

export function setSuspended(f: Flags, id: string, suspended: boolean, now = Date.now()): Flags {
  return { ...f, [id]: { suspended, at: now } }
}

export const mergeFlags: (a: Flags, b: Flags) => Flags = mergeByAt
export const serializeFlags: (f: Flags) => string = serializeById

export function isFlags(v: unknown): v is Flags {
  return (
    !!v &&
    typeof v === 'object' &&
    !Array.isArray(v) &&
    Object.values(v).every((x) => !!x && typeof x.suspended === 'boolean' && isNum(x.at))
  )
}

const escapeHtml =(s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

const tex = (src: string, displayMode: boolean) => katex.renderToString(src, { displayMode, throwOnError: false, output: 'html' })

/** 卡面的行内格式：$$公式$$、$公式$、`code` 原样保留，其余先转义，再把 **加粗** 换成标签 */
export function inlineMd(s: string): string {
  return s
    .split(/(\$\$[^$]+\$\$|\$[^$\n]+\$|`[^`]+`)/)
    .map((part) => {
      if (part.length > 4 && part.startsWith('$$') && part.endsWith('$$')) return tex(part.slice(2, -2), true)
      if (part.length > 2 && part.startsWith('$') && part.endsWith('$')) return tex(part.slice(1, -1), false)
      if (part.length > 1 && part.startsWith('`') && part.endsWith('`')) return `<code>${escapeHtml(part.slice(1, -1))}</code>`
      return escapeHtml(part).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    })
    .join('')
}
