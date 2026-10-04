/**
 * 原语闪卡：FSRS 调度（ts-fsrs）、出题顺序、记录的合并与校验、卡面的行内格式（纯函数）。
 * 自评用 FSRS 原生的 Again / Hard / Good / Easy；每张卡存一份 FSRS 状态，日期用 ISO 字符串，方便写进 json。
 */
import { createEmptyCard, fsrs, generatorParameters, Rating, State, TypeConvert, type Card as FsrsCard, type Grade } from 'ts-fsrs'

export type Deck = 'torch' | 'triton'

export interface Card {
  /** 稳定 id，复习记录按它存；改题面不要改 id */
  id: string
  deck: Deck
  /** 小标题，比如 stride、mask、online softmax */
  topic: string
  q: string
  /** 答案，支持行内 `code` 和 **加粗** */
  a: string
  /** 可选的代码块，放在答案下面 */
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

/** 学习中的卡可以提前 20 分钟复习（Anki 的 learn ahead limit），否则「1 分钟后」的卡要干等 */
export const LEARN_AHEAD_MS = 20 * 60_000

export type Bucket = 'new' | 'learning' | 'due' | 'later'

export function bucketOf(s: Sched | undefined, now: Date): Bucket {
  if (!s || s.state === State.New) return 'new'
  if (isLearning(s)) return 'learning'
  return Date.parse(s.due) <= now.getTime() ? 'due' : 'later'
}

export function countBuckets(cards: Card[], p: Progress, now: Date): Record<Bucket, number> {
  const c: Record<Bucket, number> = { new: 0, learning: 0, due: 0, later: 0 }
  for (const card of cards) c[bucketOf(p[card.id], now)]++
  return c
}

/**
 * 下一张：先出已到期的（学习中 + 复习，按到期时间），再出新卡（按文件顺序，受这一轮的新卡额度限制），
 * 最后才提前出 20 分钟内到期的学习中卡片。skipped 是这一轮跳过的。
 */
export function nextCard(cards: Card[], p: Progress, now: Date, skipped: Set<string>, newLeft: number): Card | undefined {
  const t = now.getTime()
  const pool = cards.filter((c) => !skipped.has(c.id))
  const byDue = (a: Card, b: Card) => Date.parse(p[a.id].due) - Date.parse(p[b.id].due)
  const started = pool.filter((c) => p[c.id] && p[c.id].state !== State.New)

  const due = started.filter((c) => Date.parse(p[c.id].due) <= t).sort(byDue)
  if (due.length) return due[0]
  if (newLeft > 0) {
    const fresh = pool.find((c) => bucketOf(p[c.id], now) === 'new')
    if (fresh) return fresh
  }
  return started.filter((c) => isLearning(p[c.id]) && Date.parse(p[c.id].due) <= t + LEARN_AHEAD_MS).sort(byDue)[0]
}

/** 学过的卡里，最早到期的那张什么时候到期（给「复习完了」的提示用） */
export function nextDueAt(cards: Card[], p: Progress): number | null {
  const ts = cards.map((c) => p[c.id]).filter((s): s is Sched => !!s && s.state !== State.New).map((s) => Date.parse(s.due))
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

/** 两份批注按卡片合并，较新的修改（at 更大）胜出 */
export function mergeNotes(a: Notes, b: Notes): Notes {
  const out: Notes = { ...a }
  for (const [id, note] of Object.entries(b)) if (!out[id] || note.at > out[id].at) out[id] = note
  return out
}

export function isNotes(v: unknown): v is Notes {
  return (
    !!v &&
    typeof v === 'object' &&
    !Array.isArray(v) &&
    Object.values(v).every((x) => !!x && typeof x.text === 'string' && isNum(x.at))
  )
}

/** 写文件时按 id 排序，丢掉空批注（文件是 dev 时唯一的数据源，删掉就是删掉） */
export function serializeNotes(n: Notes): string {
  const out: Notes = {}
  for (const id of Object.keys(n).sort()) if (n[id].text) out[id] = n[id]
  return JSON.stringify(out, null, 2) + '\n'
}

const escapeHtml =(s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

/** 卡面的行内格式：先转义，再把 `code` 和 **加粗** 换成标签 */
export function inlineMd(s: string): string {
  return escapeHtml(s)
    .split(/(`[^`]+`)/)
    .map((part) =>
      part.startsWith('`') && part.endsWith('`') && part.length > 1
        ? `<code>${part.slice(1, -1)}</code>`
        : part.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>'),
    )
    .join('')
}
