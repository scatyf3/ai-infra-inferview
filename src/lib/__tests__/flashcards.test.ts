import { describe, expect, it } from 'vitest'
import {
  LEARN_AHEAD_MS,
  Rating,
  State,
  bucketOf,
  countBuckets,
  fmtInterval,
  inlineMd,
  isNotes,
  isProgress,
  mergeNotes,
  mergeProgress,
  noteOf,
  nextCard,
  preview,
  review,
  serializeNotes,
  serializeProgress,
  setNote,
  type Card,
  type Progress,
} from '@lib/flashcards'
import { cards } from '@data/flashcards'
import progressFile from '@data/flashcard-progress.json'
import notesFile from '@data/flashcard-notes.json'

const card = (id: string): Card => ({ id, deck: 'torch', topic: 't', q: id, a: id })
const T0 = new Date('2026-10-01T09:00:00Z')
const after = (ms: number) => new Date(T0.getTime() + ms)
const DAY = 86_400_000

describe('flashcard deck', () => {
  it('id 唯一，题面和答案非空，出处是站内链接', () => {
    expect(new Set(cards.map((c) => c.id)).size).toBe(cards.length)
    for (const c of cards) {
      expect(c.q.trim() && c.a.trim()).toBeTruthy()
      if (c.ref) expect(c.ref.startsWith('/')).toBe(true)
    }
  })

  it('仓库里的记录、批注文件格式正确，且不引用已删除的卡片', () => {
    expect(isProgress(progressFile)).toBe(true)
    expect(isNotes(notesFile)).toBe(true)
    const ids = new Set(cards.map((c) => c.id))
    for (const id of [...Object.keys(progressFile), ...Object.keys(notesFile)]) expect(ids.has(id)).toBe(true)
  })
})

describe('FSRS 调度', () => {
  it('新卡：Again < Hard < Good < Easy，且 Good 还在学习步里（分钟级）', () => {
    const pv = preview(undefined, T0)
    const due = (g: Rating.Again | Rating.Hard | Rating.Good | Rating.Easy) => Date.parse(pv[g].due) - T0.getTime()
    expect(due(Rating.Again)).toBeLessThan(due(Rating.Hard))
    expect(due(Rating.Hard)).toBeLessThan(due(Rating.Good))
    expect(due(Rating.Good)).toBeLessThan(due(Rating.Easy))
    expect(due(Rating.Good)).toBeLessThan(DAY)
    expect(pv[Rating.Easy].state).toBe(State.Review)
  })

  it('review 记一次复习，日期是 ISO 字符串，能通过校验', () => {
    const p = review({}, 'a', Rating.Good, T0)
    expect(p.a.reps).toBe(1)
    expect(p.a.last_review).toBe(T0.toISOString())
    expect(isProgress(p)).toBe(true)
    expect(isProgress(JSON.parse(serializeProgress(p)))).toBe(true)
  })

  it('答对一路 Good，间隔越来越长；Again 记一次遗忘', () => {
    let p: Progress = {}
    let now = T0
    const gaps: number[] = []
    for (let i = 0; i < 5; i++) {
      p = review(p, 'a', Rating.Good, now)
      const due = new Date(p.a.due)
      gaps.push(due.getTime() - now.getTime())
      now = due
    }
    for (let i = 1; i < gaps.length; i++) expect(gaps[i]).toBeGreaterThan(gaps[i - 1])
    expect(p.a.state).toBe(State.Review)
    p = review(p, 'a', Rating.Again, now)
    expect(p.a.lapses).toBe(1)
    expect(p.a.state).toBe(State.Relearning)
  })
})

describe('出题顺序', () => {
  const cs = ['a', 'b', 'c', 'd'].map(card)
  const none = new Set<string>()

  it('先出到期的复习卡（按到期时间），再出新卡', () => {
    let p: Progress = {}
    p = review(p, 'c', Rating.Easy, T0) // 几天后到期
    p = review(p, 'b', Rating.Easy, after(1000))
    const later = new Date(Date.parse(p.c.due) + DAY * 30)
    expect(nextCard(cs, p, later, none, 10)?.id).toBe('c')
    expect(nextCard(cs, p, T0, none, 10)?.id).toBe('a')
  })

  it('新卡额度用完就不出新卡；学习中的卡可以提前 20 分钟出', () => {
    const p = review({}, 'a', Rating.Again, T0) // 1 分钟后到期
    expect(nextCard(cs, p, T0, none, 0)?.id).toBe('a')
    expect(nextCard(cs, p, T0, none, 5)?.id).toBe('b') // 有新卡额度时先出新卡
    const far = review({}, 'a', Rating.Easy, T0) // 复习卡，没到期不提前
    expect(nextCard(cs, far, after(LEARN_AHEAD_MS), none, 0)).toBeUndefined()
  })

  it('跳过的卡这一轮不再出', () => {
    expect(nextCard(cs, {}, T0, new Set(['a', 'b']), 10)?.id).toBe('c')
  })

  it('分桶计数', () => {
    let p = review({}, 'a', Rating.Again, T0)
    p = review(p, 'b', Rating.Easy, T0)
    expect(bucketOf(p.a, T0)).toBe('learning')
    expect(countBuckets(cs, p, T0)).toEqual({ new: 2, learning: 1, due: 0, later: 1 })
    expect(countBuckets(cs, p, new Date(Date.parse(p.b.due) + 1)).due).toBe(1)
  })
})

describe('记录', () => {
  it('合并时最近复习过的胜出', () => {
    const old = review({}, 'a', Rating.Again, T0)
    const fresh = review({}, 'a', Rating.Easy, after(DAY))
    expect(mergeProgress(old, fresh).a).toEqual(fresh.a)
    expect(mergeProgress(fresh, old).a).toEqual(fresh.a)
  })

  it('校验', () => {
    const ok = review({}, 'a', Rating.Good, T0)
    expect(isProgress(ok)).toBe(true)
    expect(isProgress({ a: { ...ok.a, due: 'not a date' } })).toBe(false)
    expect(isProgress({ a: { ...ok.a, state: 7 } })).toBe(false)
    expect(isProgress({ a: { fam: 2, at: 1, n: 1 } })).toBe(false)
    expect(isProgress([])).toBe(false)
  })

  it('序列化按 id 排序', () => {
    const p = review(review({}, 'b', Rating.Good, T0), 'a', Rating.Good, T0)
    const s = serializeProgress(p)
    expect(s.indexOf('"a"')).toBeLessThan(s.indexOf('"b"'))
    expect(s.endsWith('\n')).toBe(true)
  })
})

describe('批注', () => {
  it('写入时去掉首尾空白；合并时较新的修改胜出，删除（空批注）也能盖过旧的', () => {
    const a = setNote({}, 'x', '  记得 keepdim  ', 100)
    expect(noteOf(a, 'x')).toBe('记得 keepdim')
    const deleted = setNote(a, 'x', '', 200)
    expect(noteOf(mergeNotes(a, deleted), 'x')).toBe('')
    expect(noteOf(mergeNotes(deleted, a), 'x')).toBe('')
    expect(noteOf({}, 'y')).toBe('')
  })

  it('写文件时丢掉空批注、按 id 排序', () => {
    const n = setNote(setNote(setNote({}, 'b', 'B', 1), 'a', 'A', 1), 'c', '', 1)
    const parsed = JSON.parse(serializeNotes(n))
    expect(Object.keys(parsed)).toEqual(['a', 'b'])
    expect(isNotes(parsed)).toBe(true)
  })

  it('校验', () => {
    expect(isNotes({ a: { text: 'x', at: 1 } })).toBe(true)
    expect(isNotes({ a: { text: 1, at: 1 } })).toBe(false)
    expect(isNotes({ a: { text: 'x' } })).toBe(false)
    expect(isNotes([])).toBe(false)
  })
})

describe('格式', () => {
  it('fmtInterval', () => {
    expect(fmtInterval(10 * 60_000)).toBe('10分钟')
    expect(fmtInterval(5 * 3600_000)).toBe('5小时')
    expect(fmtInterval(3 * DAY)).toBe('3天')
    expect(fmtInterval(45 * DAY)).toBe('1.5个月')
    expect(fmtInterval(0)).toBe('1分钟')
  })

  it('inlineMd 先转义 HTML，再处理 code 和加粗；code 里的 ** 不当加粗', () => {
    expect(inlineMd('`a < b` 和 **c**')).toBe('<code>a &lt; b</code> 和 <b>c</b>')
    expect(inlineMd('<script>')).toBe('&lt;script&gt;')
    expect(inlineMd('`x ** 2`')).toBe('<code>x ** 2</code>')
  })
})
