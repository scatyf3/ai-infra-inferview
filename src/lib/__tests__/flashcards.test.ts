import { describe, expect, it } from 'vitest'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import {
  DECKS,
  Rating,
  State,
  bucketOf,
  countBuckets,
  fmtInterval,
  inlineMd,
  isFlags,
  isNotes,
  isSuspended,
  isProgress,
  mergeFlags,
  mergeNotes,
  mergeProgress,
  noteOf,
  nextCard,
  preview,
  review,
  serializeFlags,
  serializeNotes,
  serializeProgress,
  setNote,
  setSuspended,
  type Card,
  type Progress,
} from '@lib/flashcards'
import { cards, retiredIds } from '@data/flashcards'
import progressFile from '@data/flashcard-progress.json'
import notesFile from '@data/flashcard-notes.json'
import flagsFile from '@data/flashcard-flags.json'

const card = (id: string): Card => ({ id, deck: 'torch', topic: 't', q: id, a: id })
const T0 = new Date('2026-10-01T09:00:00Z')
const after = (ms: number) => new Date(T0.getTime() + ms)
const DAY = 86_400_000

describe('flashcard deck', () => {
  it('id 唯一，题面和答案非空，出处是站内链接', () => {
    expect(new Set(cards.map((c) => c.id)).size).toBe(cards.length)
    for (const id of retiredIds) expect(cards.some((c) => c.id === id), `${id} 已退役，别再用`).toBe(false)
    for (const c of cards) {
      expect(c.q.trim() && c.a.trim()).toBeTruthy()
      if (c.ref) expect(c.ref.startsWith('/')).toBe(true)
    }
  })

  it('每张卡的牌组都登记过，出处指向站内存在的页面', () => {
    const decks = new Set(DECKS.map((d) => d.id))
    const docs = fileURLToPath(new URL('../../../docs', import.meta.url))
    for (const c of cards) {
      expect(decks.has(c.deck), c.id).toBe(true)
      if (!c.ref) continue
      const page = c.ref.split('#')[0].replace(/\/$/, '/index')
      expect(existsSync(`${docs}${page}.md`), `${c.id} → ${c.ref}`).toBe(true)
    }
  })

  it('仓库里的记录、批注文件格式正确，且不引用已删除的卡片', () => {
    expect(isProgress(progressFile)).toBe(true)
    expect(isNotes(notesFile)).toBe(true)
    expect(isFlags(flagsFile)).toBe(true)
    const ids = new Set([...cards.map((c) => c.id), ...retiredIds])
    for (const id of [...Object.keys(progressFile), ...Object.keys(notesFile), ...Object.keys(flagsFile)]) expect(ids.has(id)).toBe(true)
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

  it('先出到期的复习卡（按到期时间），再出新卡', () => {
    let p: Progress = {}
    p = review(p, 'c', Rating.Easy, T0) // 几天后到期
    p = review(p, 'b', Rating.Easy, after(1000))
    const later = new Date(Date.parse(p.c.due) + DAY * 30)
    expect(nextCard(cs, p, later, { newLeft: 10 })?.id).toBe('c')
    expect(nextCard(cs, p, T0, { newLeft: 10 })?.id).toBe('a')
  })

  it('新卡额度用完就不出新卡；没到期的学习中卡片不提前出', () => {
    const p = review({}, 'a', Rating.Good, T0) // 10 分钟后到期
    expect(nextCard(cs, p, T0, { newLeft: 0 })).toBeUndefined() // 刚评完不会马上再出
    expect(nextCard(cs, p, after(10 * 60_000), { newLeft: 0 })?.id).toBe('a')
    expect(nextCard(cs, p, T0, { newLeft: 5 })?.id).toBe('b')
  })

  it('不传新卡额度就不限量', () => {
    let p: Progress = {}
    for (const c of cs) p = review(p, c.id, Rating.Easy, T0)
    expect(nextCard([...cs, card('z')], p, T0)?.id).toBe('z')
  })

  it('跳过的卡排到最后：别的出完了，再按跳过的先后出', () => {
    expect(nextCard(cs, {}, T0, { newLeft: 10, deferred: ['a', 'b'] })?.id).toBe('c')
    const two = cs.slice(0, 2)
    expect(nextCard(two, {}, T0, { newLeft: 10, deferred: ['b', 'a'] })?.id).toBe('b')
    // 跳过的新卡不受新卡额度限制：这一轮已经出过一次，额度被别的新卡用完也要回来
    expect(nextCard(two, {}, T0, { newLeft: 0, deferred: ['a', 'b'] })?.id).toBe('a')
    // 跳过的卡也要到期才出
    const learning = review({}, 'a', Rating.Good, T0)
    expect(nextCard([cs[0]], learning, T0, { newLeft: 10, deferred: ['a'] })).toBeUndefined()
    // 到期的复习卡被跳过，排在新卡后面
    const p = review({}, 'a', Rating.Again, T0)
    expect(nextCard(two, p, after(DAY), { newLeft: 10, deferred: ['a'] })?.id).toBe('b')
    expect(nextCard(two, review(p, 'b', Rating.Easy, after(DAY)), after(DAY), { newLeft: 10, deferred: ['a'] })?.id).toBe('a')
  })

  it('暂停的卡不出，也不算进到期提示', () => {
    const flags = setSuspended({}, 'a', true, 1)
    expect(nextCard(cs, {}, T0, { newLeft: 10, flags })?.id).toBe('b')
    expect(nextCard(cs, {}, T0, { newLeft: 10, flags: setSuspended(flags, 'a', false, 2) })?.id).toBe('a')
    expect(nextCard([cs[0]], {}, T0, { newLeft: 10, flags, deferred: ['a'] })).toBeUndefined()
  })

  it('分桶计数', () => {
    let p = review({}, 'a', Rating.Again, T0)
    p = review(p, 'b', Rating.Easy, T0)
    expect(bucketOf(p.a, T0)).toBe('learning')
    expect(countBuckets(cs, p, T0)).toEqual({ new: 2, learning: 1, due: 0, later: 1, suspended: 0 })
    expect(countBuckets(cs, p, new Date(Date.parse(p.b.due) + 1)).due).toBe(1)
    expect(countBuckets(cs, p, T0, setSuspended({}, 'a', true, 1))).toEqual({ new: 2, learning: 0, due: 0, later: 1, suspended: 1 })
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

  it('写文件时按 id 排序，删除记录（空批注）也保留', () => {
    const n = setNote(setNote(setNote({}, 'b', 'B', 1), 'a', 'A', 1), 'c', '', 1)
    const parsed = JSON.parse(serializeNotes(n))
    expect(Object.keys(parsed)).toEqual(['a', 'b', 'c'])
    expect(isNotes(parsed)).toBe(true)
  })

  it('另一台设备内存里的旧批注，不会把已删除的批注合并回来', () => {
    const old = setNote({}, 'x', '旧批注', 100) // 设备 B 打开页面时读到的
    const file = JSON.parse(serializeNotes(setNote(old, 'x', '', 200))) // 设备 A 删掉后写进文件
    const fromB = setNote(old, 'y', '设备 B 新写的', 300) // 设备 B 写别的卡，带着旧的 x 一起发上来
    const merged = mergeNotes(file, fromB)
    expect(noteOf(merged, 'x')).toBe('')
    expect(noteOf(merged, 'y')).toBe('设备 B 新写的')
  })

  it('校验', () => {
    expect(isNotes({ a: { text: 'x', at: 1 } })).toBe(true)
    expect(isNotes({ a: { text: 1, at: 1 } })).toBe(false)
    expect(isNotes({ a: { text: 'x' } })).toBe(false)
    expect(isNotes([])).toBe(false)
  })
})

describe('暂停标记', () => {
  it('恢复也留记录，合并时较新的切换胜出', () => {
    const on = setSuspended({}, 'a', true, 100)
    const off = setSuspended(on, 'a', false, 200)
    expect(isSuspended(on, 'a')).toBe(true)
    expect(isSuspended(mergeFlags(off, on), 'a')).toBe(false)
    expect(JSON.parse(serializeFlags(off))).toEqual({ a: { suspended: false, at: 200 } })
  })

  it('校验', () => {
    expect(isFlags({ a: { suspended: true, at: 1 } })).toBe(true)
    expect(isFlags({ a: { suspended: 'yes', at: 1 } })).toBe(false)
    expect(isFlags([])).toBe(false)
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
    expect(inlineMd('`x ** 2`')).toBe('<code>x ** 2</code>')  })

  it('inlineMd 把 $…$ 和 $$…$$ 交给 KaTeX；code 里的 $ 不当公式', () => {
    expect(inlineMd('$x^2$')).toContain('class="katex"')
    expect(inlineMd('$x^2$')).not.toContain('katex-display')
    expect(inlineMd('$$\\sum_i x_i$$')).toContain('katex-display')
    expect(inlineMd('`echo $HOME $PATH`')).toBe('<code>echo $HOME $PATH</code>')
    expect(inlineMd('a < $b$ **c**')).toMatch(/^a &lt; <span class="katex">.*<\/span> <b>c<\/b>$/)
  })
})
