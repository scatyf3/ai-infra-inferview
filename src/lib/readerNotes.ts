/**
 * 读者划词批注的数据：页面路径 → 这一页的高亮列表（纯函数，组件见 ReaderNotes.vue）。
 *
 * 多台设备会各自改、再合并，所以每条都带一个修改时间 at，合并时按 id 取较新的那份。
 * 删除不是把条目拿掉，而是留一条 deleted 记录（删除记录）：它靠更新的 at 盖过别的设备上的旧副本，
 * 直接拿掉的话，别的设备一合并，删掉的高亮又会回来。
 */
import type { TextQuote } from './annotate'

export interface ReaderNote extends TextQuote {
  id: string
  note: string
  created: number
  /** 最近一次修改的时间戳（ms）；老数据没有这个字段，按 created 算 */
  at?: number
  /** 删除记录：页面上不显示，只用来在合并时盖过旧副本 */
  deleted?: boolean
}

export type ReaderNotes = Record<string, ReaderNote[]>

const stamp = (n: ReaderNote) => n.at ?? n.created

/** 页面上要显示的（去掉删除记录） */
export const liveNotes = (list: ReaderNote[] | undefined): ReaderNote[] => (list ?? []).filter((n) => !n.deleted)

/** 按 id 合并，b 里较新的（at 更大）胜出；一样新时保留 a 的 */
export function mergeReaderNotes(a: ReaderNotes, b: ReaderNotes): ReaderNotes {
  const out: ReaderNotes = { ...a }
  for (const [path, list] of Object.entries(b)) {
    const byId = new Map((out[path] ?? []).map((n) => [n.id, n]))
    for (const n of list) {
      const cur = byId.get(n.id)
      if (!cur || stamp(n) > stamp(cur)) byId.set(n.id, n)
    }
    out[path] = [...byId.values()]
  }
  return out
}

export function addNote(s: ReaderNotes, path: string, n: ReaderNote): ReaderNotes {
  return { ...s, [path]: [...(s[path] ?? []), n] }
}

export function editNote(s: ReaderNotes, path: string, id: string, note: string, now = Date.now()): ReaderNotes {
  return { ...s, [path]: (s[path] ?? []).map((n) => (n.id === id ? { ...n, note, at: now } : n)) }
}

/** 删除：留下删除记录，原文锚点保留（排查用），批注内容清掉 */
export function removeNotes(s: ReaderNotes, path: string, ids: Iterable<string>, now = Date.now()): ReaderNotes {
  const drop = new Set(ids)
  return {
    ...s,
    [path]: (s[path] ?? []).map((n) => (drop.has(n.id) && !n.deleted ? { ...n, note: '', deleted: true, at: now } : n)),
  }
}

const isNum = (v: unknown) => typeof v === 'number' && Number.isFinite(v)

export function isReaderNote(v: unknown): v is ReaderNote {
  if (!v || typeof v !== 'object') return false
  const n = v as ReaderNote
  return (
    typeof n.id === 'string' &&
    !!n.id &&
    typeof n.exact === 'string' &&
    typeof n.prefix === 'string' &&
    typeof n.suffix === 'string' &&
    typeof n.note === 'string' &&
    isNum(n.created) &&
    (n.at === undefined || isNum(n.at)) &&
    (n.deleted === undefined || typeof n.deleted === 'boolean')
  )
}

export function isReaderNotes(v: unknown): v is ReaderNotes {
  return (
    !!v &&
    typeof v === 'object' &&
    !Array.isArray(v) &&
    Object.values(v).every((list) => Array.isArray(list) && list.every(isReaderNote))
  )
}

/** 路径、条目都排好序，保证 diff 稳定；删除记录要留着（理由见文件头） */
export function serializeReaderNotes(s: ReaderNotes): string {
  const out: ReaderNotes = {}
  for (const path of Object.keys(s).sort()) {
    const list = s[path]
    if (list.length) out[path] = [...list].sort((a, b) => a.created - b.created || a.id.localeCompare(b.id))
  }
  return JSON.stringify(out, null, 2) + '\n'
}
