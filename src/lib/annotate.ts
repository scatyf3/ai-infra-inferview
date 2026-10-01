/**
 * 高亮与批注的纯逻辑部分（无 DOM，可单测）：
 * - renderMarks：把 JSON 数据里的 `==重点==` / `==重点=={批注}` 转成 HTML，给 widget 用 v-html 渲染
 * - makeQuote / locateQuote：读者划词笔记的文本锚点（exact + 前后文），文档改动后仍能重新定位
 */

const ESC: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ESC[c])
}

// 内容两端不能是空白，避免把 `a == b` 误识别成高亮
const MARK_RE = /==(\S|\S[\s\S]*?\S)==(?:\{([^{}]*)\})?/g

export function renderMarks(text: string): string {
  return escapeHtml(text).replace(MARK_RE, (_, body: string, note?: string) =>
    note == null
      ? `<mark>${body}</mark>`
      : `<mark class="anno" tabindex="0">${body}<span class="anno-note" role="tooltip">${note}</span></mark>`,
  )
}

export interface TextQuote {
  exact: string
  prefix: string
  suffix: string
}

export const QUOTE_CONTEXT = 32

export function makeQuote(full: string, start: number, end: number, context = QUOTE_CONTEXT): TextQuote {
  return {
    exact: full.slice(start, end),
    prefix: full.slice(Math.max(0, start - context), start),
    suffix: full.slice(end, end + context),
  }
}

function commonPrefixLen(a: string, b: string): number {
  let i = 0
  while (i < a.length && i < b.length && a[i] === b[i]) i++
  return i
}

function commonSuffixLen(a: string, b: string): number {
  let i = 0
  while (i < a.length && i < b.length && a[a.length - 1 - i] === b[b.length - 1 - i]) i++
  return i
}

/** 在 full 中找 exact 的所有出现，按前后文匹配长度挑最像的一处；找不到返回 null */
export function locateQuote(full: string, q: TextQuote): { start: number; end: number } | null {
  if (!q.exact) return null
  let best = -1
  let bestScore = -1
  for (let i = full.indexOf(q.exact); i !== -1; i = full.indexOf(q.exact, i + 1)) {
    const end = i + q.exact.length
    const score =
      commonSuffixLen(full.slice(Math.max(0, i - q.prefix.length), i), q.prefix) +
      commonPrefixLen(full.slice(end, end + q.suffix.length), q.suffix)
    if (score > bestScore) {
      best = i
      bestScore = score
    }
  }
  return best < 0 ? null : { start: best, end: best + q.exact.length }
}
