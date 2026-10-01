import type MarkdownIt from 'markdown-it'

/**
 * 作者侧高亮 / 批注语法：
 *   ==重点==            → <mark>重点</mark>
 *   ==重点=={批注内容}   → 悬停 / 聚焦显示批注的 <mark class="anno">
 * 两部分都按 inline markdown 解析，可以放 `code`、**粗体**、链接、公式。
 */
export function markPlugin(md: MarkdownIt) {
  md.inline.ruler.before('emphasis', 'mark_anno', (state, silent) => {
    const { src, pos: start, posMax: max } = state
    if (src.charCodeAt(start) !== 0x3d || src.charCodeAt(start + 1) !== 0x3d) return false

    const close = src.indexOf('==', start + 2)
    if (close < 0 || close + 2 > max) return false
    const body = src.slice(start + 2, close)
    if (!body || /^\s|\s$/.test(body)) return false

    let end = close + 2
    let note: string | null = null
    if (src.charCodeAt(end) === 0x7b /* { */) {
      let depth = 0
      for (let i = end; i < max; i++) {
        const c = src[i]
        if (c === '\\') i++
        else if (c === '{') depth++
        else if (c === '}' && --depth === 0) {
          note = src.slice(end + 1, i)
          end = i + 1
          break
        }
      }
    }

    if (!silent) {
      const html = (content: string) => {
        state.push('html_inline', '', 0).content = content
      }
      const inline = (s: string) => {
        // 解析到独立数组再追加：嵌套 parse 的后处理会合并相邻 text token，
        // 直接写进 state.tokens 会打乱外层 emphasis 记录的下标
        const out: typeof state.tokens = []
        state.md.inline.parse(s, state.md, state.env, out)
        for (const t of out) {
          state.tokens.push(t)
          state.tokens_meta.push(null)
        }
      }
      html(note == null ? '<mark>' : '<mark class="anno" tabindex="0">')
      inline(body)
      if (note != null) {
        html('<span class="anno-note" role="tooltip">')
        inline(note)
        html('</span>')
      }
      html('</mark>')
    }
    state.pos = end
    return true
  })
}
