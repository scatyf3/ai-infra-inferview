import type MarkdownIt from 'markdown-it'
import type Token from 'markdown-it/lib/token.mjs'
import { glossary } from '../../../src/data/glossary'
import { buildMatcher, inGlossaryScope, pageOf, splitTerms, type GlossaryEntry } from '../../../src/lib/glossary'

const NESTED = Symbol('glossary-nested')

/**
 * 术语悬浮提示：每页第一次在正文里出现的术语，渲染成
 *   <span class="term">MFU<span class="term-note">定义 · 详见</span></span>
 * 不碰标题、链接文字、代码、公式和作者批注里的字。术语表在 src/data/glossary.ts。
 */
export function glossaryPlugin(md: MarkdownIt) {
  const matcher = buildMatcher(glossary)

  // 定义按 inline markdown 解析成 token，链接走 VitePress 自己的 link 渲染（会补上 base）。
  // 每次重新解析：渲染时 link 插件会原地改 href，缓存的 token 复用会重复加 base
  const noteTokens = (e: GlossaryEntry, make: (c: string) => Token) => {
    const src = (e.full ? `**${e.term}** · ${e.full}\n` : `**${e.term}**\n`) + e.def + (e.link ? ` [详见 →](${e.link})` : '')
    const parsed = md.parseInline(src, { [NESTED]: true })[0].children ?? []
    return [make('<span class="term-note" role="tooltip">'), ...parsed.map((t) => (t.type === 'softbreak' ? make('<br>') : t)), make('</span>')]
  }

  md.core.ruler.push('glossary', (state) => {
    const env = state.env ?? {}
    if (env[NESTED] || !inGlossaryScope(env.relativePath ?? '')) return
    const here = pageOf(env.relativePath)
    const seen = new Set<GlossaryEntry>()
    const skip = (e: GlossaryEntry) => e.link === here

    const blocks = state.tokens
    for (let i = 0; i < blocks.length; i++) {
      const block = blocks[i]
      if (block.type !== 'inline' || !block.children) continue
      if (blocks[i - 1]?.type === 'heading_open') continue

      const make = (content: string) => {
        const t = new state.Token('html_inline', '', 0)
        t.content = content
        return t
      }
      const out: Token[] = []
      let inLink = 0
      let inMark = 0
      for (const t of block.children) {
        if (t.type === 'link_open') inLink++
        else if (t.type === 'link_close') inLink--
        else if (t.type === 'html_inline' && /^<mark\b/.test(t.content)) inMark++
        else if (t.type === 'html_inline' && t.content === '</mark>') inMark--

        if (t.type !== 'text' || inLink || inMark) {
          out.push(t)
          continue
        }
        for (const seg of splitTerms(t.content, matcher, seen, skip)) {
          if (typeof seg === 'string') {
            const text = new state.Token('text', '', 0)
            text.content = seg
            out.push(text)
          } else {
            out.push(make('<span class="term" tabindex="0">'))
            const text = new state.Token('text', '', 0)
            text.content = seg.text
            out.push(text, ...noteTokens(seg.entry, make), make('</span>'))
          }
        }
      }
      block.children = out
    }
  })
}
