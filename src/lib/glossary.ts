/**
 * 术语悬浮提示：正文里每页第一次出现的术语（MFU、ZeRO……）加虚线下划线，悬停显示一句话定义。
 * 这里只放纯逻辑（范围判断、文本切分），markdown-it 插件在 docs/.vitepress/markdown/glossary.ts。
 */
export interface GlossaryEntry {
  /** 提示框里的标题 */
  term: string
  /** 正文里要匹配的写法，区分大小写，按 ASCII 单词边界匹配 */
  match: string[]
  /** 英文全称 */
  full?: string
  /** 一句话定义，按 inline markdown 渲染，可以写 $公式$、`code` */
  def: string
  /** 详细讲解的页面，站内绝对路径，不带 .md */
  link?: string
}

export type Segment = string | { entry: GlossaryEntry; text: string }

/** 先只在训练栈这部分页面开：docs/train/ 和训练栈的格子页 */
export function inGlossaryScope(relativePath: string): boolean {
  return /^train\//.test(relativePath) || /^stack\/(tr-|train-layer-)/.test(relativePath)
}

/** 'train/compute-mfu.md' → '/train/compute-mfu'，用来判断术语的详解页是不是当前页 */
export function pageOf(relativePath: string): string {
  return '/' + relativePath.replace(/\.md$/, '').replace(/(^|\/)index$/, '$1')
}

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** 所有写法拼成一个正则；长的在前，免得 FSDP2 被 FSDP 抢先匹配 */
export function buildMatcher(entries: GlossaryEntry[]) {
  const byText = new Map<string, GlossaryEntry>()
  for (const e of entries) for (const m of e.match) byText.set(m, e)
  const alts = [...byText.keys()].sort((a, b) => b.length - a.length).map(escape)
  const re = new RegExp(`(?<![A-Za-z0-9_])(${alts.join('|')})(?![A-Za-z0-9_])`, 'g')
  return { re, byText }
}

/**
 * 把一段文本切成普通文字和术语。`seen` 记录本页已经标过的术语，每个只标第一次；
 * `skip` 里的术语不标（比如它的详解页就是当前页）。
 */
export function splitTerms(
  text: string,
  matcher: ReturnType<typeof buildMatcher>,
  seen: Set<GlossaryEntry>,
  skip: (e: GlossaryEntry) => boolean = () => false,
): Segment[] {
  const out: Segment[] = []
  let last = 0
  for (const m of text.matchAll(matcher.re)) {
    const entry = matcher.byText.get(m[1])!
    if (seen.has(entry) || skip(entry)) continue
    seen.add(entry)
    if (m.index > last) out.push(text.slice(last, m.index))
    out.push({ entry, text: m[1] })
    last = m.index + m[1].length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}
