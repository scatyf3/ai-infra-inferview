import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import matter from 'gray-matter'
import type { DefaultTheme } from 'vitepress'
import { domains } from './domains'
import { layerOfTopic } from './layers'
import { dirSection, overrides, sectionOrder, sections } from './sections'

const docsRoot = fileURLToPath(new URL('..', import.meta.url))

interface Entry {
  /** 相对 docs/ 的路径，不带 .md，如 inference/flash-attention */
  key: string
  dir: string
  title: string
  order: number
  stack: (string | number)[]
}

function readEntry(key: string): Entry | undefined {
  const file = path.join(docsRoot, `${key}.md`)
  if (!fs.existsSync(file)) return undefined
  const fm = matter(fs.readFileSync(file, 'utf8')).data
  return {
    key,
    dir: key.split('/')[0],
    title: fm.title ?? key,
    order: fm.order ?? 99,
    stack: Array.isArray(fm.stack) ? fm.stack : [],
  }
}

function readDomain(dir: string): Entry[] {
  const abs = path.join(docsRoot, dir)
  if (!fs.existsSync(abs)) return []
  return fs
    .readdirSync(abs)
    .filter((f) => f.endsWith('.md') && f !== 'index.md')
    .map((f) => readEntry(`${dir}/${f.replace(/\.md$/, '')}`)!)
}

/** 文章归哪一组，规则见 sections.ts */
function sectionOf(e: Entry): string {
  if (overrides[e.key]) return overrides[e.key]
  if (dirSection[e.dir]) return dirSection[e.dir]
  const first = e.stack[0]
  if (typeof first === 'number') return `L${first}`
  if (typeof first === 'string' && layerOfTopic.has(first)) return `L${layerOfTopic.get(first)}`
  return 'overview'
}

/** 按首页推理栈分层生成 sidebar；空组不显示。 */
export function buildSidebar(): DefaultTheme.Sidebar {
  const domainRank = new Map(domains.map((d) => [d.dir, d.order]))
  const entries = domains.flatMap((d) => readDomain(d.dir))
  const byKey = new Map(entries.map((e) => [e.key, e]))

  const groups = new Map<string, Entry[]>(sections.map((s) => [s.id, []]))
  for (const e of entries) groups.get(sectionOf(e))?.push(e)

  return sections
    .map((s) => {
      // 目录里的文章只在它真正归属的组里出现；/stack/ 这类不在目录里的页面按列表放
      const listed = (sectionOrder[s.id] ?? [])
        .map((k) => (byKey.has(k) ? (sectionOf(byKey.get(k)!) === s.id ? byKey.get(k) : undefined) : readEntry(k)))
        .filter((e): e is Entry => !!e)
      const listedKeys = new Set(listed.map((e) => e.key))
      const rest = groups
        .get(s.id)!
        .filter((e) => !listedKeys.has(e.key))
        .sort((a, b) => (domainRank.get(a.dir) ?? 99) - (domainRank.get(b.dir) ?? 99) || a.order - b.order || a.key.localeCompare(b.key))
      const items = [...listed, ...rest].map((e) => ({ text: e.title, link: `/${e.key}` }))
      return { text: s.label, link: s.link, collapsed: false, items }
    })
    .filter((g) => g.items.length)
}
