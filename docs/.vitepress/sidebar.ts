import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import matter from 'gray-matter'
import type { DefaultTheme } from 'vitepress'
import { domains } from './domains'

const docsRoot = fileURLToPath(new URL('..', import.meta.url))

interface Entry {
  slug: string
  title: string
  order: number
}

function readDomain(dir: string): { index?: Entry; topics: Entry[] } {
  const abs = path.join(docsRoot, dir)
  if (!fs.existsSync(abs)) return { topics: [] }
  const topics: Entry[] = []
  let index: Entry | undefined
  for (const file of fs.readdirSync(abs)) {
    if (!file.endsWith('.md')) continue
    const fm = matter(fs.readFileSync(path.join(abs, file), 'utf8')).data
    const slug = file.replace(/\.md$/, '')
    const entry: Entry = { slug, title: fm.title ?? slug, order: fm.order ?? 99 }
    if (slug === 'index') index = entry
    else topics.push(entry)
  }
  topics.sort((a, b) => a.order - b.order || a.slug.localeCompare(b.slug))
  return { index, topics }
}

/** 从各领域目录的 frontmatter 生成 sidebar。 */
export function buildSidebar(): DefaultTheme.Sidebar {
  return [...domains]
    .sort((a, b) => a.order - b.order)
    .map((d) => {
      const { topics } = readDomain(d.dir)
      return {
        text: d.label,
        link: `/${d.dir}/`,
        collapsed: false,
        items: topics.map((t) => ({ text: t.title, link: `/${d.dir}/${t.slug}` })),
      }
    })
}
