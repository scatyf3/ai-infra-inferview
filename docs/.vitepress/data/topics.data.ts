import { createContentLoader } from 'vitepress'

export type Status = 'todo' | 'draft' | 'reviewed'
export const STATUSES: Status[] = ['todo', 'draft', 'reviewed']

export interface Topic {
  url: string
  domain: string
  slug: string
  title: string
  status: Status
  tags: string[]
  difficulty: number
  related: string[]
  order: number
  isIndex: boolean
}

declare const data: Topic[]
export { data }

export default createContentLoader('*/*.md', {
  transform(raw): Topic[] {
    return raw
      .map((p) => {
        // url 形如 /inference/memory-accounting 或 /inference/
        const parts = p.url.split('/').filter(Boolean)
        const domain = parts[0]
        const slug = parts[1] ?? 'index'
        const isIndex = slug === 'index'
        const fm = p.frontmatter
        const status: Status = fm.status ?? 'todo'
        if (!isIndex && !STATUSES.includes(status)) {
          throw new Error(`[topics] invalid status "${status}" in ${p.url} (expected ${STATUSES.join(' | ')})`)
        }
        if (!fm.title) throw new Error(`[topics] missing title in ${p.url}`)
        return {
          url: p.url,
          domain,
          slug,
          title: fm.title,
          status,
          tags: fm.tags ?? [],
          difficulty: fm.difficulty ?? 2,
          related: fm.related ?? [],
          order: fm.order ?? 99,
          isIndex,
        }
      })
      .sort((a, b) => a.domain.localeCompare(b.domain) || a.order - b.order || a.slug.localeCompare(b.slug))
  },
})
