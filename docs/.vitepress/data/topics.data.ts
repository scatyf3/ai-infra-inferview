import { createContentLoader } from 'vitepress'
// data loader 在 vite 别名之外单独打包，这里只能走相对路径
import { isFamiliarity } from '../../../src/lib/fam'
import leetgpu from '../../../src/data/leetgpu-challenges.json'
import { LAYER_IDS, layerOfTopic } from '../layers'

const LEETGPU_IDS = new Set(leetgpu.map((c) => c.id))

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
  /** 熟练度 0–4（0 最熟），null = 未评；语义见 src/lib/fam.ts */
  familiarity: number | null
  /** 对应的 LeetGPU 题号，见 src/data/leetgpu-challenges.json */
  leetgpu: number[]
  /** 挂在推理栈分层图的哪些位置：字符串 = 小主题 id，数字 = 整层；见 ../layers.ts */
  stack: (string | number)[]
  /** 由 stack 推出来的层号，升序去重；空 = 不在推理栈上 */
  layers: number[]
}

declare const data: Topic[]
export { data }

export default createContentLoader('*/*.md', {
  transform(raw): Topic[] {
    return raw
      // docs/stack/ 是分层图小主题的介绍页，由 stack.data.ts 收集，不算知识库文章
      .filter((p) => !p.url.startsWith('/stack/'))
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
        if (!isFamiliarity(fm.familiarity)) {
          throw new Error(`[topics] invalid familiarity "${fm.familiarity}" in ${p.url} (expected 0 | 1 | 1.5 | 2 | 3 | 3.5 | 4)`)
        }
        const stack: (string | number)[] = fm.stack ?? []
        for (const s of stack) {
          const ok = typeof s === 'number' ? LAYER_IDS.has(s) : layerOfTopic.has(s)
          if (!ok) throw new Error(`[topics] unknown stack entry "${s}" in ${p.url} (expected a layer 0–8 or a topic id from layers.ts)`)
        }
        const layers = [...new Set(stack.map((s) => (typeof s === 'number' ? s : layerOfTopic.get(s)!)))]
        const lg: number[] = fm.leetgpu ?? []
        for (const id of lg) {
          if (!LEETGPU_IDS.has(id)) throw new Error(`[topics] unknown leetgpu id ${id} in ${p.url}`)
        }
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
          familiarity: fm.familiarity ?? null,
          leetgpu: lg,
          stack,
          layers: layers.sort((a, b) => a - b),
        }
      })
      .sort((a, b) => a.domain.localeCompare(b.domain) || a.order - b.order || a.slug.localeCompare(b.slug))
  },
})
