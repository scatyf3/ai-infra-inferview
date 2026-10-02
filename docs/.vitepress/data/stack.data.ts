import { createContentLoader } from 'vitepress'
import { LAYER_IDS, layerOfTopic } from '../layers'

/**
 * docs/stack/<id>.md：分层图里每个格子的基础介绍，构建时渲染成 HTML 供首页面板直接展示。
 * id 是小主题 id（如 kv-paged），或 layer-N 表示第 N 层的总述。
 */
export interface StackIntro {
  id: string
  title: string
  url: string
  /** 去掉了页面 h1 的正文 HTML */
  html: string
}

declare const data: StackIntro[]
export { data }

export default createContentLoader('stack/*.md', {
  render: true,
  transform(raw): StackIntro[] {
    return raw.map((p) => {
      const id = p.url.split('/').filter(Boolean)[1]
      const layerMatch = id.match(/^layer-(\d+)$/)
      const ok = layerMatch ? LAYER_IDS.has(Number(layerMatch[1])) : layerOfTopic.has(id)
      if (!ok) throw new Error(`[stack] docs/stack/${id}.md 不对应 layers.ts 里的任何层或小主题`)
      return {
        id,
        title: p.frontmatter.title ?? id,
        url: p.url,
        html: (p.html ?? '').replace(/^\s*<h1[\s\S]*?<\/h1>/, '').trim(),
      }
    })
  },
})
