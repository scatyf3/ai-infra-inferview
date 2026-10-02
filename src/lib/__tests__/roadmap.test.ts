import { describe, expect, it } from 'vitest'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { layers, layerOfTopic } from '../../../docs/.vitepress/layers'
import { stages } from '../../../docs/.vitepress/roadmap'

const docs = fileURLToPath(new URL('../../../docs', import.meta.url))

describe('roadmap', () => {
  it('每个小主题恰好属于一个阶段', () => {
    const seen = stages.flatMap((s) => s.topics)
    for (const id of seen) expect(layerOfTopic.has(id), id).toBe(true)
    expect(new Set(seen).size).toBe(seen.length)
    const all = layers.flatMap((l) => l.topics.map((t) => t.id))
    expect([...seen].sort()).toEqual([...all].sort())
  })

  it('引用的文章都存在', () => {
    for (const s of stages) {
      for (const url of s.articles) expect(existsSync(`${docs}${url}.md`), url).toBe(true)
    }
  })
})
