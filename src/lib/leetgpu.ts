import raw from '@data/leetgpu-challenges.json'

// 题目清单是 https://api.leetgpu.com/api/v1/challenges 的快照，只留 id / 标题 / 难度。
// frontmatter 里按 id 引用（`leetgpu: [80, 12]`），标题改了也不用动各题页面。

export type LeetGPUDifficulty = 'easy' | 'medium' | 'hard'

export interface LeetGPUChallenge {
  id: number
  title: string
  difficulty: LeetGPUDifficulty
  access: string
}

export const challenges = raw as LeetGPUChallenge[]
export const challengeById = new Map(challenges.map((c) => [c.id, c]))

/** 和 leetgpu.com 前端路由同一个规则：/challenges/:name 用 slug(title) 反查题目 */
export function slugOf(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
}

export const urlOf = (c: LeetGPUChallenge) => `https://leetgpu.com/challenges/${slugOf(c.title)}`
