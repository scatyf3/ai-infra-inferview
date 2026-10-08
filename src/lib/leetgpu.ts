import { slugOf } from './leetgpuPage'
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

export { slugOf }

export const urlOf = (c: LeetGPUChallenge) => `https://leetgpu.com/challenges/${slugOf(c.title)}`
