// LeetGPU 题解页的文件名和骨架。不依赖别名，dev 服务器插件（docs/.vitepress/leetgpuNewPage.ts）也能直接引用。

/** 和 leetgpu.com 前端路由同一个规则：/challenges/:name 用 slug(title) 反查题目 */
export function slugOf(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
}

/** 新题解页的 markdown：frontmatter 挂上题号，看板就会把这题算作「做过」 */
export function solutionPageSkeleton(c: { id: number; title: string }): string {
  const slug = slugOf(c.title)
  return `---
title: ${c.title}
status: draft
tags: [leetgpu]
difficulty: 2
order: 99
related: []
stack: []
leetgpu: [${c.id}]
---

# ${c.title}

> [LeetGPU #${c.id}](https://leetgpu.com/challenges/${slug}) ·

## 题解

\`\`\`python
\`\`\`

## 这题的坑

1.

其余见 [通用语法坑](./#通用语法坑)。
`
}
