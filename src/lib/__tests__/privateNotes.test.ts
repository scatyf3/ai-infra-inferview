import { describe, expect, it } from 'vitest'
import {
  NoAccessError,
  listNotes,
  noteTitle,
  parseCache,
  protectMath,
  readNote,
  renderNote,
  resolvePath,
  sortPaths,
} from '@lib/privateNotes'

const repo = { owner: 'o', name: 'r', branch: 'main' }

function fakeFetch(status: number, body: unknown, seen: { url?: string; init?: RequestInit } = {}) {
  return async (url: string, init: RequestInit) => {
    seen.url = url
    seen.init = init
    const text = typeof body === 'string' ? body : JSON.stringify(body)
    return new Response(text, { status })
  }
}

describe('私有笔记：渲染', () => {
  it('代码里的 $ 不当公式，代码外的公式用 KaTeX', () => {
    const { text, math } = protectMath('价格 `$5` 和 $x^2$\n\n```py\na = "$b$"\n```\n\n$$\\frac{1}{2}$$')
    expect(math.map((m) => m.tex)).toEqual(['x^2', '\\frac{1}{2}'])
    expect(text).toContain('`$5`')
    expect(text).toContain('a = "$b$"')
    const html = renderNote('a.md', '行内 $x^2$\n\n$$\\frac{1}{2}$$\n\n`$y$`')
    expect(html).toContain('class="katex"')
    expect(html).toContain('katex-display')
    expect(html).toContain('<code>$y$</code>')
    expect(html).not.toContain('MATHPH')
  })

  it('相对 .md 链接改成页内的 #/路径，外链不动', () => {
    const html = renderNote('companies/a.md', '[项目](../projects.md) [同目录](b.md#x) [外链](https://x.com/a.md)')
    expect(html).toContain('href="#/projects.md"')
    expect(html).toContain('href="#/companies/b.md#x"')
    expect(html).toContain('href="https://x.com/a.md"')
  })

  it('不渲染原始 HTML', () => {
    expect(renderNote('a.md', '<script>alert(1)</script>')).not.toContain('<script>')
  })

  it('路径、排序、标题', () => {
    expect(resolvePath('a/b/c.md', '../d.md')).toBe('a/d.md')
    expect(resolvePath('c.md', './d.md')).toBe('d.md')
    expect(sortPaths(['z.md', 'README.md', 'a/b.md'])).toEqual(['README.md', 'a/b.md', 'z.md'])
    expect(noteTitle('x/intro.md', '# 自我介绍\n\n...')).toBe('自我介绍')
    expect(noteTitle('x/intro.md')).toBe('intro')
  })

  it('缓存坏了就当没有', () => {
    expect(parseCache('{bad')).toEqual({ paths: [], files: {} })
    expect(parseCache(JSON.stringify({ paths: ['a.md'], files: { 'a.md': 'x' } })).paths).toEqual(['a.md'])
  })
})

describe('私有笔记：读取', () => {
  it('列出 .md 文件，带 token', async () => {
    const seen: { url?: string; init?: RequestInit } = {}
    const tree = { tree: [{ path: 'README.md', type: 'blob' }, { path: 'a', type: 'tree' }, { path: 'a/x.md', type: 'blob' }, { path: 'img.png', type: 'blob' }] }
    const paths = await listNotes('t0k', repo, fakeFetch(200, tree, seen))
    expect(paths).toEqual(['README.md', 'a/x.md'])
    expect(seen.url).toBe('https://api.github.com/repos/o/r/git/trees/main?recursive=1')
    expect((seen.init?.headers as Record<string, string>).Authorization).toBe('Bearer t0k')
  })

  it('读原文，路径逐段编码', async () => {
    const seen: { url?: string } = {}
    const text = await readNote('t', '面经/a b.md', repo, fakeFetch(200, '# hi', seen))
    expect(text).toBe('# hi')
    expect(seen.url).toBe('https://api.github.com/repos/o/r/contents/%E9%9D%A2%E7%BB%8F/a%20b.md?ref=main')
  })

  it('读不到私有仓库（404）、token 无效（401）都报没有权限', async () => {
    await expect(listNotes('t', repo, fakeFetch(404, { message: 'Not Found' }))).rejects.toBeInstanceOf(NoAccessError)
    await expect(readNote('t', 'a.md', repo, fakeFetch(401, {}))).rejects.toBeInstanceOf(NoAccessError)
  })
})
