/**
 * 私有笔记：内容放在私有仓库里，站点只放读取和渲染的代码。
 *
 * - 读：用这台设备上配置的 GitHub token 调 REST API。token 读不到私有仓库时，GitHub 对私有仓库一律回 404，
 *   这里统一报「没有权限」，不区分「仓库不存在」。
 * - 渲染：markdown-it 出 HTML，$公式$ 和 $$公式$$ 用 KaTeX；代码块里的 $ 不当公式。
 * - 文件之间的相对链接（xxx.md）改成页内的 #/路径，点了在本页切换。
 */
import MarkdownIt from 'markdown-it'
import katex from 'katex'
import { PRIVATE_REPO, type RepoConfig } from './github'

export { PRIVATE_REPO }

const API = 'https://api.github.com'

type Fetch = (url: string, init: RequestInit) => Promise<Response>

/** token 无效、过期，或者读不到私有仓库 */
export class NoAccessError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'NoAccessError'
  }
}

async function get(fetchFn: Fetch, token: string, url: string, accept: string): Promise<Response> {
  let res: Response
  try {
    res = await fetchFn(url, { headers: { Authorization: `Bearer ${token}`, Accept: accept } })
  } catch (e) {
    throw new Error(`连不上 GitHub：${(e as Error).message}`)
  }
  if (res.status === 401) throw new NoAccessError('GitHub token 无效或已过期')
  if (res.status === 403 || res.status === 404) {
    throw new NoAccessError(`这个 token 读不到 ${PRIVATE_REPO.owner}/${PRIVATE_REPO.name}（需要给它加上这个仓库的 Contents: Read）`)
  }
  if (!res.ok) throw new Error(`GitHub 返回 ${res.status}：${res.statusText}`)
  return res
}

/** 私有仓库里所有 .md 文件的路径，README 排最前，其余按路径排序 */
export async function listNotes(token: string, repo: RepoConfig = PRIVATE_REPO, fetchFn: Fetch = fetch): Promise<string[]> {
  const url = `${API}/repos/${repo.owner}/${repo.name}/git/trees/${repo.branch}?recursive=1`
  const res = await get(fetchFn, token, url, 'application/vnd.github+json')
  const json = (await res.json()) as { tree: { path: string; type: string }[] }
  return sortPaths(json.tree.filter((e) => e.type === 'blob' && e.path.endsWith('.md')).map((e) => e.path))
}

export function sortPaths(paths: string[]): string[] {
  const isReadme = (p: string) => p.toLowerCase() === 'readme.md'
  return [...paths].sort((a, b) => Number(isReadme(b)) - Number(isReadme(a)) || a.localeCompare(b))
}

/** 读一个文件的原文 */
export async function readNote(token: string, path: string, repo: RepoConfig = PRIVATE_REPO, fetchFn: Fetch = fetch): Promise<string> {
  const enc = path.split('/').map(encodeURIComponent).join('/')
  const url = `${API}/repos/${repo.owner}/${repo.name}/contents/${enc}?ref=${repo.branch}`
  const res = await get(fetchFn, token, url, 'application/vnd.github.raw+json')
  return res.text()
}

/** 第一个 # 标题；没有就用文件名 */
export function noteTitle(path: string, text?: string): string {
  const m = text?.match(/^#\s+(.+?)\s*#*\s*$/m)
  if (m) return m[1]
  return path.replace(/\.md$/, '').split('/').pop()!
}

/** 相对路径按当前文件所在目录解析，支持 ./ 和 ../ */
export function resolvePath(from: string, rel: string): string {
  const parts = from.split('/').slice(0, -1)
  for (const seg of rel.split('/')) {
    if (seg === '' || seg === '.') continue
    if (seg === '..') parts.pop()
    else parts.push(seg)
  }
  return parts.join('/')
}

const md = new MarkdownIt({ html: false, linkify: false, typographer: false })

const tex = (src: string, displayMode: boolean) => katex.renderToString(src, { displayMode, throwOnError: false, output: 'html' })

// 占位符只含字母数字，markdown 不会改动它
const PH = (i: number) => `MATHPH${i}XEND`

/** 把代码之外的公式换成占位符，返回替换后的文本和公式列表 */
export function protectMath(src: string): { text: string; math: { tex: string; display: boolean }[] } {
  const math: { tex: string; display: boolean }[] = []
  // 按 ``` 代码块和 `行内代码` 切开，只在代码之外找公式
  const pieces = src.split(/(```[\s\S]*?```|`[^`\n]+`)/)
  const text = pieces
    .map((piece, i) => {
      if (i % 2 === 1) return piece
      return piece
        .replace(/\$\$([\s\S]+?)\$\$/g, (_, t: string) => {
          math.push({ tex: t.trim(), display: true })
          return PH(math.length - 1)
        })
        .replace(/\$([^$\n]+?)\$/g, (_, t: string) => {
          math.push({ tex: t, display: false })
          return PH(math.length - 1)
        })
    })
    .join('')
  return { text, math }
}

/** 渲染一个私有笔记；links 里收集到的站内链接都改成 #/路径 */
export function renderNote(path: string, src: string): string {
  const { text, math } = protectMath(src)
  const env = { path }
  const tokens = md.parse(text, env)
  for (const t of tokens) {
    for (const c of t.children ?? []) {
      if (c.type !== 'link_open') continue
      const href = c.attrGet('href') ?? ''
      if (/^[a-z]+:|^#|^\//i.test(href) || !/\.md(#.*)?$/.test(href)) continue
      const [file, hash] = href.split('#')
      c.attrSet('href', `#/${resolvePath(path, file)}${hash ? `#${hash}` : ''}`)
    }
  }
  let html = md.renderer.render(tokens, md.options, env)
  // 单独成段的 $$...$$：去掉外面的 <p>，换成块级公式
  html = html.replace(/<p>MATHPH(\d+)XEND<\/p>/g, (_, i: string) => tex(math[+i].tex, true))
  html = html.replace(/MATHPH(\d+)XEND/g, (_, i: string) => tex(math[+i].tex, math[+i].display))
  return html
}

// ---------- 本机缓存 ----------
// 私有内容会缓存在这台设备的 localStorage 里，离线（手机出门）也能看；断开 GitHub 时一起清掉。

export const PRIVATE_CACHE_KEY = 'inferview:private-cache'

export interface PrivateCache {
  paths: string[]
  files: Record<string, string>
}

export function emptyCache(): PrivateCache {
  return { paths: [], files: {} }
}

export function parseCache(raw: string | null): PrivateCache {
  if (!raw) return emptyCache()
  try {
    const c = JSON.parse(raw) as PrivateCache
    if (Array.isArray(c.paths) && c.files && typeof c.files === 'object') return c
  } catch {
    // 坏了就当没有
  }
  return emptyCache()
}
