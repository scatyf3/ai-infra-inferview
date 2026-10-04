import { execSync } from 'node:child_process'
import { readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import type { Plugin } from 'vite'
import { SYNC_DOCS } from '../../src/lib/syncDocs'
import { StaleError, type RemoteFile, type RemoteSnapshot } from '../../src/lib/sync'
import { AuthError, DATA_REPO, githubRemote } from '../../src/lib/github'

/**
 * dev server 上的读者数据接口（批注、闪卡记录），页面没连 GitHub token 时走这里（见 theme/sync.ts）。
 *
 * - 电脑上有 GitHub 登录（环境变量 GH_TOKEN / GITHUB_TOKEN，或者 `gh auth login` 过）：
 *   直接读写 GitHub 上的 data 分支，和手机、线上站点是同一份数据。token 留在服务端，不发给页面。
 * - 没有登录：退回读写本机仓库里的 json，靠 git 同步。
 *
 * 接口和 GitHub 那边一个形状（src/lib/sync.ts 的 Remote）：
 *   GET /__data → { backend, texts, version }
 *   PUT /__data { files, version, summary, device } → 204；版本对不上 409，token 不行 401
 */

export const DATA_ENDPOINT = '/__data'

const repoRoot = fileURLToPath(new URL('../../', import.meta.url))
const PATHS = SYNC_DOCS.map((d) => d.path)

/**
 * 依次找：环境变量 → gh 的登录 → git 凭据管理器里存的 github.com 凭据（git push 用的那份）。
 * dev server 可能是从 PATH 里没有 gh 的 shell 启动的，所以还要有 git 这条退路；全程不弹任何登录框。
 */
function findToken(): string | null {
  const env = process.env.GH_TOKEN || process.env.GITHUB_TOKEN
  if (env) return env
  const quiet = { stdio: ['pipe', 'pipe', 'ignore'] as ('pipe' | 'ignore')[], timeout: 8000 }
  try {
    const t = execSync('gh auth token', quiet).toString().trim()
    if (t) return t
  } catch {}
  try {
    const out = execSync('git credential fill', {
      ...quiet,
      input: 'protocol=https\nhost=github.com\n\n',
      env: { ...process.env, GIT_TERMINAL_PROMPT: '0', GCM_INTERACTIVE: 'never' },
    }).toString()
    const t = /^password=(.+)$/m.exec(out)?.[1]?.trim()
    if (t) return t
  } catch {}
  return null
}

async function readBody(req: AsyncIterable<unknown>): Promise<string> {
  let body = ''
  for await (const chunk of req) {
    body += chunk
    if (body.length > 5_000_000) throw new Error('payload too large')
  }
  return body
}

/** 页面发来的文件只能是那几个数据文件，内容要能通过校验 */
function checkFiles(files: unknown): RemoteFile[] {
  if (!Array.isArray(files) || !files.length) throw new Error('files 为空')
  return files.map((f) => {
    const spec = SYNC_DOCS.find((d) => d.path === f?.path)
    if (!spec || typeof f.text !== 'string') throw new Error(`不认识的文件 ${f?.path}`)
    if (!spec.validate(JSON.parse(f.text))) throw new Error(`${f.path} 格式不对`)
    return { path: f.path, text: f.text }
  })
}

// ---------- 没登录时：本机仓库文件 ----------

async function readFiles(): Promise<RemoteSnapshot> {
  const texts: Record<string, string | null> = {}
  for (const p of PATHS) texts[p] = await readFile(repoRoot + p, 'utf8').catch(() => null)
  return { texts, version: null }
}

/** 没有版本号可比，写之前和文件里现有的按条目合并（较新的胜出） */
async function writeFiles(files: RemoteFile[]) {
  for (const f of files) {
    const spec = SYNC_DOCS.find((d) => d.path === f.path)!
    const existing = JSON.parse(await readFile(repoRoot + f.path, 'utf8').catch(() => '{}'))
    const merged = spec.merge(spec.validate(existing) ? existing : spec.empty(), JSON.parse(f.text))
    await writeFile(repoRoot + f.path, spec.serialize(merged))
  }
}

export function dataStore(): Plugin {
  return {
    name: 'data-store',
    apply: 'serve',
    config: () => ({
      // 退回本机文件时，写文件不触发 HMR，否则每次保存页面都会刷新
      server: { watch: { ignored: PATHS.map((p) => (repoRoot + p).replace(/\\/g, '/')) } },
    }),
    configureServer(server) {
      const token = findToken()
      const backend = token ? 'github' : 'files'
      server.config.logger.info(
        token
          ? `  读者数据：读写 GitHub 上 ${DATA_REPO.owner}/${DATA_REPO.name} 的 ${DATA_REPO.branch} 分支（用本机的 GitHub 登录）`
          : '  读者数据：没找到 GitHub 登录（gh auth login 或 GH_TOKEN），读写本机仓库里的 src/data/*.json',
      )

      server.middlewares.use(DATA_ENDPOINT, async (req, res) => {
        const send = (status: number, body?: unknown) => {
          res.statusCode = status
          if (body === undefined) return res.end()
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify(body))
        }
        try {
          if (req.method === 'GET') {
            const snap = token ? await githubRemote({ token }).read(PATHS) : await readFiles()
            return send(200, { backend, ...snap })
          }
          if (req.method === 'PUT') {
            const { files, version, summary, device } = JSON.parse(await readBody(req))
            const checked = checkFiles(files)
            if (token) {
              const dev = typeof device === 'string' ? device.slice(0, 20) : undefined
              await githubRemote({ token, device: dev }).write(checked, version ?? null, String(summary ?? '').slice(0, 80))
            } else {
              await writeFiles(checked)
            }
            return send(204)
          }
          return send(405)
        } catch (e) {
          if (e instanceof StaleError) return send(409, { error: e.message })
          if (e instanceof AuthError) return send(401, { error: e.message })
          return send(500, { error: String((e as Error).message ?? e) })
        }
      })
    },
  }
}
