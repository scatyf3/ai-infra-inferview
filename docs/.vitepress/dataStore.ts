import { readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import type { Plugin } from 'vite'
import { SYNC_DOCS, type DocSpec } from '../../src/lib/syncDocs'

/**
 * 没配 GitHub token 时的退路：dev server 直接读写仓库里的数据文件（批注、闪卡复习记录等），靠 git 同步。
 * 配了 token 之后页面直接和 GitHub 上的 data 分支同步，不走这里（见 theme/sync.ts）。
 *
 * 可能有多个页面同时在写（比如手机通过 docs:dev:lan 连进来）：PUT 不直接覆盖，
 * 而是和文件里现有的按条目合并（较新的胜出），把合并结果返回给页面。
 * 生产构建里这些文件被打包进站点，只当初始数据用。
 */

export const DATA_ENDPOINT = '/__data'

const repoRoot = fileURLToPath(new URL('../../', import.meta.url))

function docEndpoint(spec: DocSpec<any>): Plugin {
  const file = repoRoot + spec.path
  return {
    name: `data-store:${spec.key}`,
    apply: 'serve',
    config: () => ({
      // 写文件不触发 HMR，否则每次保存页面都会刷新
      server: { watch: { ignored: [file.replace(/\\/g, '/')] } },
    }),
    configureServer(server) {
      server.middlewares.use(`${DATA_ENDPOINT}/${spec.key}`, async (req, res) => {
        try {
          if (req.method === 'GET') {
            const text = await readFile(file, 'utf8').catch(() => spec.serialize(spec.empty()))
            res.setHeader('Content-Type', 'application/json')
            res.end(text)
            return
          }
          if (req.method === 'PUT') {
            let body = ''
            for await (const chunk of req) {
              body += chunk
              if (body.length > 5_000_000) throw new Error('payload too large')
            }
            const data = JSON.parse(body)
            if (!spec.validate(data)) {
              res.statusCode = 400
              res.end(`invalid ${spec.key} payload`)
              return
            }
            const existing = JSON.parse(await readFile(file, 'utf8').catch(() => '{}'))
            const text = spec.serialize(spec.merge(spec.validate(existing) ? existing : spec.empty(), data))
            await writeFile(file, text)
            res.setHeader('Content-Type', 'application/json')
            res.end(text)
            return
          }
          res.statusCode = 405
          res.end()
        } catch (e) {
          res.statusCode = 500
          res.end(String(e))
        }
      })
    },
  }
}

export function dataStore(): Plugin[] {
  return SYNC_DOCS.map(docEndpoint)
}
