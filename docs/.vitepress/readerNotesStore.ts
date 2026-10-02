import { readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import type { Plugin } from 'vite'

/**
 * dev 时把读者批注读写到仓库里的 src/data/reader-notes.json，靠 git 在设备间同步。
 * 生产构建里这个文件被打包进站点（只读），见 ReaderNotes.vue。
 */

export const NOTES_ENDPOINT = '/__reader-notes'
export const NOTES_FILE = fileURLToPath(new URL('../../src/data/reader-notes.json', import.meta.url))

type Store = Record<string, { id: string; created: number }[]>

// 路径、笔记都排好序，保证 diff 稳定
function serialize(store: Store): string {
  const out: Store = {}
  for (const path of Object.keys(store).sort()) {
    const list = store[path]
    if (list.length) out[path] = [...list].sort((a, b) => a.created - b.created || a.id.localeCompare(b.id))
  }
  return JSON.stringify(out, null, 2) + '\n'
}

function isStore(v: unknown): v is Store {
  if (!v || typeof v !== 'object' || Array.isArray(v)) return false
  return Object.values(v).every(
    (list) => Array.isArray(list) && list.every((n) => n && typeof n.id === 'string' && typeof n.exact === 'string'),
  )
}

export function readerNotesStore(): Plugin {
  return {
    name: 'reader-notes-store',
    apply: 'serve',
    config: () => ({
      // 写文件不触发 HMR，否则每次保存批注页面都会刷新
      server: { watch: { ignored: [NOTES_FILE.replace(/\\/g, '/')] } },
    }),
    configureServer(server) {
      server.middlewares.use(NOTES_ENDPOINT, async (req, res) => {
        try {
          if (req.method === 'GET') {
            const text = await readFile(NOTES_FILE, 'utf8').catch(() => '{}')
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
            const store = JSON.parse(body)
            if (!isStore(store)) {
              res.statusCode = 400
              res.end('invalid notes payload')
              return
            }
            await writeFile(NOTES_FILE, serialize(store))
            res.statusCode = 204
            res.end()
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
