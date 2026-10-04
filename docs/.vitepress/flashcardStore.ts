import { readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import type { Plugin } from 'vite'
import { isNotes, isProgress, serializeNotes, serializeProgress } from '../../src/lib/flashcards'

/**
 * dev 时把闪卡的复习记录和批注读写到仓库里的两个 json，靠 git 在设备间同步。
 * 生产构建里这两个文件被打包进站点（只读），见 Flashcards.vue。
 */

export const FLASHCARD_ENDPOINT = '/__flashcards'
export const FLASHCARD_NOTES_ENDPOINT = '/__flashcard-notes'
const dataFile = (name: string) => fileURLToPath(new URL(`../../src/data/${name}`, import.meta.url))

interface JsonFileStore {
  name: string
  endpoint: string
  file: string
  validate: (v: unknown) => boolean
  serialize: (v: any) => string
}

function jsonFileStore({ name, endpoint, file, validate, serialize }: JsonFileStore): Plugin {
  return {
    name,
    apply: 'serve',
    config: () => ({
      // 写文件不触发 HMR，否则每次保存页面都会刷新
      server: { watch: { ignored: [file.replace(/\\/g, '/')] } },
    }),
    configureServer(server) {
      server.middlewares.use(endpoint, async (req, res) => {
        try {
          if (req.method === 'GET') {
            const text = await readFile(file, 'utf8').catch(() => '{}')
            res.setHeader('Content-Type', 'application/json')
            res.end(text)
            return
          }
          if (req.method === 'PUT') {
            let body = ''
            for await (const chunk of req) {
              body += chunk
              if (body.length > 2_000_000) throw new Error('payload too large')
            }
            const data = JSON.parse(body)
            if (!validate(data)) {
              res.statusCode = 400
              res.end(`invalid ${name} payload`)
              return
            }
            await writeFile(file, serialize(data))
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

export function flashcardStore(): Plugin[] {
  return [
    jsonFileStore({
      name: 'flashcard-progress-store',
      endpoint: FLASHCARD_ENDPOINT,
      file: dataFile('flashcard-progress.json'),
      validate: isProgress,
      serialize: serializeProgress,
    }),
    jsonFileStore({
      name: 'flashcard-notes-store',
      endpoint: FLASHCARD_NOTES_ENDPOINT,
      file: dataFile('flashcard-notes.json'),
      validate: isNotes,
      serialize: serializeNotes,
    }),
  ]
}
