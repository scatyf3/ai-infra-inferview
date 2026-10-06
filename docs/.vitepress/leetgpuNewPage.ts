import { writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import type { Plugin } from 'vite'
// config 在 vite 别名之外打包，只能走相对路径
import challenges from '../../src/data/leetgpu-challenges.json'
import { slugOf, solutionPageSkeleton } from '../../src/lib/leetgpuPage'

/**
 * dev 时在 LeetGPU 看板上点「新建」，往 docs/leetgpu/ 写一个题解骨架页。
 * 只在 `vitepress dev` 生效；生产构建里没有这个接口，按钮也不显示（见 LeetGPUBoard.vue）。
 */

export const NEW_PAGE_ENDPOINT = '/__leetgpu-new'
const DIR = fileURLToPath(new URL('../leetgpu/', import.meta.url))

export function leetgpuNewPage(): Plugin {
  return {
    name: 'leetgpu-new-page',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(NEW_PAGE_ENDPOINT, async (req, res) => {
        const reply = (status: number, body: unknown) => {
          res.statusCode = status
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify(body))
        }
        try {
          if (req.method !== 'POST') return reply(405, { error: 'POST only' })
          let body = ''
          for await (const chunk of req) {
            body += chunk
            if (body.length > 1000) return reply(413, { error: 'payload too large' })
          }
          const id = Number(JSON.parse(body).id)
          const c = challenges.find((x) => x.id === id)
          if (!c) return reply(404, { error: `unknown leetgpu id ${id}` })

          const slug = slugOf(c.title)
          const url = `/leetgpu/${slug}`
          try {
            // wx：文件已存在就失败，绝不覆盖已经写过的题解
            await writeFile(`${DIR}${slug}.md`, solutionPageSkeleton(c), { flag: 'wx' })
          } catch (e) {
            if ((e as NodeJS.ErrnoException).code === 'EEXIST') return reply(409, { url, error: 'page already exists' })
            throw e
          }
          reply(201, { url })
        } catch (e) {
          reply(500, { error: String(e) })
        }
      })
    },
  }
}
