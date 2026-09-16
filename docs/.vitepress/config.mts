import { defineConfig } from 'vitepress'
import { fileURLToPath } from 'node:url'
import { buildSidebar } from './sidebar'

export default defineConfig({
  lang: 'zh-CN',
  title: 'AI Infra Inferview',
  description: '北美 AI Infra 面试知识库：推理系统、并行通信、GPU 算子、框架内功、Post-train 与手撕',
  base: '/ai-infra-inferview/',
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: false,
  markdown: {
    math: true,
    lineNumbers: false,
  },
  themeConfig: {
    nav: [
      { text: '知识图谱', link: '/' },
      { text: '推理', link: '/inference/' },
      { text: '并行', link: '/parallel/' },
      { text: '框架', link: '/framework/' },
      { text: '手撕', link: '/handson/' },
    ],
    sidebar: buildSidebar(),
    search: { provider: 'local', options: { detailedView: true } },
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新' },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '主题',
    socialLinks: [{ icon: 'github', link: 'https://github.com/scatyf3/ai-infra-inferview' }],
  },
  vite: {
    resolve: {
      alias: {
        '@lib': fileURLToPath(new URL('../../src/lib', import.meta.url)),
        '@data': fileURLToPath(new URL('../../src/data', import.meta.url)),
      },
    },
  },
})
