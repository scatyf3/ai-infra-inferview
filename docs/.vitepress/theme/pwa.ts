import { withBase } from 'vitepress'

/** 生产环境注册离线缓存（docs/public/sw.js）；dev 时不注册，免得缓存住正在改的页面 */
export function registerServiceWorker() {
  if (import.meta.env.SSR || import.meta.env.DEV || !('serviceWorker' in navigator)) return
  window.addEventListener('load', () => {
    navigator.serviceWorker.register(withBase('/sw.js'), { scope: withBase('/') }).catch((e) => {
      console.error('[pwa] service worker 注册失败', e)
    })
  })
}
