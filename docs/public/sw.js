/*
 * 离线缓存（PWA）。只管站点自己的静态文件，不碰 GitHub API：读者数据的离线靠 localStorage（见 theme/sync.ts）。
 * - assets/ 下的文件名带内容哈希，内容不会变：先用缓存，没有再取网络。
 * - 页面（HTML）：先取网络拿最新的，断网时用缓存，所以访问过的页面离线也能打开。
 * - 其他（图标、manifest 等）：先给缓存，后台再更新。
 * 改了缓存策略就把 VERSION 加一，旧缓存会在新 service worker 激活时删掉。
 */
const VERSION = 1
const CACHE = `inferview-v${VERSION}`
const SCOPE = new URL(self.registration.scope).pathname
// 装好就先缓存首页和闪卡页，第一次离线打开也有东西看
const PRECACHE = [SCOPE, `${SCOPE}flashcards`, `${SCOPE}handson/flashcards`]

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then((c) => Promise.all(PRECACHE.map((u) => c.add(u).catch(() => {}))))
      .then(() => self.skipWaiting()),
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('inferview-') && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  )
})

async function cacheFirst(req) {
  const hit = await caches.match(req)
  if (hit) return hit
  const res = await fetch(req)
  if (res.ok) (await caches.open(CACHE)).put(req, res.clone())
  return res
}

async function networkFirst(req) {
  try {
    const res = await fetch(req)
    if (res.ok) (await caches.open(CACHE)).put(req, res.clone())
    return res
  } catch (e) {
    const hit = (await caches.match(req)) || (await caches.match(req.url.replace(/\.html$/, ''))) || (await caches.match(SCOPE))
    if (hit) return hit
    throw e
  }
}

async function staleWhileRevalidate(req) {
  const cache = await caches.open(CACHE)
  const hit = await cache.match(req)
  const fresh = fetch(req)
    .then((res) => {
      if (res.ok) cache.put(req, res.clone())
      return res
    })
    .catch(() => hit)
  return hit || fresh
}

self.addEventListener('fetch', (event) => {
  const req = event.request
  if (req.method !== 'GET') return
  const url = new URL(req.url)
  if (url.origin !== self.location.origin || !url.pathname.startsWith(SCOPE)) return
  if (url.pathname.startsWith(`${SCOPE}assets/`)) event.respondWith(cacheFirst(req))
  else if (req.mode === 'navigate' || url.pathname.endsWith('.html')) event.respondWith(networkFirst(req))
  else event.respondWith(staleWhileRevalidate(req))
})
