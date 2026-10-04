import { computed, onMounted, reactive, shallowRef, type Ref } from 'vue'
import { SYNC_DOCS, type DocSpec } from '@lib/syncDocs'
import { InvalidRemoteError, StaleError, syncDocs, type Remote, type SyncTarget } from '@lib/sync'
import { AuthError, DATA_REPO, canWrite, checkToken, githubRemote, rawRemote } from '@lib/github'
import readerNotesBundled from '@data/reader-notes.json'
import progressBundled from '@data/flashcard-progress.json'
import cardNotesBundled from '@data/flashcard-notes.json'
import flagsBundled from '@data/flashcard-flags.json'

/**
 * 读者数据（批注、闪卡复习记录 / 批注 / 暂停）在设备间的同步，全站共用一份。
 *
 * 三种模式，数据都以 GitHub 上的 data 分支为准：
 * - github：这台设备配了 token。直接和 data 分支同步，手机出门也能用（见 src/lib/github.ts）。
 * - server：没配 token 的 dev server。dev server 用电脑上的 GitHub 登录代为读写 data 分支；
 *   电脑上没登录时才退回读写本机仓库里的 json（见 ../dataStore.ts，backend 字段说明是哪种）。
 * - local：没配 token 的线上站点。从公开地址只读拉取 data 分支，改动只存在这台设备的浏览器里。
 *
 * 不管哪种模式，每次改动都先写进 localStorage，界面立刻更新、离线也不丢；
 * 停手一会儿、切到后台、重新联网时再同步。同步就是「读远端 → 按条目合并 → 有变化就写回」，
 * 合并规则见各数据的 merge（都是较新的胜出，删除留记录），所以本地副本随时整份并进去都安全。
 */

export type SyncMode = 'github' | 'server' | 'local'
export type SyncStatus = 'idle' | 'syncing' | 'offline' | 'error'

export const syncState = reactive({
  mode: 'local' as SyncMode,
  status: 'idle' as SyncStatus,
  /** 本机有改动还没同步上去 */
  pending: false,
  /** 上次同步成功的时间戳 */
  lastSync: 0,
  error: '',
  /** github 模式下的 GitHub 用户名 */
  login: '',
  /** server 模式下 dev server 实际读写的地方：data 分支，还是本机仓库文件 */
  backend: '' as '' | 'github' | 'files',
})

/** 改动会不会同步到 GitHub 的 data 分支（给页面上的说明文字用） */
export const syncsToGitHub = computed(
  () => syncState.mode === 'github' || (syncState.mode === 'server' && syncState.backend !== 'files'),
)

export const DATA_BRANCH_URL = `https://github.com/${DATA_REPO.owner}/${DATA_REPO.name}/tree/${DATA_REPO.branch}/src/data`

const TOKEN_KEY = 'inferview:gh-token'
const LOGIN_KEY = 'inferview:gh-login'
const PENDING_KEY = 'inferview:sync-pending'
const SERVER_ENDPOINT = '/__data'

/** 停手多久后同步：GitHub 上每次同步是一个 commit，攒一攒 */
const DEBOUNCE = { github: 2000, server: 2000, local: 0 }
/** 出错后多久重试 */
const RETRY_MS = 30_000
/** 前台时多久拉一次别的设备的改动 */
const POLL_MS = 120_000

interface Entry<T> {
  spec: DocSpec<T>
  data: Ref<T>
}

const bundled: Record<string, unknown> = {
  'reader-notes': readerNotesBundled,
  'flashcard-progress': progressBundled,
  'flashcard-notes': cardNotesBundled,
  'flashcard-flags': flagsBundled,
}

// 模块级单例：组件卸载再挂载、换页面，用的都是同一份数据
const entries = new Map<string, Entry<any>>(
  SYNC_DOCS.map((spec) => {
    const b = bundled[spec.key]
    return [spec.key, { spec, data: shallowRef(spec.validate(b) ? b : spec.empty()) }]
  }),
)

// ---------- localStorage ----------

const ls = {
  get(k: string): string | null {
    try {
      return localStorage.getItem(k)
    } catch {
      return null
    }
  },
  set(k: string, v: string) {
    try {
      localStorage.setItem(k, v)
    } catch {}
  },
  del(k: string) {
    try {
      localStorage.removeItem(k)
    } catch {}
  },
}

function loadLocal<T>(e: Entry<T>): T | null {
  try {
    const v = JSON.parse(ls.get(e.spec.localKey) || 'null')
    return e.spec.validate(v) ? v : null
  } catch {
    return null
  }
}

const saveLocal = (e: Entry<any>) => ls.set(e.spec.localKey, JSON.stringify(e.data.value))

function setPending(v: boolean) {
  syncState.pending = v
  v ? ls.set(PENDING_KEY, '1') : ls.del(PENDING_KEY)
}

// ---------- 远端 ----------

function deviceName(): string {
  const ua = navigator.userAgent
  if (/iPhone/.test(ua)) return 'iPhone'
  if (/iPad/.test(ua)) return 'iPad'
  if (/Android/.test(ua)) return 'Android'
  return '桌面'
}

/** dev server 上没有数据接口（比如 vitepress preview）：退回只存本机 */
class NoServerError extends Error {}

/** dev server 代为读写（见 ../dataStore.ts）；接口和 GitHub 一个形状，版本对不上回 409 */
const serverRemote: Remote = {
  async read(paths) {
    const res = await fetch(SERVER_ENDPOINT, { cache: 'no-store' })
    if (res.status === 404) throw new NoServerError('dev server 上没有数据接口')
    if (!res.ok) throw new Error(`dev server 返回 ${res.status}：${await res.text()}`)
    const body = (await res.json()) as { backend: 'github' | 'files'; texts: Record<string, string | null>; version: string | null }
    syncState.backend = body.backend
    return { texts: Object.fromEntries(paths.map((p) => [p, body.texts[p] ?? null])), version: body.version }
  },
  async write(files, version, summary) {
    const res = await fetch(SERVER_ENDPOINT, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files, version, summary, device: deviceName() }),
    })
    if (res.ok) return
    const msg = ((await res.json().catch(() => null)) as { error?: string } | null)?.error ?? res.statusText
    if (res.status === 409) throw new StaleError(msg)
    if (res.status === 401) throw new AuthError(`dev server 上的 GitHub 登录不能用：${msg}`)
    throw new Error(`dev server 写入失败：${res.status} ${msg}`)
  },
}

function makeRemote(keepalive: boolean): Remote | null {
  if (syncState.mode === 'server') return serverRemote
  if (syncState.mode === 'local') return rawRemote()
  const token = ls.get(TOKEN_KEY)
  if (!token) return null
  const r = githubRemote({ token, device: deviceName() })
  r.keepalive = keepalive
  return r
}

// ---------- 同步调度 ----------

/** 每次本地改动加一；同步开始时记下，结束时没变才算「全部同步上去了」 */
let version = 0
let timer: ReturnType<typeof setTimeout> | undefined
let running: Promise<void> | null = null
let again = false
let lastAttempt = 0

const targets: SyncTarget<any>[] = [...entries.values()].map((e) => ({
  spec: e.spec,
  get: () => e.data.value,
  set(v) {
    // 内容没变就不换引用，免得页面上的高亮之类白白重画
    if (e.spec.serialize(v) === e.spec.serialize(e.data.value)) return
    e.data.value = v
    saveLocal(e)
  },
}))

function schedule(delay: number) {
  clearTimeout(timer)
  timer = setTimeout(() => void runSync(), delay)
}

async function runSync(keepalive = false): Promise<void> {
  if (running) {
    again = true
    return running
  }
  const remote = makeRemote(keepalive)
  if (!remote) return
  if (!navigator.onLine) {
    syncState.status = 'offline'
    return
  }
  lastAttempt = Date.now()
  const startVersion = version
  syncState.status = 'syncing'
  let ok = false
  running = (async () => {
    try {
      await syncDocs(remote, targets)
      ok = true
      syncState.status = 'idle'
      syncState.error = ''
      syncState.lastSync = Date.now()
      if (version === startVersion) setPending(false)
      else again = true
    } catch (e) {
      console.error('[sync]', e)
      syncState.status = 'error'
      syncState.error = (e as Error).message
      if (e instanceof NoServerError) {
        syncState.mode = 'local'
        syncState.status = 'idle'
        schedule(0)
      } else if (!(e instanceof AuthError || e instanceof InvalidRemoteError)) {
        schedule(RETRY_MS)
      }
    } finally {
      running = null
    }
  })()
  await running
  if (ok && again) {
    again = false
    schedule(DEBOUNCE[syncState.mode])
  }
}

/** 拉别的设备的改动；前台切换很频繁，短时间内不重复拉 */
function pull(minGapMs: number) {
  if (Date.now() - lastAttempt >= minGapMs) schedule(0)
}

// ---------- 启动 ----------

let started = false

function start() {
  if (started) return
  started = true

  // 本地缓存并进打包进来的初始数据
  for (const e of entries.values()) {
    const local = loadLocal(e)
    if (local) e.data.value = e.spec.merge(e.data.value, local)
  }

  const token = ls.get(TOKEN_KEY)
  syncState.mode = token ? 'github' : import.meta.env.DEV ? 'server' : 'local'
  syncState.login = token ? ls.get(LOGIN_KEY) ?? '' : ''
  syncState.pending = ls.get(PENDING_KEY) === '1'

  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') pull(5_000)
    // 切到后台（手机上锁屏、切 app）：有没同步的就赶紧推上去
    else if (syncState.pending) void runSync(true)
  })
  window.addEventListener('pagehide', () => {
    if (syncState.pending) void runSync(true)
  })
  window.addEventListener('focus', () => pull(15_000))
  window.addEventListener('online', () => schedule(0))
  window.addEventListener('offline', () => {
    if (syncState.mode !== 'local') syncState.status = 'offline'
  })
  setInterval(() => {
    if (document.visibilityState === 'visible') pull(POLL_MS)
  }, POLL_MS)

  schedule(0)
}

// ---------- 给组件用的 ----------

/**
 * 拿一份同步的数据。data 直接改（赋新对象），改完调 persist()。
 * 必须在组件 setup 里调用：第一次挂载时才读本地缓存、开始同步，避免和服务端渲染的结果对不上。
 */
export function useSyncedDoc<T>(spec: DocSpec<T>): { data: Ref<T>; persist: () => void } {
  const e = entries.get(spec.key) as Entry<T>
  onMounted(start)
  return {
    data: e.data,
    persist() {
      version++
      saveLocal(e)
      if (syncState.mode === 'local') return
      setPending(true)
      schedule(DEBOUNCE[syncState.mode])
    },
  }
}

export function syncNow() {
  schedule(0)
}

/** 验证并保存 token，切到 github 模式；不合格时抛错，原来的模式不变 */
export async function connectGitHub(token: string): Promise<void> {
  token = token.trim()
  const info = await checkToken(token)
  if (!canWrite(info.permission)) {
    throw new Error(`token 能登录（${info.login}），但没有 ${DATA_REPO.owner}/${DATA_REPO.name} 的写权限`)
  }
  ls.set(TOKEN_KEY, token)
  ls.set(LOGIN_KEY, info.login)
  syncState.mode = 'github'
  syncState.login = info.login
  syncState.error = ''
  syncState.status = 'idle'
  // 本机已有的数据（比如之前只存在浏览器里的）一起推上去
  setPending(true)
  schedule(0)
}

export function disconnectGitHub() {
  ls.del(TOKEN_KEY)
  ls.del(LOGIN_KEY)
  syncState.login = ''
  syncState.error = ''
  syncState.status = 'idle'
  syncState.mode = import.meta.env.DEV ? 'server' : 'local'
  schedule(0)
}
