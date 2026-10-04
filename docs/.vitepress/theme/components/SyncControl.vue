<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { DATA_REPO } from '@lib/github'
import { DATA_BRANCH_URL, connectGitHub, disconnectGitHub, syncNow, syncState } from '../sync'

/**
 * 导航栏上的同步状态 + 设置面板：连 / 断 GitHub、立即同步、装到手机桌面的提示。
 * 同步逻辑在 ../sync.ts。
 */

const open = ref(false)
const token = ref('')
const busy = ref(false)
const formError = ref('')
const now = ref(Date.now())
let ticker: number | undefined

const label = computed(() => {
  if (syncState.mode === 'local') return '本机'
  if (syncState.status === 'syncing') return '同步中'
  if (syncState.status === 'offline') return syncState.pending ? '离线·待同步' : '离线'
  if (syncState.status === 'error') return '同步失败'
  if (syncState.pending) return '待同步'
  return syncState.mode === 'file' ? '仓库文件' : '已同步'
})
const tone = computed(() => {
  if (syncState.mode === 'local') return 'muted'
  if (syncState.status === 'error') return 'danger'
  if (syncState.status === 'syncing' || syncState.pending || syncState.status === 'offline') return 'warn'
  return 'ok'
})

function ago(t: number): string {
  if (!t) return '还没有'
  const s = Math.round((now.value - t) / 1000)
  if (s < 10) return '刚刚'
  if (s < 60) return `${s} 秒前`
  if (s < 3600) return `${Math.round(s / 60)} 分钟前`
  return new Date(t).toLocaleString()
}

// GitHub 支持用 URL 参数预填 fine-grained token 的表单；不认的参数会被忽略，下面也写了手动步骤
const tokenUrl =
  'https://github.com/settings/personal-access-tokens/new?' +
  new URLSearchParams({
    name: 'inferview 同步',
    description: '读者批注和闪卡记录的同步',
    target_name: DATA_REPO.owner,
    contents: 'write',
  })

async function save() {
  if (!token.value.trim()) return
  busy.value = true
  formError.value = ''
  try {
    await connectGitHub(token.value)
    token.value = ''
  } catch (e) {
    formError.value = (e as Error).message
  } finally {
    busy.value = false
  }
}

function disconnect() {
  if (confirm('断开后这台设备不再同步（本机已有的数据保留）。要换 token 或者这台设备不再用了才需要断开。确定？')) disconnectGitHub()
}

// ---------- 装到桌面 ----------
const standalone = ref(false)
const isIOS = ref(false)
let installEvent: (Event & { prompt: () => Promise<void> }) | null = null
const canInstall = ref(false)

function onBeforeInstall(e: Event) {
  e.preventDefault()
  installEvent = e as typeof installEvent
  canInstall.value = true
}
async function install() {
  await installEvent?.prompt()
  installEvent = null
  canInstall.value = false
}

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape') open.value = false
}

onMounted(() => {
  standalone.value = matchMedia('(display-mode: standalone)').matches || (navigator as { standalone?: boolean }).standalone === true
  isIOS.value = /iPhone|iPad/.test(navigator.userAgent)
  window.addEventListener('beforeinstallprompt', onBeforeInstall)
  window.addEventListener('keydown', onKey)
  ticker = window.setInterval(() => (now.value = Date.now()), 15_000)
})
onBeforeUnmount(() => {
  window.removeEventListener('beforeinstallprompt', onBeforeInstall)
  window.removeEventListener('keydown', onKey)
  clearInterval(ticker)
})
</script>

<template>
  <div class="sc">
    <button class="sc-btn" :class="tone" :title="`数据同步：${label}`" @click="open = !open; now = Date.now()">
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
        <path
          d="M7 18a4.5 4.5 0 0 1-.6-8.96A6 6 0 0 1 18 8.5a4 4 0 0 1-.5 9.5H7z"
          fill="none"
          stroke="currentColor"
          stroke-width="1.7"
          stroke-linejoin="round"
        />
      </svg>
      <i class="sc-dot" :class="[tone, { spin: syncState.status === 'syncing' }]" />
      <span class="sc-label">{{ label }}</span>
    </button>

    <Teleport to="body">
      <div v-if="open" class="sc-mask" @click.self="open = false">
        <section class="sc-panel" role="dialog" aria-label="数据同步">
          <header>
            <b>数据同步</b>
            <button class="sc-x" aria-label="关闭" @click="open = false">×</button>
          </header>

          <p class="sc-muted">
            页面上的批注、闪卡复习记录 / 批注 / 暂停，每次改动先存在这台设备上，再自动同步。
          </p>

          <!-- 已连 GitHub -->
          <template v-if="syncState.mode === 'github'">
            <dl class="sc-kv">
              <dt>账号</dt>
              <dd>{{ syncState.login || '—' }}</dd>
              <dt>状态</dt>
              <dd :class="tone">{{ label }}</dd>
              <dt>上次同步</dt>
              <dd>{{ ago(syncState.lastSync) }}</dd>
              <dt>存在</dt>
              <dd><a :href="DATA_BRANCH_URL" target="_blank" rel="noreferrer">{{ DATA_REPO.name }} 的 {{ DATA_REPO.branch }} 分支</a></dd>
            </dl>
            <p v-if="syncState.error" class="sc-err">{{ syncState.error }}</p>
            <div class="sc-row">
              <button class="sc-primary" :disabled="syncState.status === 'syncing'" @click="syncNow">立即同步</button>
              <span class="sc-spacer" />
              <button class="sc-plain" @click="disconnect">断开</button>
            </div>
          </template>

          <!-- 没连 -->
          <template v-else>
            <p>
              现在：<b>{{ syncState.mode === 'file' ? '写进本机仓库的文件，靠 git 同步' : '只存在这台设备的浏览器里' }}</b>。
              连上 GitHub 后，手机和电脑在任何网络下都同步同一份数据。
            </p>
            <ol class="sc-steps">
              <li>
                <a :href="tokenUrl" target="_blank" rel="noreferrer">新建一个 fine-grained token</a>：
                Repository access 选 <i>Only select repositories</i> → <code>{{ DATA_REPO.name }}</code>；
                Permissions 里 <i>Contents</i> 选 <i>Read and write</i>。
              </li>
              <li>把生成的 token 粘贴到下面。它只存在这台设备上；手机丢了就去 GitHub 吊销它。</li>
            </ol>
            <form class="sc-row" @submit.prevent="save">
              <input v-model="token" type="password" autocomplete="off" placeholder="github_pat_…" class="sc-input" />
              <button class="sc-primary" type="submit" :disabled="busy || !token.trim()">{{ busy ? '验证中…' : '连接' }}</button>
            </form>
            <p v-if="formError" class="sc-err">{{ formError }}</p>
            <p v-else-if="syncState.error" class="sc-err">{{ syncState.error }}</p>
            <p v-if="syncState.mode === 'file'" class="sc-muted">
              提示：电脑上也连上之后，数据以 GitHub 上的 {{ DATA_REPO.branch }} 分支为准，本机仓库里的这几个 json 不再更新。
            </p>
          </template>

          <!-- 装到桌面 -->
          <div v-if="!standalone" class="sc-install">
            <b>装到手机桌面</b>
            <span v-if="canInstall"><button class="sc-plain" @click="install">安装</button> 全屏打开，离线也能复习。</span>
            <span v-else-if="isIOS">Safari 里点「分享」→「添加到主屏幕」，之后从桌面图标打开就是全屏、离线可用。</span>
            <span v-else>手机浏览器菜单里选「安装应用」或「添加到主屏幕」，之后全屏打开、离线可用。</span>
          </div>
        </section>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.sc { display: flex; align-items: center; margin-left: 8px; }
.sc-btn {
  position: relative;
  display: flex;
  align-items: center;
  gap: 6px;
  height: 32px;
  padding: 0 8px;
  border-radius: 8px;
  color: var(--vp-c-text-2);
  font-size: 13px;
  white-space: nowrap;
}
.sc-btn:hover { color: var(--vp-c-text-1); background: var(--vp-c-default-soft); }
.sc-dot {
  position: absolute;
  left: 19px;
  top: 7px;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  border: 1.5px solid var(--vp-c-bg);
  box-sizing: content-box;
}
.sc-dot.ok { background: var(--vp-c-green-1); }
.sc-dot.warn { background: var(--vp-c-yellow-1); }
.sc-dot.danger { background: var(--vp-c-red-1); }
.sc-dot.muted { background: var(--vp-c-text-3); }
.sc-dot.spin { animation: sc-pulse 1s ease-in-out infinite; }
@keyframes sc-pulse { 50% { opacity: 0.3; } }
@media (max-width: 640px) {
  .sc { margin-left: 0; }
  .sc-label { display: none; }
}

.sc-mask {
  position: fixed;
  inset: 0;
  z-index: 100;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 72px 16px 16px;
  background: rgba(0, 0, 0, 0.35);
}
.sc-panel {
  width: min(440px, 100%);
  max-height: calc(100vh - 96px);
  overflow-y: auto;
  padding: 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg-elv);
  box-shadow: var(--vp-shadow-4);
  font-size: 14px;
  line-height: 1.6;
}
.sc-panel header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; font-size: 16px; }
.sc-x { font-size: 22px; line-height: 1; padding: 0 6px; color: var(--vp-c-text-2); }
.sc-panel p { margin: 8px 0; }
.sc-muted { color: var(--vp-c-text-2); font-size: 13px; }
.sc-err { color: var(--vp-c-red-1); font-size: 13px; word-break: break-word; }
.sc-kv { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; margin: 12px 0; }
.sc-kv dt { color: var(--vp-c-text-2); }
.sc-kv dd { margin: 0; }
.sc-kv .ok { color: var(--vp-c-green-1); }
.sc-kv .warn { color: var(--vp-c-yellow-1); }
.sc-kv .danger { color: var(--vp-c-red-1); }
.sc-steps { margin: 8px 0; padding-left: 20px; }
.sc-steps li { margin: 4px 0; }
.sc-steps code { font-size: 12px; }
.sc-row { display: flex; align-items: center; gap: 8px; margin-top: 12px; }
.sc-spacer { flex: 1; }
.sc-input {
  flex: 1;
  min-width: 0;
  height: 38px;
  padding: 0 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  /* 16px 以下 iOS 聚焦时会自动放大页面 */
  font-size: 16px;
}
.sc-primary,
.sc-plain {
  height: 38px;
  padding: 0 14px;
  border-radius: 8px;
  font-weight: 500;
}
.sc-primary { background: var(--vp-c-brand-3); color: var(--vp-c-white); }
.sc-primary:hover:not(:disabled) { background: var(--vp-c-brand-2); }
.sc-primary:disabled { opacity: 0.5; cursor: not-allowed; }
.sc-plain { border: 1px solid var(--vp-c-divider); color: var(--vp-c-text-1); }
.sc-install {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-2);
  font-size: 13px;
}
.sc-install b { color: var(--vp-c-text-1); }
.sc-install .sc-plain { height: 30px; padding: 0 10px; margin-right: 6px; }
</style>
