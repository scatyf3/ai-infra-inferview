<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { PRIVATE_REPO } from '@lib/github'
import { githubToken, syncState } from '../sync'

/**
 * 「我的准备」：读私有仓库里的 markdown，在浏览器里渲染。内容不进公开仓库和构建产物。
 * 读取和渲染在 @lib/privateNotes（按需加载，markdown-it 只在这一页下载）。
 * 读到的内容缓存在这台设备的 localStorage 里，离线也能看；断开 GitHub 时一起清掉（见 ../sync.ts）。
 */

type Lib = typeof import('@lib/privateNotes')
let lib: Lib | null = null
const loadLib = async () => (lib ??= await import('@lib/privateNotes'))

const token = ref<string | null>(null)
const paths = ref<string[]>([])
const files = ref<Record<string, string>>({})
const current = ref('')
const html = ref('')
const loading = ref(false)
const error = ref('')

const repoUrl = `https://github.com/${PRIVATE_REPO.owner}/${PRIVATE_REPO.name}`
const editUrl = computed(() => (current.value ? `${repoUrl}/edit/${PRIVATE_REPO.branch}/${current.value}` : repoUrl))

function titleOf(p: string): string {
  return lib ? lib.noteTitle(p, files.value[p]) : p
}

function saveCache() {
  if (!lib) return
  try {
    localStorage.setItem(lib.PRIVATE_CACHE_KEY, JSON.stringify({ paths: paths.value, files: files.value }))
  } catch {
    // 存不下就算了，下次重新拉
  }
}

function pathFromHash(): string {
  const h = decodeURIComponent(location.hash.replace(/^#\/?/, ''))
  return h.split('#')[0]
}

async function show(p: string) {
  const l = await loadLib()
  current.value = p
  if (files.value[p] !== undefined) html.value = l.renderNote(p, files.value[p])
  if (!token.value) return
  try {
    const text = await l.readNote(token.value, p)
    files.value = { ...files.value, [p]: text }
    if (current.value === p) html.value = l.renderNote(p, text)
    saveCache()
  } catch (e) {
    error.value = (e as Error).message
  }
}

async function refresh() {
  const l = await loadLib()
  token.value = githubToken()
  error.value = ''
  if (!token.value) {
    paths.value = []
    files.value = {}
    html.value = ''
    return
  }
  const cache = l.parseCache(localStorage.getItem(l.PRIVATE_CACHE_KEY))
  if (!paths.value.length) {
    paths.value = cache.paths
    files.value = cache.files
  }
  loading.value = true
  try {
    paths.value = await l.listNotes(token.value)
    // 仓库里删掉的文件，缓存也删掉
    files.value = Object.fromEntries(Object.entries(files.value).filter(([p]) => paths.value.includes(p)))
    saveCache()
    // 列表里的标题要读文件第一行；文件不多，直接都读一遍
    await Promise.all(
      paths.value.map(async (p) => {
        const text = await l.readNote(token.value!, p)
        files.value = { ...files.value, [p]: text }
      }),
    )
    saveCache()
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    loading.value = false
  }
  // 刚把所有文件读了一遍（或者离线时用缓存），直接渲染，不再单独请求
  const want = pathFromHash()
  const target = paths.value.includes(want) ? want : paths.value[0]
  if (target) {
    current.value = target
    html.value = l.renderNote(target, files.value[target] ?? '')
  }
}

function clearCache() {
  if (lib) localStorage.removeItem(lib.PRIVATE_CACHE_KEY)
  files.value = {}
  html.value = ''
  refresh()
}

function onHash() {
  const p = pathFromHash()
  if (p && p !== current.value && paths.value.includes(p)) show(p)
}

onMounted(() => {
  refresh()
  window.addEventListener('hashchange', onHash)
})
onBeforeUnmount(() => window.removeEventListener('hashchange', onHash))

// 在导航栏的同步面板里连上 / 断开 GitHub 后，重新检查
watch(() => [syncState.mode, syncState.login], refresh)
</script>

<template>
  <div class="pn">
    <div v-if="!token" class="pn-box">
      <p><b>这页的内容在私有仓库里</b>，只在配置了 GitHub token 的设备上显示，站点本身不包含这些内容。</p>
      <ol>
        <li>点导航栏右上角的同步按钮，连上 GitHub。</li>
        <li>
          token 的 Repository access 里要包含 <code>{{ PRIVATE_REPO.name }}</code>（Contents: Read）。
          已经有 token 的话，去 GitHub 编辑它、加上这个仓库就行，不用重新粘贴。
        </li>
      </ol>
    </div>

    <template v-else>
      <div class="pn-bar">
        <span class="pn-muted">来自私有仓库 <a :href="repoUrl" target="_blank" rel="noreferrer">{{ PRIVATE_REPO.owner }}/{{ PRIVATE_REPO.name }}</a></span>
        <span class="pn-actions">
          <a :href="editUrl" target="_blank" rel="noreferrer">在 GitHub 上编辑</a>
          <button class="pn-btn" :disabled="loading" @click="refresh">{{ loading ? '读取中…' : '刷新' }}</button>
          <button class="pn-btn" title="删掉这台设备上缓存的私有内容" @click="clearCache">清除本机缓存</button>
        </span>
      </div>
      <p v-if="error" class="pn-error">{{ error }}</p>

      <div class="pn-main">
        <nav class="pn-list">
          <a
            v-for="p in paths"
            :key="p"
            :href="`#/${p}`"
            :class="{ active: p === current }"
          >
            <span>{{ titleOf(p) }}</span>
            <small v-if="p.includes('/')" class="pn-muted">{{ p.split('/').slice(0, -1).join('/') }}</small>
          </a>
          <p v-if="!paths.length && !loading" class="pn-muted">还没有文件。</p>
        </nav>
        <article class="pn-content vp-doc" v-html="html" />
      </div>
    </template>
  </div>
</template>

<style scoped>
.pn { margin-top: 8px; }
.pn-box { border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 12px 16px; }
.pn-box ol { margin: 8px 0 0; padding-left: 20px; }
.pn-bar { display: flex; flex-wrap: wrap; gap: 8px; justify-content: space-between; align-items: center; margin-bottom: 12px; font-size: 14px; }
.pn-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.pn-btn { border: 1px solid var(--vp-c-divider); border-radius: 6px; padding: 2px 10px; font-size: 13px; }
.pn-btn:disabled { opacity: 0.6; }
.pn-muted { color: var(--vp-c-text-2); }
.pn-error { color: var(--vp-c-danger-1); font-size: 14px; }
.pn-main { display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 24px; }
.pn-list { display: flex; flex-direction: column; gap: 2px; font-size: 14px; }
.pn-list a { display: flex; flex-direction: column; padding: 6px 10px; border-radius: 6px; color: var(--vp-c-text-1); text-decoration: none; }
.pn-list a:hover { background: var(--vp-c-bg-soft); }
.pn-list a.active { background: var(--vp-c-default-soft); font-weight: 600; }
.pn-list small { font-size: 12px; font-weight: 400; }
.pn-content { min-width: 0; }
@media (max-width: 720px) {
  .pn-main { grid-template-columns: 1fr; }
  .pn-list { flex-direction: row; flex-wrap: wrap; }
}
</style>
