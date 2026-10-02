<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { onContentUpdated, useRoute } from 'vitepress'
import { locateQuote, makeQuote, type TextQuote } from '@lib/annotate'
import bundled from '@data/reader-notes.json'

/**
 * 读者侧划词高亮 + 批注。选中正文 → 浮动工具条；点高亮 → 编辑批注；右下角按钮看本页笔记、导入导出。
 * 用文本锚点（原文 + 前后文）定位，文档小改后仍能找回。
 * 存储：dev 时读写仓库里的 src/data/reader-notes.json（见 readerNotesStore.ts），随 git 跨设备同步；
 * 线上该文件打包进站点只读，本机新增/修改存 localStorage，按 id 覆盖文件里的。
 * 交互 widget（.widget）里的 DOM 由 Vue 管理，不往里插 mark。
 */

interface Note extends TextQuote {
  id: string
  note: string
  created: number
}
type Store = Record<string, Note[]>

const KEY = 'inferview:reader-notes:v1'
const ENDPOINT = '/__reader-notes'
const ROOT = '.vp-doc'
const SKIP = '.widget, .rn-ui, .anno-note, .header-anchor, script, style'

const route = useRoute()
const ready = ref(false)
const hasDoc = ref(false)
const store = ref<Store>({})
// 是否以仓库文件为数据源（dev 且 dev server 可达）
const fileMode = ref(import.meta.env.DEV)
const orphans = ref<Set<string>>(new Set())
const order = ref<string[]>([])

const notes = computed(() => store.value[route.path] ?? [])
const sortedNotes = computed(() => {
  const rank = new Map(order.value.map((id, i) => [id, i]))
  return [...notes.value].sort((a, b) => (rank.get(a.id) ?? 1e9) - (rank.get(b.id) ?? 1e9))
})

// ---------- 持久化 ----------
function loadLocal(): Store {
  try {
    return JSON.parse(localStorage.getItem(KEY) || '{}')
  } catch {
    return {}
  }
}

function saveLocal(s: Store) {
  try {
    localStorage.setItem(KEY, JSON.stringify(s))
  } catch {}
}

/** 按 id 合并，over 覆盖 base；顺带丢掉格式不对的条目 */
function merge(base: Store, over: Store): Store {
  const out: Store = { ...base }
  for (const [path, list] of Object.entries(over)) {
    if (!Array.isArray(list)) continue
    const byId = new Map((out[path] ?? []).map((n) => [n.id, n]))
    for (const n of list) if (n?.id && typeof n.exact === 'string') byId.set(n.id, n)
    out[path] = [...byId.values()]
  }
  return out
}

async function putFile(s: Store) {
  const res = await fetch(ENDPOINT, { method: 'PUT', body: JSON.stringify(s) })
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
}

async function load(): Promise<Store> {
  if (!fileMode.value) return merge(bundled as Store, loadLocal())
  try {
    const res = await fetch(ENDPOINT)
    if (!res.ok) throw new Error(`${res.status}`)
    const file = (await res.json()) as Store
    // 浏览器里的旧笔记（或之前写文件失败暂存的）并入文件，成功后清掉本地副本
    const local = loadLocal()
    if (!Object.keys(local).length) return file
    const merged = merge(file, local)
    await putFile(merged)
    localStorage.removeItem(KEY)
    return merged
  } catch (e) {
    console.error('[reader-notes] 读不到笔记文件，退回 localStorage', e)
    fileMode.value = false
    return merge(bundled as Store, loadLocal())
  }
}

// 写文件串行，保证落盘顺序和操作顺序一致
let saving: Promise<void> = Promise.resolve()

function persist() {
  const snapshot = store.value
  if (!fileMode.value) return saveLocal(snapshot)
  saving = saving
    .then(() => putFile(snapshot))
    .catch((e) => {
      console.error('[reader-notes] 写笔记文件失败，暂存到 localStorage', e)
      saveLocal(snapshot)
    })
}

async function reload() {
  await saving
  store.value = await load()
  applyAll()
}

function setNotes(list: Note[]) {
  const next = { ...store.value }
  if (list.length) next[route.path] = list
  else delete next[route.path]
  store.value = next
  persist()
}

// ---------- DOM 文本索引 ----------
interface TextIndex {
  nodes: Text[]
  starts: number[]
  full: string
}

function buildIndex(root: Element): TextIndex {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) => (n.parentElement?.closest(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  })
  const nodes: Text[] = []
  const starts: number[] = []
  let full = ''
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    nodes.push(n as Text)
    starts.push(full.length)
    full += (n as Text).data
  }
  return { nodes, starts, full }
}

function rangeOffsets(range: Range, idx: TextIndex): { start: number; end: number } | null {
  let start = -1
  let end = -1
  idx.nodes.forEach((n, i) => {
    if (!range.intersectsNode(n)) return
    const s = n === range.startContainer ? range.startOffset : 0
    const e = n === range.endContainer ? range.endOffset : n.length
    if (e <= s) return
    if (start < 0) start = idx.starts[i] + s
    end = idx.starts[i] + e
  })
  if (start < 0) return null
  // 去掉首尾空白，跨段选择时不把换行算进去
  while (start < end && /\s/.test(idx.full[start])) start++
  while (end > start && /\s/.test(idx.full[end - 1])) end--
  return end > start ? { start, end } : null
}

function wrap(idx: TextIndex, start: number, end: number, n: Note) {
  const segs: [Text, number, number][] = []
  idx.nodes.forEach((node, i) => {
    const s = Math.max(start, idx.starts[i]) - idx.starts[i]
    const e = Math.min(end, idx.starts[i] + node.length) - idx.starts[i]
    if (e > s && node.data.slice(s, e).trim()) segs.push([node, s, e])
  })
  for (const [node, s, e] of segs) {
    let t = node
    if (e < t.length) t.splitText(e)
    if (s > 0) t = t.splitText(s)
    const m = document.createElement('mark')
    m.className = n.note ? 'rn-hl has-note' : 'rn-hl'
    m.dataset.rnId = n.id
    if (n.note) m.title = n.note
    t.parentNode!.insertBefore(m, t)
    m.appendChild(t)
  }
}

function unwrapAll(root: Element) {
  const parents = new Set<Node>()
  root.querySelectorAll('mark.rn-hl').forEach((m) => {
    const p = m.parentNode!
    while (m.firstChild) p.insertBefore(m.firstChild, m)
    p.removeChild(m)
    parents.add(p)
  })
  parents.forEach((p) => p.normalize())
}

function applyAll() {
  const root = document.querySelector(ROOT)
  hasDoc.value = !!root
  if (!root) return
  unwrapAll(root)
  const lost = new Set<string>()
  const located: [number, string][] = []
  for (const n of notes.value) {
    // 每次 wrap 都会拆分文本节点，所以重新建索引（总文本不变，偏移仍有效）
    const idx = buildIndex(root)
    const loc = locateQuote(idx.full, n)
    if (!loc) {
      lost.add(n.id)
      continue
    }
    wrap(idx, loc.start, loc.end, n)
    located.push([loc.start, n.id])
  }
  orphans.value = lost
  order.value = located.sort((a, b) => a[0] - b[0]).map(([, id]) => id)
}

// ---------- 选择工具条 ----------
const toolbar = ref<{ top: number; left: number } | null>(null)
let pending: { start: number; end: number } | null = null

function onSelectionEnd(e: Event) {
  if ((e.target as Element | null)?.closest?.('.rn-ui')) return
  setTimeout(() => {
    const sel = window.getSelection()
    const root = document.querySelector(ROOT)
    toolbar.value = null
    pending = null
    if (!sel || sel.isCollapsed || !sel.rangeCount || !root) return
    const range = sel.getRangeAt(0)
    const anc = range.commonAncestorContainer
    const ancEl = anc.nodeType === 1 ? (anc as Element) : anc.parentElement
    if (!root.contains(anc) || ancEl?.closest(SKIP)) return
    const off = rangeOffsets(range, buildIndex(root))
    if (!off) return
    pending = off
    const rect = range.getBoundingClientRect()
    toolbar.value = { top: rect.top + window.scrollY - 44, left: rect.left + window.scrollX + rect.width / 2 }
  })
}

function addHighlight(withNote: boolean) {
  const root = document.querySelector(ROOT)
  if (!pending || !root) return
  const idx = buildIndex(root)
  const n: Note = {
    id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
    ...makeQuote(idx.full, pending.start, pending.end),
    note: '',
    created: Date.now(),
  }
  setNotes([...notes.value, n])
  window.getSelection()?.removeAllRanges()
  toolbar.value = null
  pending = null
  applyAll()
  if (withNote) nextTick(() => openEditor(n.id))
}

// ---------- 批注编辑 ----------
const editing = ref<{ id: string; top: number; left: number } | null>(null)
const draft = ref('')
const editorInput = ref<HTMLTextAreaElement | null>(null)

function openEditor(id: string) {
  const m = document.querySelector(`mark.rn-hl[data-rn-id="${id}"]`)
  const n = notes.value.find((x) => x.id === id)
  if (!m || !n) return
  const rect = m.getBoundingClientRect()
  const left = Math.min(rect.left + window.scrollX, window.scrollX + document.documentElement.clientWidth - 316)
  editing.value = { id, top: rect.bottom + window.scrollY + 6, left: Math.max(window.scrollX + 8, left) }
  draft.value = n.note
  nextTick(() => editorInput.value?.focus())
}

function saveNote() {
  if (!editing.value) return
  const id = editing.value.id
  setNotes(notes.value.map((n) => (n.id === id ? { ...n, note: draft.value.trim() } : n)))
  editing.value = null
  applyAll()
}

function removeNote(id: string) {
  setNotes(notes.value.filter((n) => n.id !== id))
  if (editing.value?.id === id) editing.value = null
  applyAll()
}

function onDocClick(e: MouseEvent) {
  const t = e.target as Element
  if (t.closest('.rn-ui')) return
  const m = t.closest('mark.rn-hl') as HTMLElement | null
  if (m && window.getSelection()?.isCollapsed) openEditor(m.dataset.rnId!)
  else editing.value = null
}

// ---------- 面板 ----------
const panelOpen = ref(false)
const importInput = ref<HTMLInputElement | null>(null)

function jumpTo(id: string) {
  const m = document.querySelector(`mark.rn-hl[data-rn-id="${id}"]`)
  if (!m) return
  m.scrollIntoView({ behavior: 'smooth', block: 'center' })
  document.querySelectorAll(`mark.rn-hl[data-rn-id="${id}"]`).forEach((el) => {
    el.classList.remove('rn-flash')
    void (el as HTMLElement).offsetWidth
    el.classList.add('rn-flash')
  })
}

function exportAll() {
  const blob = new Blob([JSON.stringify(store.value, null, 2)], { type: 'application/json' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `inferview-notes-${new Date().toISOString().slice(0, 10)}.json`
  a.click()
  URL.revokeObjectURL(a.href)
}

async function importFile(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (!file) return
  try {
    store.value = merge(store.value, JSON.parse(await file.text()) as Store)
    persist()
    applyAll()
  } catch {
    alert('导入失败：不是有效的笔记 JSON')
  }
  ;(e.target as HTMLInputElement).value = ''
}

function clearPage() {
  if (confirm(`删除本页全部 ${notes.value.length} 条高亮？`)) {
    setNotes([])
    applyAll()
  }
}

// ---------- 生命周期 ----------
function onStorage(e: StorageEvent) {
  if (e.key !== KEY || fileMode.value) return
  reload()
}

// 文件模式下回到窗口就重读：其他标签页的修改、git pull 进来的笔记都会生效
function onFocus() {
  if (fileMode.value && !editing.value) reload()
}

onContentUpdated(() => {
  editing.value = null
  toolbar.value = null
  nextTick(applyAll)
})

onMounted(async () => {
  document.addEventListener('mouseup', onSelectionEnd)
  document.addEventListener('keyup', onSelectionEnd)
  document.addEventListener('click', onDocClick)
  window.addEventListener('storage', onStorage)
  window.addEventListener('focus', onFocus)
  store.value = await load()
  ready.value = true
  applyAll()
})

onBeforeUnmount(() => {
  document.removeEventListener('mouseup', onSelectionEnd)
  document.removeEventListener('keyup', onSelectionEnd)
  document.removeEventListener('click', onDocClick)
  window.removeEventListener('storage', onStorage)
  window.removeEventListener('focus', onFocus)
})
</script>

<template>
  <Teleport v-if="ready" to="body">
    <div
      v-if="toolbar"
      class="rn-ui rn-toolbar"
      :style="{ top: toolbar.top + 'px', left: toolbar.left + 'px' }"
      @mousedown.prevent
    >
      <button @click="addHighlight(false)">高亮</button>
      <button @click="addHighlight(true)">批注</button>
    </div>

    <div v-if="editing" class="rn-ui rn-editor" :style="{ top: editing.top + 'px', left: editing.left + 'px' }">
      <textarea
        ref="editorInput"
        v-model="draft"
        rows="3"
        placeholder="写点批注…（⌘/Ctrl+Enter 保存）"
        @keydown.meta.enter="saveNote"
        @keydown.ctrl.enter="saveNote"
        @keydown.esc="editing = null"
      />
      <div class="rn-actions">
        <button class="danger" @click="removeNote(editing.id)">删除高亮</button>
        <span class="spacer" />
        <button @click="editing = null">取消</button>
        <button class="primary" @click="saveNote">保存</button>
      </div>
    </div>

    <template v-if="hasDoc">
      <button
        class="rn-ui rn-fab"
        :class="{ active: panelOpen }"
        :title="`本页笔记（${notes.length}）`"
        @click="panelOpen = !panelOpen"
      >
        ✎<span v-if="notes.length" class="rn-count">{{ notes.length }}</span>
      </button>

      <aside v-if="panelOpen" class="rn-ui rn-panel">
        <header>
          <b>本页笔记</b>
          <span class="rn-muted">
            选中正文即可高亮 / 批注，{{
              fileMode ? '自动写入 src/data/reader-notes.json，提交到 git 即可同步' : '新增内容仅保存在本浏览器'
            }}
          </span>
        </header>
        <p v-if="!notes.length" class="rn-muted rn-empty">还没有高亮。</p>
        <ol v-else>
          <li v-for="n in sortedNotes" :key="n.id" :class="{ orphan: orphans.has(n.id) }">
            <button class="rn-quote" :disabled="orphans.has(n.id)" @click="jumpTo(n.id)">
              {{ n.exact.length > 80 ? n.exact.slice(0, 80) + '…' : n.exact }}
            </button>
            <div v-if="n.note" class="rn-note">{{ n.note }}</div>
            <div v-if="orphans.has(n.id)" class="rn-muted">原文已改动，找不到这段了</div>
            <button class="rn-del" title="删除" @click="removeNote(n.id)">×</button>
          </li>
        </ol>
        <footer>
          <button @click="exportAll">导出全部</button>
          <button @click="importInput?.click()">导入</button>
          <button v-if="notes.length" class="danger" @click="clearPage">清空本页</button>
          <input ref="importInput" type="file" accept="application/json,.json" hidden @change="importFile" />
        </footer>
      </aside>
    </template>
  </Teleport>
</template>
