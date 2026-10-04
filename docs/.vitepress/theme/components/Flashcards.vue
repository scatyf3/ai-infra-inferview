<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch, type Directive } from 'vue'
import { withBase } from 'vitepress'
import {
  DECKS,
  GRADES,
  STATE_LABEL,
  State,
  bucketOf,
  countBuckets,
  fmtInterval,
  inlineMd,
  isSuspended,
  nextCard,
  nextDueAt,
  noteOf,
  preview,
  review,
  setNote,
  setSuspended,
  type Bucket,
  type Deck,
  type Grade,
} from '@lib/flashcards'
import { cards } from '@data/flashcards'
import { cardNotesSpec, flagsSpec, progressSpec } from '@lib/syncDocs'
import { syncState, useSyncedDoc } from '../sync'

/**
 * 原语闪卡：FSRS 间隔重复（Again / Hard / Good / Easy）+ 每张卡的批注和暂停 + 全部卡片列表。
 * 复习记录、批注、暂停标记各存一个 json（src/data/flashcard-{progress,notes,flags}.json），
 * 改动先存本机，再在设备间同步（GitHub / 本机仓库文件，见 ../sync.ts）。
 */

const NEW_PER_ROUND = 20

/** app：独立的闪卡页（/flashcards）用，去掉长说明，手机上评分按钮固定在屏幕底部，点题目即可翻面 */
const props = defineProps<{ app?: boolean }>()

const progressStore = useSyncedDoc(progressSpec)
const notesStore = useSyncedDoc(cardNotesSpec)
const flagsStore = useSyncedDoc(flagsSpec)
const progress = progressStore.data
const notes = notesStore.data
const flags = flagsStore.data
const storageText = computed(() =>
  syncState.mode === 'github'
    ? '先存在这台设备上，再自动同步到 GitHub，手机和电脑共用一份'
    : syncState.mode === 'file'
      ? '写进仓库的 src/data/flashcard-*.json，随 git 同步'
      : '存在这台设备的浏览器里（右上角 ☁ 连上 GitHub 后可在设备间同步）',
)

// ---------- 筛选与统计 ----------
type DeckFilter = Deck | 'all'
const deck = ref<DeckFilter>('all')
const tab = ref<'review' | 'list'>('review')
const deckCards = computed(() => cards.filter((c) => deck.value === 'all' || c.deck === deck.value))
const deckFilters: DeckFilter[] = ['all', ...DECKS.map((d) => d.id)]
const deckLabel = (d: DeckFilter) => (d === 'all' ? '全部' : DECKS.find((x) => x.id === d)!.label)
const deckStyle = (d: Deck) => ({ background: DECKS.find((x) => x.id === d)?.color })
const deckCount = (d: DeckFilter) => (d === 'all' ? cards.length : cards.filter((c) => c.deck === d).length)

// 学习中的卡几分钟后到期，时间要跟着走
const now = ref(new Date())
let ticker: number | undefined

const BUCKETS: { key: Bucket; label: string }[] = [
  { key: 'due', label: '待复习' },
  { key: 'learning', label: '学习中' },
  { key: 'new', label: '新卡' },
  { key: 'later', label: '未到期' },
  { key: 'suspended', label: '已暂停' },
]
const buckets = computed(() => countBuckets(deckCards.value, progress.value, now.value, flags.value))
const bucketOfCard = (id: string) => bucketOf(progress.value[id], now.value, isSuspended(flags.value, id))
const segments = computed(() => {
  const n = deckCards.value.length || 1
  return BUCKETS.filter((b) => buckets.value[b.key]).map((b) => ({ key: b.key, w: (100 * buckets.value[b.key]) / n }))
})

// ---------- 复习 ----------
// 这一轮跳过的卡，按跳过的先后；它们排到最后，别的出完了再出
const deferred = ref<string[]>([])
const newLimit = ref(NEW_PER_ROUND)
const newSeen = ref(0)
const reviewed = ref(0)
const flipped = ref(false)

const cur = computed(() =>
  nextCard(deckCards.value, progress.value, now.value, {
    newLeft: newLimit.value - newSeen.value,
    deferred: deferred.value,
    flags: flags.value,
  }),
)
const curSched = computed(() => (cur.value ? progress.value[cur.value.id] : undefined))
watch(() => cur.value?.id, () => { flipped.value = false; editing.value = null })
const pv = computed(() => (cur.value && flipped.value ? preview(curSched.value, now.value) : null))
const newLeft = computed(() => Math.min(buckets.value.new, Math.max(0, newLimit.value - newSeen.value)))
const nextDue = computed(() => nextDueAt(deckCards.value, progress.value, flags.value))

function restart() {
  deferred.value = []
  newLimit.value = NEW_PER_ROUND
  newSeen.value = 0
  reviewed.value = 0
  now.value = new Date()
}
watch(deck, restart)

function flip() { if (cur.value) flipped.value = true }
/** 排到这一轮最后；再跳一次就再排到最后 */
function skip() {
  const c = cur.value
  if (!c) return
  saveNote()
  deferred.value = [...deferred.value.filter((id) => id !== c.id), c.id]
}
function give(g: Grade) {
  const c = cur.value
  if (!c || !flipped.value) return
  saveNote() // 写到一半就评分：先把批注存下来，换卡时草稿不会丢
  const t = new Date()
  if (bucketOf(progress.value[c.id], t) === 'new') newSeen.value++
  progress.value = review(progress.value, c.id, g, t)
  deferred.value = deferred.value.filter((id) => id !== c.id)
  now.value = t
  reviewed.value++
  progressStore.persist()
}
/** 暂停：不再出现在复习里，直到恢复；复习记录原样保留 */
function suspend(id: string, on: boolean) {
  saveNote()
  flags.value = setSuspended(flags.value, id, on)
  flagsStore.persist()
}

function onKey(e: KeyboardEvent) {
  if (tab.value !== 'review' || e.metaKey || e.ctrlKey || e.altKey) return
  const t = e.target as HTMLElement | null
  if (t && (t.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName))) return
  const g = GRADES.find((x) => x.key === e.key)
  if (e.key === ' ' && cur.value) {
    e.preventDefault()
    flipped.value ? give(GRADES[2].grade) : flip()
  } else if (g && flipped.value) give(g.grade)
  else if ((e.key === 'n' || e.key === 'N') && flipped.value && cur.value) {
    e.preventDefault() // 不然这个 n 会打进刚聚焦的输入框
    editNote(cur.value.id)
  } else if (e.key === 'ArrowRight') skip()
}

// ---------- 批注 ----------
const editing = ref<string | null>(null)
const draft = ref('')
const vFocus: Directive<HTMLTextAreaElement> = { mounted: (el) => el.focus() }

function editNote(id: string) {
  editing.value = id
  draft.value = noteOf(notes.value, id)
}
function saveNote() {
  if (editing.value === null) return
  if (draft.value.trim() !== noteOf(notes.value, editing.value)) {
    notes.value = setNote(notes.value, editing.value, draft.value)
    notesStore.persist()
  }
  editing.value = null
}
function cancelNote() { editing.value = null }

// ---------- 列表 ----------
const open = ref<Set<string>>(new Set())
function toggleOpen(id: string) {
  const s = new Set(open.value)
  s.has(id) ? s.delete(id) : s.add(id)
  open.value = s
}
function dueText(id: string) {
  const s = progress.value[id]
  if (!s || s.state === State.New) return '—'
  const ms = Date.parse(s.due) - now.value.getTime()
  return ms <= 0 ? '已到期' : `${fmtInterval(ms)}后`
}

// 切回来时更新「现在」，到期的卡才会出来；别的设备的改动由 ../sync.ts 拉过来
function onVisible() {
  if (document.visibilityState === 'visible') now.value = new Date()
}

onMounted(() => {
  window.addEventListener('keydown', onKey)
  document.addEventListener('visibilitychange', onVisible)
  ticker = window.setInterval(() => { now.value = new Date() }, 30_000)
  // useSyncedDoc 的 onMounted 先跑，这时本机缓存已经并进来了
  restart()
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey)
  document.removeEventListener('visibilitychange', onVisible)
  clearInterval(ticker)
})
</script>

<template>
  <div class="widget fc" :class="{ 'fc-app': props.app }">
    <div class="fc-head">
      <div class="toggle-group">
        <button v-for="d in deckFilters" :key="d" :class="{ active: deck === d }" @click="deck = d">
          {{ deckLabel(d) }} <span class="fc-n">{{ deckCount(d) }}</span>
        </button>
      </div>
      <div class="toggle-group">
        <button :class="{ active: tab === 'review' }" @click="tab = 'review'">复习</button>
        <button :class="{ active: tab === 'list' }" @click="tab = 'list'">全部卡片</button>
      </div>
    </div>

    <div class="fc-bar">
      <i v-for="s in segments" :key="s.key" :class="'b-' + s.key" :style="{ width: s.w + '%' }" />
    </div>
    <div class="fc-legend">
      <span v-for="b in BUCKETS" :key="b.key" :class="'b-' + b.key"><i />{{ b.label }} <b>{{ buckets[b.key] }}</b></span>
    </div>

    <!-- ========== 复习 ========== -->
    <template v-if="tab === 'review'">
      <div v-if="cur" class="fc-card">
        <div class="fc-meta">
          <span class="fc-deck" :style="deckStyle(cur.deck)">{{ deckLabel(cur.deck) }}</span>
          <span class="muted">{{ cur.topic }}</span>
          <span class="fc-state" :class="'b-' + bucketOf(curSched, now)">{{ STATE_LABEL[curSched?.state ?? State.New] }}</span>
          <span v-if="curSched" class="fc-hist muted">复习 {{ curSched.reps }} 次 · 忘记 {{ curSched.lapses }} 次</span>
          <span class="fc-left muted">本轮还有新卡 {{ newLeft }}</span>
        </div>
        <div class="fc-q" :class="{ tappable: props.app && !flipped }" v-html="inlineMd(cur.q)" @click="props.app && flip()" />

        <button v-if="!flipped" class="fc-flip" @click="flip">翻面 <kbd>空格</kbd></button>
        <template v-else>
          <div class="fc-a" v-html="inlineMd(cur.a)" />
          <pre v-if="cur.code" class="fc-code"><code>{{ cur.code }}</code></pre>
          <a v-if="cur.ref" class="fc-ref" :href="withBase(cur.ref)" target="_blank">出处 →</a>
          <div class="fc-notebox">
            <div v-if="editing === cur.id" class="fc-note-edit">
              <textarea
                v-model="draft"
                v-focus
                rows="3"
                placeholder="自己的理解、容易错的地方……（支持 `code`）"
                @keydown.esc.prevent="cancelNote"
                @keydown.ctrl.enter.prevent="saveNote"
                @keydown.meta.enter.prevent="saveNote"
              />
              <div class="fc-note-actions">
                <button class="btn" @click="saveNote">保存 <kbd>Ctrl+Enter</kbd></button>
                <button class="btn" @click="cancelNote">取消 <kbd>Esc</kbd></button>
              </div>
            </div>
            <div v-else-if="noteOf(notes, cur.id)" class="fc-note-text" title="点击编辑" @click="editNote(cur.id)">
              <span class="fc-note-icon">✎</span><span v-html="inlineMd(noteOf(notes, cur.id))" />
            </div>
            <button v-else class="fc-note-add" @click="editNote(cur.id)">✎ 加批注 <kbd>N</kbd></button>
          </div>

          <div class="fc-rate">
            <button v-for="g in GRADES" :key="g.grade" :class="'g' + g.grade" :title="g.hint" @click="give(g.grade)">
              <span class="fc-ivl">{{ pv ? fmtInterval(Date.parse(pv[g.grade].due) - now.getTime()) : '' }}</span>
              <b>{{ g.name }}</b>
              <span class="fc-hint">{{ g.hint }}</span>
              <kbd>{{ g.key }}</kbd>
            </button>
          </div>
        </template>
        <div class="fc-foot">
          <button class="btn" title="不再出现在复习里，直到在「全部卡片」里恢复" @click="suspend(cur.id, true)">暂停这张</button>
          <button class="btn" title="排到这一轮最后，别的卡出完了再出" @click="skip">跳过 <kbd>→</kbd></button>
          <span v-if="deferred.length" class="muted fc-deferred">已跳过 {{ deferred.length }} 张，排在最后</span>
        </div>
      </div>

      <div v-else class="fc-done">
        <p>
          这一轮没有要复习的卡了<template v-if="reviewed">，复习了 <b>{{ reviewed }}</b> 次</template>。
          <template v-if="nextDue && nextDue > now.getTime()">下一张 <b>{{ fmtInterval(nextDue - now.getTime()) }}</b>后到期。</template>
        </p>
        <p v-if="buckets.suspended" class="muted">已暂停 {{ buckets.suspended }} 张，在「全部卡片」里可以恢复。</p>
        <div class="fc-done-btns">
          <button v-if="buckets.new && !newLeft" class="btn" @click="newLimit += 10">再学 10 张新卡</button>
        </div>
      </div>

      <p v-if="props.app" class="muted fc-note">
        改动{{ storageText }}。<a :href="withBase('/handson/flashcards')">评分规则和快捷键 →</a>
      </p>
      <p v-else class="muted fc-note">
        FSRS 按你的每次评分估计记忆的衰减，算出下次该复习的时间：先出到期的卡，再出新卡（每轮 {{ NEW_PER_ROUND }} 张）。
        按钮上方是选它之后多久再出现。翻面后按 <kbd>N</kbd> 给这张卡写批注。
        跳过的卡排到这一轮最后；暂停的卡不再出现，直到在「全部卡片」里恢复。
        复习记录、批注和暂停{{ storageText }}。
      </p>
    </template>

    <!-- ========== 列表 ========== -->
    <table v-else class="fc-table">
      <thead><tr><th>状态</th><th>组</th><th>题目</th><th>下次</th><th>稳定度</th></tr></thead>
      <tbody>
        <template v-for="c in deckCards" :key="c.id">
          <tr class="fc-row" @click="toggleOpen(c.id)">
            <td>
              <span class="fc-state" :class="'b-' + bucketOfCard(c.id)">{{
                isSuspended(flags, c.id) ? '已暂停' : STATE_LABEL[progress[c.id]?.state ?? State.New]
              }}</span>
            </td>
            <td><span class="fc-deck" :style="deckStyle(c.deck)">{{ deckLabel(c.deck) }}</span></td>
            <td><span class="muted fc-topic">{{ c.topic }}</span> <span v-html="inlineMd(c.q)" /><span v-if="noteOf(notes, c.id)" class="fc-has-note" title="有批注"> ✎</span></td>
            <td class="muted fc-num">{{ dueText(c.id) }}</td>
            <td class="muted fc-num" title="记忆稳定度：降到 90% 记得住所需的天数">{{ progress[c.id] && progress[c.id].state !== State.New ? `${progress[c.id].stability.toFixed(1)}天` : '—' }}</td>
          </tr>
          <tr v-if="open.has(c.id)" class="fc-ans">
            <td colspan="5">
              <div class="fc-ans-a" v-html="inlineMd(c.a)" />
              <pre v-if="c.code" class="fc-code"><code>{{ c.code }}</code></pre>
              <a v-if="c.ref" class="fc-ref" :href="withBase(c.ref)">出处 →</a>
              <button class="btn fc-suspend" @click="suspend(c.id, !isSuspended(flags, c.id))">
                {{ isSuspended(flags, c.id) ? '恢复复习' : '暂停这张' }}
              </button>
              <div class="fc-notebox">
                <div v-if="editing === c.id" class="fc-note-edit">
                  <textarea
                    v-model="draft"
                    v-focus
                    rows="3"
                    placeholder="自己的理解、容易错的地方……（支持 `code`）"
                    @keydown.esc.prevent="cancelNote"
                    @keydown.ctrl.enter.prevent="saveNote"
                    @keydown.meta.enter.prevent="saveNote"
                  />
                  <div class="fc-note-actions">
                    <button class="btn" @click="saveNote">保存 <kbd>Ctrl+Enter</kbd></button>
                    <button class="btn" @click="cancelNote">取消 <kbd>Esc</kbd></button>
                  </div>
                </div>
                <div v-else-if="noteOf(notes, c.id)" class="fc-note-text" title="点击编辑" @click="editNote(c.id)">
                  <span class="fc-note-icon">✎</span><span v-html="inlineMd(noteOf(notes, c.id))" />
                </div>
                <button v-else class="fc-note-add" @click="editNote(c.id)">✎ 加批注</button>
              </div>
            </td>
          </tr>
        </template>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.b-due { --c: var(--fam4); } .b-learning { --c: var(--fam3); } .b-new { --c: var(--st-todo); } .b-later { --c: var(--fam1); }
.b-suspended { --c: var(--vp-c-text-3); }
.g1 { --c: var(--fam4); } .g2 { --c: var(--fam3); } .g3 { --c: var(--fam1); } .g4 { --c: var(--fam2); }

.fc-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px; margin-bottom: 12px; }
.fc-n { opacity: 0.7; font-family: var(--vp-font-family-mono); font-size: 11px; }

.fc-bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--vp-c-bg); }
.fc-bar i { background: var(--c); }
.fc-legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin: 6px 0 12px; font-size: 12px; color: var(--wg-muted); }
.fc-legend i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; background: var(--c); margin-right: 4px; }
.fc-legend b { color: var(--wg-text); font-family: var(--vp-font-family-mono); }

.fc-card { border: 1px solid var(--wg-border); border-radius: 10px; background: var(--vp-c-bg); padding: 16px 18px; min-height: 180px; }
.fc-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; font-size: 12px; margin-bottom: 10px; }
.fc-left { margin-left: auto; }
.fc-hist, .fc-left { font-size: 11.5px; }
.fc-deck { padding: 1px 7px; border-radius: 10px; font: 600 11px/1.6 var(--vp-font-family-mono); color: #fff; background: #ee4c2c; }
.fc-state { padding: 1px 7px; border-radius: 4px; font-size: 11px; line-height: 1.6; white-space: nowrap; color: var(--c); background: color-mix(in srgb, var(--c) 14%, transparent); }
.fc-q { font-size: 16px; font-weight: 600; line-height: 1.7; }
.fc-a { margin-top: 14px; padding-top: 12px; border-top: 1px dashed var(--wg-border); line-height: 1.8; }
/* 答案里的 \n 显示成换行，方便写 1. 2. 3. 和多行公式 */
.fc-a, .fc-ans-a { white-space: pre-line; }
.fc-q :deep(code), .fc-a :deep(code), .fc-table :deep(code) { font-size: 0.88em; padding: 1px 5px; border-radius: 4px; background: var(--vp-c-bg-soft); }
.fc-code { margin: 10px 0 0; padding: 10px 12px; border-radius: 8px; background: var(--vp-code-block-bg); font-size: 12.5px; line-height: 1.6; overflow-x: auto; }
.fc-code code { font-family: var(--vp-font-family-mono); color: var(--vp-code-block-color); }
.fc-ref { display: inline-block; margin-top: 8px; font-size: 12.5px; }

.fc-flip { margin-top: 18px; width: 100%; padding: 10px; border: 1px dashed var(--wg-border); border-radius: 8px; background: transparent; color: var(--wg-muted); cursor: pointer; font-size: 13px; }
.fc-flip:hover { border-color: var(--vp-c-brand-1); color: var(--vp-c-brand-1); }

.fc-rate { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; margin-top: 16px; }
@media (max-width: 520px) { .fc-rate { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.fc-rate button { position: relative; display: flex; flex-direction: column; align-items: center; gap: 1px; padding: 6px 6px 8px; border: 1px solid var(--wg-border); border-top: 3px solid var(--c); border-radius: 6px; background: var(--vp-c-bg); color: var(--wg-text); cursor: pointer; }
.fc-rate button:hover { background: color-mix(in srgb, var(--c) 12%, var(--vp-c-bg)); }
.fc-rate b { font-size: 13.5px; color: var(--c); }
.fc-ivl { font: 600 11.5px/1.4 var(--vp-font-family-mono); color: var(--wg-muted); }
.fc-hint { font-size: 11px; line-height: 1.35; color: var(--wg-muted); text-align: center; }
.fc-rate kbd { position: absolute; top: 4px; right: 5px; }
kbd { padding: 0 5px; border: 1px solid var(--wg-border); border-radius: 4px; font: 11px/1.5 var(--vp-font-family-mono); color: var(--wg-muted); background: var(--vp-c-bg-soft); }

.fc-notebox { margin-top: 12px; }
.fc-note-text { display: flex; gap: 6px; padding: 8px 10px; border-left: 3px solid var(--vp-c-yellow-1); border-radius: 0 6px 6px 0; background: var(--vp-c-yellow-soft); font-size: 13px; line-height: 1.7; white-space: pre-wrap; cursor: text; }
.fc-note-icon { color: var(--vp-c-yellow-1); }
.fc-note-add { padding: 3px 10px; border: 1px dashed var(--wg-border); border-radius: 6px; background: transparent; color: var(--wg-muted); font-size: 12px; cursor: pointer; }
.fc-note-add:hover { border-color: var(--vp-c-yellow-1); color: var(--vp-c-yellow-1); }
.fc-note-edit textarea { width: 100%; padding: 6px 8px; border: 1px solid var(--vp-c-yellow-1); border-radius: 6px; background: var(--vp-c-bg); color: var(--wg-text); font: inherit; font-size: 13px; line-height: 1.6; resize: vertical; }
.fc-note-actions { display: flex; gap: 6px; margin-top: 4px; }
.fc-has-note { color: var(--vp-c-yellow-1); }
.fc-foot { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 8px; margin-top: 12px; }
.fc-deferred { order: -1; margin-right: auto; font-size: 12px; }
.fc-suspend { margin: 8px 0 0 12px; padding: 2px 10px; font-size: 12px; }
.fc-done { padding: 22px; text-align: center; border: 1px dashed var(--wg-border); border-radius: 10px; }
.fc-done p { margin: 0 0 10px; }
.fc-done-btns { display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; }
.fc-note { font-size: 12px; line-height: 1.7; margin-top: 12px; }

.fc-table { width: 100%; display: table; }
.fc-row { cursor: pointer; }
.fc-row:hover td { background: var(--vp-c-bg); }
.fc-topic { font-size: 11.5px; }
.fc-num { font-family: var(--vp-font-family-mono); font-size: 11.5px; white-space: nowrap; }
.fc-ans td { background: var(--vp-c-bg); line-height: 1.75; }
/* 手机上没有键盘，快捷键提示只占地方 */
@media (pointer: coarse) { kbd { display: none; } }

/* ---------- 独立页面（app） ---------- */
.fc-app { border: none; background: transparent; padding: 0; margin: 0; }
.fc-app .fc-card { display: flex; flex-direction: column; min-height: 52vh; padding: 20px 20px 16px; }
.fc-app .fc-q { font-size: 18px; }
.fc-app .fc-q.tappable { flex: 1; cursor: pointer; -webkit-tap-highlight-color: transparent; }
.fc-app .fc-a { font-size: 15.5px; }
.fc-app .fc-foot { margin-top: auto; padding-top: 12px; }
.fc-app .fc-note a { color: var(--vp-c-brand-1); text-decoration: none; }
.fc-app .fc-table th, .fc-app .fc-table td { padding: 8px 10px; border-bottom: 1px solid var(--wg-border); text-align: left; vertical-align: top; }
.fc-app .fc-table th { font-size: 12px; color: var(--wg-muted); font-weight: 500; }
@media (max-width: 640px) {
  .fc-app .fc-card { padding: 16px 14px 12px; border-radius: 12px; }
  /* 翻面和评分固定在屏幕底部，单手拇指够得着 */
  .fc-app .fc-flip,
  .fc-app .fc-rate {
    position: sticky;
    bottom: 0;
    z-index: 2;
    margin-left: -14px;
    margin-right: -14px;
    width: auto;
    padding: 10px 14px calc(10px + env(safe-area-inset-bottom));
    border: none;
    border-top: 1px solid var(--wg-border);
    border-radius: 0;
    background: var(--vp-c-bg);
  }
  .fc-app .fc-flip { font-size: 15px; color: var(--vp-c-brand-1); }
  .fc-app .fc-rate { grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; }
  .fc-app .fc-rate button { padding: 8px 2px 10px; }
  .fc-app .fc-hint { display: none; }
  .fc-app .fc-meta .fc-hist, .fc-app .fc-meta .fc-left { display: none; }
  /* 列表窄屏只留状态、题目、下次；题目里的长代码允许断行，不然整张表被撑出屏幕 */
  .fc-app .fc-table td:nth-child(3), .fc-app .fc-table td:nth-child(3) :deep(code) { overflow-wrap: anywhere; word-break: break-word; }
  .fc-app .fc-table th, .fc-app .fc-table td { padding: 8px 6px; }
  .fc-app .fc-table th:nth-child(2), .fc-app .fc-table td:nth-child(2),
  .fc-app .fc-table th:nth-child(5), .fc-app .fc-table td:nth-child(5) { display: none; }
}
</style>
