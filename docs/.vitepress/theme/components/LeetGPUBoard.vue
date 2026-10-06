<script setup lang="ts">
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import { famCls, famInfo } from '@lib/fam'
import { challenges, urlOf } from '@lib/leetgpu'
import type { LeetGPUChallenge, LeetGPUDifficulty } from '@lib/leetgpu'
import { data as topics } from '../../data/topics.data'
import type { Status, Topic } from '../../data/topics.data'

// 「做过」= docs/leetgpu/ 下有一页 frontmatter 挂着这个题号；熟练度、笔记状态都从那页读
const solvedById = computed(() => {
  const m = new Map<number, Topic>()
  for (const t of topics) if (t.domain === 'leetgpu' && !t.isIndex) for (const id of t.leetgpu) m.set(id, t)
  return m
})

interface Row extends LeetGPUChallenge { page?: Topic }
const rows = computed<Row[]>(() =>
  [...challenges].sort((a, b) => a.id - b.id).map((c) => ({ ...c, page: solvedById.value.get(c.id) })),
)

const DIFFS: { key: LeetGPUDifficulty; label: string }[] = [
  { key: 'easy', label: 'Easy' },
  { key: 'medium', label: 'Med.' },
  { key: 'hard', label: 'Hard' },
]

// 筛选
type Show = 'all' | 'solved' | 'todo'
const show = ref<Show>('solved')
const diff = ref<LeetGPUDifficulty | 'any'>('any')
const q = ref('')
const visible = computed(() => {
  const kw = q.value.trim().toLowerCase()
  return rows.value.filter(
    (r) =>
      (show.value === 'all' || (show.value === 'solved') === !!r.page) &&
      (diff.value === 'any' || r.difficulty === diff.value) &&
      (!kw || r.title.toLowerCase().includes(kw) || String(r.id) === kw),
  )
})

// dev 时未做的题可以一键建题解骨架页（接口见 docs/.vitepress/leetgpuNewPage.ts），生产构建里不显示
const canCreate = import.meta.env.DEV
const creating = ref<number | null>(null)
async function createPage(id: number) {
  creating.value = id
  try {
    const res = await fetch('/__leetgpu-new', { method: 'POST', body: JSON.stringify({ id }) })
    const data = await res.json()
    if (!data.url) throw new Error(data.error ?? res.statusText)
    window.location.href = withBase(data.url)
  } catch (e) {
    console.error('[leetgpu] 新建题解失败', e)
    creating.value = null
  }
}

const noteText: Record<Status, string> = { todo: '未写', draft: '草稿', reviewed: '已复习' }
const diffLabel: Record<LeetGPUDifficulty, string> = { easy: 'Easy', medium: 'Med.', hard: 'Hard' }
</script>

<template>
  <div class="lb">
    <div class="lb-filters">
      <div class="lb-tabs" role="tablist">
        <button v-for="s in (['solved', 'todo', 'all'] as Show[])" :key="s" :class="{ on: show === s }" @click="show = s">
          {{ { solved: '已做', todo: '未做', all: '全部' }[s] }}
        </button>
      </div>
      <div class="lb-tabs">
        <button :class="{ on: diff === 'any' }" @click="diff = 'any'">全部难度</button>
        <button v-for="d in DIFFS" :key="d.key" :class="[d.key, { on: diff === d.key }]" @click="diff = d.key">{{ d.label }}</button>
      </div>
      <input v-model="q" class="lb-search" type="search" placeholder="搜题目或题号" />
    </div>

    <table class="lb-table">
      <thead>
        <tr><th class="c-fam">熟练度</th><th class="c-id">#</th><th>题目</th><th class="c-diff">难度</th><th class="c-note">题解</th><th class="c-oj" /></tr>
      </thead>
      <tbody>
        <tr v-for="r in visible" :key="r.id" :class="{ done: r.page }">
          <td class="c-fam">
            <span
              v-if="r.page"
              class="lb-sq"
              :class="famCls(r.page.familiarity)"
              :title="`${famInfo(r.page.familiarity).short} · ${famInfo(r.page.familiarity).label}（改 frontmatter 的 familiarity）`"
            >{{ famInfo(r.page.familiarity).short }}</span>
          </td>
          <td class="c-id">{{ r.id }}</td>
          <td class="c-title">
            <a v-if="r.page" :href="withBase(r.page.url)">{{ r.title }}</a>
            <span v-else>{{ r.title }}</span>
          </td>
          <td class="c-diff"><span class="lb-d" :class="r.difficulty">{{ diffLabel[r.difficulty] }}</span></td>
          <td class="c-note">
            <span v-if="r.page" class="lb-note" :class="r.page.status">{{ noteText[r.page.status] }}</span>
            <button
              v-else-if="canCreate"
              class="lb-new"
              :disabled="creating !== null"
              :title="`在 docs/leetgpu/ 下新建 #${r.id} 的题解页`"
              @click="createPage(r.id)"
            >{{ creating === r.id ? '…' : '+ 新建' }}</button>
          </td>
          <td class="c-oj"><a :href="urlOf(r)" target="_blank" rel="noopener" title="在 LeetGPU 打开">↗</a></td>
        </tr>
        <tr v-if="!visible.length"><td colspan="6" class="lb-empty">没有符合条件的题</td></tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.lb { border: 1px solid var(--wg-border); border-radius: var(--wg-radius); background: var(--wg-bg); padding: 14px 18px 12px; margin: 16px 0 24px; }

/* 熟练度色，和 HandsonProgress 同一套 */
.f0 { --c: var(--fam0); } .f1 { --c: var(--fam1); } .f1_5 { --c: var(--fam15); } .f2 { --c: var(--fam2); }
.f3 { --c: var(--fam3); } .f3_5 { --c: var(--fam35); } .f4 { --c: var(--fam4); }

.lb-filters { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 8px; }
.lb-tabs { display: inline-flex; border: 1px solid var(--wg-border); border-radius: 6px; overflow: hidden; }
.lb-tabs button { padding: 3px 10px; font-size: 12.5px; color: var(--vp-c-text-2); background: var(--vp-c-bg); }
.lb-tabs button:not(:last-child) { border-right: 1px solid var(--wg-border); }
.lb-tabs button.on { background: var(--vp-c-brand-soft); color: var(--vp-c-brand-1); font-weight: 600; }
.lb-search { flex: 1; min-width: 140px; padding: 3px 10px; font-size: 12.5px; border: 1px solid var(--wg-border); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-1); }

.lb-table { display: table; width: 100%; margin: 0; border-collapse: collapse; font-size: 13.5px; }
.lb-table th, .lb-table td { border: none; border-bottom: 1px solid var(--wg-border); padding: 7px 8px; background: transparent; }
.lb-table tr { background: transparent !important; }
.lb-table th { font-size: 12px; font-weight: 500; color: var(--wg-muted); text-align: left; }
.lb-table tbody tr:hover { background: var(--vp-c-bg-soft) !important; }
.lb-table tbody tr:not(.done) .c-title { color: var(--vp-c-text-2); }
.c-fam { width: 52px; } .c-id { width: 44px; font-family: var(--vp-font-family-mono); color: var(--wg-muted); }
.c-diff { width: 60px; } .c-note { width: 64px; } .c-oj { width: 28px; text-align: center; }
.c-title a { font-weight: 600; color: var(--vp-c-text-1); text-decoration: none; }
.c-title a:hover { color: var(--vp-c-brand-1); }
.c-oj a { color: var(--wg-muted); text-decoration: none; }
.c-oj a:hover { color: var(--vp-c-brand-1); }

.lb-sq { display: inline-grid; place-items: center; min-width: 38px; height: 22px; padding: 0 4px; border-radius: 5px; font: 700 11px/1 var(--vp-font-family-mono); color: #fff; background: var(--c); cursor: help; }
.lb-sq.fnone { background: transparent; color: var(--wg-muted); border: 1.5px dashed var(--st-todo); }
.lb-d { font-weight: 600; font-size: 12.5px; }
.lb-d.easy { color: var(--lg-easy); } .lb-d.medium { color: var(--lg-medium); } .lb-d.hard { color: var(--lg-hard); }
.lb-note { font-size: 12px; color: var(--wg-muted); }
.lb-new { padding: 1px 6px; font-size: 11.5px; color: var(--vp-c-brand-1); border: 1px dashed var(--vp-c-brand-1); border-radius: 5px; white-space: nowrap; }
.lb-new:hover:not(:disabled) { background: var(--vp-c-brand-soft); }
.lb-new:disabled { opacity: 0.5; cursor: wait; }
.lb-note.draft { color: var(--st-draft); } .lb-note.reviewed { color: var(--st-reviewed); }
.lb-empty { text-align: center; color: var(--wg-muted); padding: 18px; }

@media (max-width: 640px) {
  .lb { padding: 14px; }
  .c-note, .c-id { display: none; }
}
</style>
