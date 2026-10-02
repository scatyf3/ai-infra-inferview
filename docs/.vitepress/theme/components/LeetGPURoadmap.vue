<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { FAM_LEVELS, countByFam, famCls, famInfo, famKey } from '@lib/fam'
import { challengeById, urlOf } from '@lib/leetgpu'
import type { LeetGPUDifficulty } from '@lib/leetgpu'
import { roadmap } from '@data/leetgpu-roadmap'
import type { Impl, RoadmapGroup } from '@data/leetgpu-roadmap'
import { data as topics } from '../../data/topics.data'
import type { Topic } from '../../data/topics.data'

// 和 LeetGPUBoard 同一个口径：docs/leetgpu/ 下有页面挂着题号 = 做过
const pageById = computed(() => {
  const m = new Map<number, Topic>()
  for (const t of topics) if (t.domain === 'leetgpu' && !t.isIndex) for (const id of t.leetgpu) m.set(id, t)
  return m
})

const stages = computed(() =>
  roadmap.map((s) => ({
    ...s,
    groups: s.groups.map((g) => {
      const items = g.items.map((it) => ({ ...it, c: challengeById.get(it.id)!, page: pageById.value.get(it.id) }))
      const required = items.filter((i) => !i.optional)
      return { ...g, items, done: required.filter((i) => i.page).length, total: required.length }
    }),
  })),
)

// 总览：题单里的必做题，同一题出现在多个分组只算一次
const tracked = computed(() => {
  const m = new Map<number, (typeof stages.value)[number]['groups'][number]['items'][number]>()
  for (const s of stages.value) for (const g of s.groups) for (const it of g.items) if (!it.optional) m.set(it.id, it)
  return [...m.values()]
})
const solved = computed(() => tracked.value.filter((it) => it.page))

const DIFFS: { key: LeetGPUDifficulty; label: string }[] = [
  { key: 'easy', label: 'Easy' },
  { key: 'medium', label: 'Med.' },
  { key: 'hard', label: 'Hard' },
]
const byDiff = computed(() =>
  DIFFS.map((d) => {
    const total = tracked.value.filter((it) => it.c.difficulty === d.key).length
    const done = solved.value.filter((it) => it.c.difficulty === d.key).length
    return { ...d, total, done, pct: total ? (100 * done) / total : 0 }
  }),
)

const R = 44
const CIRC = 2 * Math.PI * R
const ringDash = computed(() => `${(CIRC * solved.value.length) / Math.max(1, tracked.value.length)} ${CIRC}`)

// 熟练度分布只统计做过的题
const famCounts = computed(() => countByFam(solved.value.map((it) => ({ familiarity: it.page?.familiarity ?? null }))))
const famSegments = computed(() => {
  const n = Math.max(1, solved.value.length)
  return FAM_LEVELS.filter((f) => famCounts.value[famKey(f)]).map((f) => ({
    cls: famCls(f),
    w: (100 * famCounts.value[famKey(f)]) / n,
  }))
})

const pct = (g: { done: number; total: number }) => (g.total ? (100 * g.done) / g.total : 0)
const implLabel: Record<Impl, string> = { torch: 'torch', triton: 'triton' }
const implTip: Record<Impl, string> = { torch: '只在乎 correctness，PyTorch 写对', triton: '在乎效率，Triton / CUDA kernel' }
const diffLabel: Record<LeetGPUDifficulty, string> = { easy: 'E', medium: 'M', hard: 'H' }
const groupKey = (g: RoadmapGroup) => g.key
</script>

<template>
  <div class="lr">
    <div class="lr-sum">
      <div class="lr-top">
        <svg class="lr-ring" viewBox="0 0 110 110" role="img" :aria-label="`已做 ${solved.length} / ${tracked.length}`">
          <circle cx="55" cy="55" :r="R" class="track" />
          <circle cx="55" cy="55" :r="R" class="fill" :stroke-dasharray="ringDash" transform="rotate(-90 55 55)" />
          <text x="55" y="52" class="num">{{ solved.length }}</text>
          <text x="55" y="70" class="of">/ {{ tracked.length }}</text>
        </svg>
        <div class="lr-diffs">
          <div v-for="d in byDiff" :key="d.key" class="lr-diff" :class="d.key">
            <span class="lr-diff-name">{{ d.label }}</span>
            <span class="lr-diff-num"><b>{{ d.done }}</b>/{{ d.total }}</span>
            <div class="lr-diff-bar"><i :style="{ width: d.pct + '%' }" /></div>
          </div>
          <div class="lr-sum-hint">题单必做题，去重后计数；选做不算</div>
        </div>
      </div>

      <div class="lr-fam">
        <div class="lr-fam-head">熟练度 <span>（做过的 {{ solved.length }} 题）</span></div>
        <div class="lr-fbar">
          <i v-for="seg in famSegments" :key="seg.cls" :class="seg.cls" :style="{ width: seg.w + '%' }" />
        </div>
        <div class="lr-keys">
          <span v-for="f in FAM_LEVELS" :key="famKey(f)" class="lr-key" :class="famCls(f)">
            <i />{{ famInfo(f).short }} {{ famInfo(f).label }} <b>{{ famCounts[famKey(f)] }}</b>
          </span>
        </div>
      </div>
    </div>

    <section v-for="s in stages" :key="s.title" class="lr-stage">
      <h3 class="lr-stage-title">{{ s.title }}</h3>

      <div v-for="g in s.groups" :key="groupKey(g)" class="lr-group">
        <div class="lr-group-head">
          <span class="lr-group-title">{{ g.title }}</span>
          <span v-if="g.desc" class="lr-group-desc">{{ g.desc }}</span>
          <span class="lr-group-num" title="不含选做">{{ g.done }}/{{ g.total }}</span>
          <div class="lr-bar"><i :style="{ width: pct(g) + '%' }" /></div>
        </div>

        <ol class="lr-items">
          <li v-for="it in g.items" :key="it.id" :class="{ done: it.page, opt: it.optional }">
            <span
              class="lr-st"
              :class="it.page ? famCls(it.page.familiarity) : 'todo'"
              :title="it.page ? `已做 · ${famInfo(it.page.familiarity).short} ${famInfo(it.page.familiarity).label}` : '未做'"
            >{{ it.page ? '✓' : '' }}</span>
            <span class="lr-id">{{ it.id }}</span>
            <span class="lr-d" :class="it.c.difficulty">{{ diffLabel[it.c.difficulty] }}</span>
            <span class="lr-title">
              <a v-if="it.page" :href="withBase(it.page.url)">{{ it.c.title }}</a>
              <span v-else>{{ it.c.title }}</span>
              <span v-if="it.optional" class="lr-opt">选做</span>
            </span>
            <span class="lr-impls">
              <span v-for="m in it.impl" :key="m" class="lr-impl" :class="m" :title="implTip[m]">{{ implLabel[m] }}</span>
            </span>
            <span class="lr-note">{{ it.note }}</span>
            <a class="lr-oj" :href="urlOf(it.c)" target="_blank" rel="noopener" title="在 LeetGPU 打开">↗</a>
          </li>
        </ol>
      </div>
    </section>
  </div>
</template>

<style scoped>
.lr { margin: 16px 0 24px; }

.lr-sum { border: 1px solid var(--wg-border); border-radius: var(--wg-radius); background: var(--wg-bg); padding: 16px 18px; margin-bottom: 20px; }
.lr-top { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; }
.lr-ring { width: 110px; height: 110px; flex: none; }
.lr-ring .track { fill: none; stroke: var(--wg-border); stroke-width: 8; }
.lr-ring .fill { fill: none; stroke: var(--vp-c-brand-1); stroke-width: 8; stroke-linecap: round; }
.lr-ring .num { text-anchor: middle; font: 700 26px var(--vp-font-family-mono); fill: var(--vp-c-text-1); }
.lr-ring .of { text-anchor: middle; font: 12px var(--vp-font-family-mono); fill: var(--wg-muted); }
.lr-diffs { flex: 1; min-width: 200px; display: grid; gap: 8px; }
.lr-diff { display: grid; grid-template-columns: 48px 60px 1fr; align-items: center; gap: 8px; font-size: 13px; }
.lr-diff.easy { --d: var(--lg-easy); } .lr-diff.medium { --d: var(--lg-medium); } .lr-diff.hard { --d: var(--lg-hard); }
.lr-diff-name { color: var(--d); font-weight: 600; }
.lr-diff-num { font-family: var(--vp-font-family-mono); color: var(--wg-muted); font-size: 12px; }
.lr-diff-num b { color: var(--vp-c-text-1); }
.lr-diff-bar { height: 6px; border-radius: 3px; background: var(--vp-c-bg); border: 1px solid var(--wg-border); overflow: hidden; }
.lr-diff-bar i { display: block; height: 100%; background: var(--d); }
.lr-sum-hint { font-size: 11.5px; color: var(--wg-muted); }

.lr-fam { margin-top: 16px; }
.lr-fam-head { font-size: 13px; font-weight: 600; color: var(--vp-c-text-1); }
.lr-fam-head span { font-weight: 400; color: var(--wg-muted); }
.lr-fbar { display: flex; height: 10px; margin: 6px 0 8px; border-radius: 5px; overflow: hidden; background: var(--vp-c-bg); border: 1px solid var(--wg-border); }
.lr-fbar i { display: block; height: 100%; background: var(--c); }
.lr-fbar i.fnone { background: var(--st-todo); opacity: 0.45; }
.lr-keys { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--wg-muted); }
.lr-key { display: inline-flex; align-items: center; gap: 5px; }
.lr-key i { width: 9px; height: 9px; border-radius: 3px; background: var(--c); }
.lr-key.fnone i { background: transparent; border: 1px dashed var(--st-todo); }
.lr-key b { color: var(--vp-c-text-1); font-variant-numeric: tabular-nums; }
.lr-stage + .lr-stage { margin-top: 20px; }
.lr-stage-title { margin: 0 0 8px !important; padding: 0 !important; border: none !important; font-size: 15px !important; }

.lr-group { border: 1px solid var(--wg-border); border-radius: var(--wg-radius); background: var(--wg-bg); padding: 10px 14px 6px; margin-bottom: 10px; }
.lr-group-head { display: flex; align-items: center; flex-wrap: wrap; gap: 4px 10px; margin-bottom: 4px; }
.lr-group-title { font-size: 14px; font-weight: 600; color: var(--vp-c-text-1); }
.lr-group-desc { font-size: 12px; color: var(--wg-muted); }
.lr-group-num { margin-left: auto; font: 12px var(--vp-font-family-mono); color: var(--wg-muted); }
.lr-bar { flex-basis: 100%; height: 5px; border-radius: 3px; background: var(--vp-c-bg); border: 1px solid var(--wg-border); overflow: hidden; }
.lr-bar i { display: block; height: 100%; background: var(--vp-c-brand-1); }

.lr-items { list-style: none; margin: 6px 0 0 !important; padding: 0 !important; }
.lr-items li {
  display: grid; grid-template-columns: 20px 30px 16px minmax(160px, 1.3fr) 104px 2fr 16px;
  align-items: center; gap: 8px; margin: 0 !important; padding: 5px 0; font-size: 13px; border-top: 1px dashed var(--wg-border);
}
.lr-items li.opt { opacity: 0.72; }

.f0 { --c: var(--fam0); } .f1 { --c: var(--fam1); } .f1_5 { --c: var(--fam15); } .f2 { --c: var(--fam2); }
.f3 { --c: var(--fam3); } .f3_5 { --c: var(--fam35); } .f4 { --c: var(--fam4); } .fnone { --c: var(--st-todo); }
.lr-st { display: grid; place-items: center; width: 18px; height: 18px; border-radius: 50%; font-size: 11px; font-weight: 700; color: #fff; background: var(--c); cursor: help; }
.lr-st.todo { background: transparent; border: 1.5px solid var(--wg-border); }

.lr-id { font: 12px var(--vp-font-family-mono); color: var(--wg-muted); }
.lr-d { font: 700 11.5px var(--vp-font-family-mono); }
.lr-d.easy { color: var(--lg-easy); } .lr-d.medium { color: var(--lg-medium); } .lr-d.hard { color: var(--lg-hard); }
.lr-title { min-width: 0; color: var(--vp-c-text-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lr-title a { font-weight: 600; color: var(--vp-c-text-1); text-decoration: none; }
.lr-title a:hover { color: var(--vp-c-brand-1); }
.lr-opt { margin-left: 6px; padding: 0 5px; border-radius: 4px; border: 1px solid var(--wg-border); font-size: 11px; color: var(--wg-muted); }

.lr-impls { display: flex; gap: 4px; }
.lr-impl { padding: 0 6px; border-radius: 4px; font: 600 11px/18px var(--vp-font-family-mono); cursor: help; }
.lr-impl.torch { color: #ee4c2c; background: color-mix(in srgb, #ee4c2c 12%, transparent); }
.lr-impl.triton { color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); }

.lr-note { min-width: 0; font-size: 12px; color: var(--wg-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lr-oj { color: var(--wg-muted); text-decoration: none; }
.lr-oj:hover { color: var(--vp-c-brand-1); }

@media (max-width: 720px) {
  .lr-items li { grid-template-columns: 20px 30px 16px 1fr auto 16px; }
  .lr-note { display: none; }
}
</style>
