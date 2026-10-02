<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { FAM_LEVELS, FAM_ORDER, countByFam, famCls, famInfo, famKey, isPassing } from '@lib/fam'
import { challengeById, urlOf } from '@lib/leetgpu'
import type { LeetGPUChallenge } from '@lib/leetgpu'
import { data as topics } from '../../data/topics.data'
import type { Status, Topic } from '../../data/topics.data'

// 按 tag 分赛道：带 triton / cuda tag 的归对应赛道，其余都是 PyTorch 白板题
const LANES = [
  { key: 'pytorch', label: 'PyTorch' },
  { key: 'triton', label: 'Triton' },
  { key: 'cuda', label: 'CUDA' },
]
const laneOf = (t: Topic) => (t.tags.includes('cuda') ? 'cuda' : t.tags.includes('triton') ? 'triton' : 'pytorch')

interface Card extends Topic { oj: LeetGPUChallenge[] }

const items = computed<Card[]>(() =>
  topics
    .filter((t) => t.domain === 'handson' && !t.isIndex)
    .map((t) => ({ ...t, oj: t.leetgpu.map((id) => challengeById.get(id)!) })),
)

// 进度条段：从最熟到最生，未评画在最后
function segments(list: Card[]) {
  const c = countByFam(list)
  const n = Math.max(1, list.length)
  return FAM_LEVELS.filter((f) => c[famKey(f)]).map((f) => ({ cls: famCls(f), w: (100 * c[famKey(f)]) / n }))
}
const barTip = (list: Card[]) => {
  const c = countByFam(list)
  return FAM_LEVELS.map((f) => `${famInfo(f).short} ${c[famKey(f)]}`).join(' · ')
}

const counts = computed(() => countByFam(items.value))
const passing = (list: Card[]) => list.filter((t) => isPassing(t.familiarity)).length

const lanes = computed(() =>
  LANES.map((l) => ({ ...l, list: items.value.filter((t) => laneOf(t) === l.key) })).filter((l) => l.list.length),
)

// 下一题：越生越先；同档按题目顺序
const next = computed(() =>
  items.value
    .filter((t) => t.familiarity === null || t.familiarity > 0)
    .sort((a, b) => FAM_ORDER[famKey(a.familiarity)] - FAM_ORDER[famKey(b.familiarity)] || a.order - b.order)[0],
)

const noteText: Record<Status, string> = { todo: '笔记未写', draft: '笔记草稿', reviewed: '笔记已复习' }
const lgDiff: Record<string, string> = { easy: 'Easy', medium: 'Med', hard: 'Hard' }
</script>

<template>
  <div class="hp">
    <div class="hp-head">
      <div>
        <span class="hp-num">{{ passing(items) }}</span><span class="hp-of">/ {{ items.length }} 写得对</span>
        <span class="hp-hint" title="熟练度到 L2（思路会 · 细节易写错）及以上才算">≥ L2</span>
      </div>
      <a v-if="next" :href="withBase(next.url)" class="hp-next">
        <span class="hp-next-tag">下一题</span>{{ next.title }} →
      </a>
    </div>

    <div class="hp-bar" :title="barTip(items)">
      <i v-for="s in segments(items)" :key="s.cls" :class="s.cls" :style="{ width: s.w + '%' }" />
    </div>

    <div class="hp-keys">
      <span v-for="f in FAM_LEVELS" :key="famKey(f)" class="hp-key" :class="famCls(f)">
        <i />{{ famInfo(f).short }} {{ famInfo(f).label }} <b>{{ counts[famKey(f)] }}</b>
      </span>
    </div>

    <section v-for="l in lanes" :key="l.key" class="hp-lane">
      <div class="hp-lane-head">
        <span class="hp-lane-name">{{ l.label }}</span>
        <span class="hp-lane-num">{{ passing(l.list) }}/{{ l.list.length }}</span>
        <div class="hp-bar sm" :title="barTip(l.list)">
          <i v-for="s in segments(l.list)" :key="s.cls" :class="s.cls" :style="{ width: s.w + '%' }" />
        </div>
      </div>

      <div class="hp-cards">
        <div v-for="t in l.list" :key="t.url" class="hp-card" :class="{ next: t.url === next?.url }">
          <span
            class="hp-sq"
            :class="famCls(t.familiarity)"
            :title="`熟练度 ${famInfo(t.familiarity).short} · ${famInfo(t.familiarity).label}（改 frontmatter 的 familiarity）`"
          >{{ famInfo(t.familiarity).short }}</span>

          <div class="hp-body">
            <a :href="withBase(t.url)" class="hp-title" :title="t.title">{{ t.title }}</a>
            <div class="hp-meta">
              <span class="hp-diff" :title="`难度 ${t.difficulty} / 5`">
                <b v-for="n in 5" :key="n" :class="{ on: n <= t.difficulty }" />
              </span>
              <span class="hp-note" :class="t.status">{{ noteText[t.status] }}</span>
            </div>
            <div v-if="t.oj.length" class="hp-oj">
              <a
                v-for="c in t.oj"
                :key="c.id"
                :href="urlOf(c)"
                target="_blank"
                rel="noopener"
                class="hp-lg"
                :class="c.difficulty"
                :title="`LeetGPU #${c.id} · ${c.title} · ${lgDiff[c.difficulty]}`"
              ><i />#{{ c.id }} {{ c.title }}</a>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.hp { border: 1px solid var(--wg-border); border-radius: var(--wg-radius); background: var(--wg-bg); padding: 16px 18px 18px; margin: 16px 0 24px; }
.hp-head { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px 16px; }
.hp-num { font-size: 28px; font-weight: 700; font-family: var(--vp-font-family-mono); color: var(--vp-c-text-1); }
.hp-of { margin-left: 6px; font-size: 14px; color: var(--wg-muted); }
.hp-hint { margin-left: 8px; font-size: 12px; font-family: var(--vp-font-family-mono); color: var(--wg-muted); cursor: help; border-bottom: 1px dotted var(--wg-muted); }

.hp-next { display: inline-flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 500; color: var(--vp-c-brand-1); text-decoration: none; }
.hp-next:hover { text-decoration: underline; }
.hp-next-tag { font-size: 12px; padding: 1px 8px; border-radius: 10px; background: var(--vp-c-brand-soft); }

/* 熟练度色：进度条段 / 图例 / 方块共用 */
.f0 { --c: var(--fam0); } .f1 { --c: var(--fam1); } .f1_5 { --c: var(--fam15); } .f2 { --c: var(--fam2); }
.f3 { --c: var(--fam3); } .f3_5 { --c: var(--fam35); } .f4 { --c: var(--fam4); }

.hp-bar { display: flex; height: 10px; margin: 10px 0 8px; border-radius: 5px; overflow: hidden; background: var(--vp-c-bg); border: 1px solid var(--wg-border); }
.hp-bar.sm { height: 6px; margin: 0; flex: 1; min-width: 60px; }
.hp-bar i { display: block; height: 100%; background: var(--c); }
.hp-bar i.fnone { background: var(--st-todo); opacity: 0.45; }

.hp-keys { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--wg-muted); }
.hp-key { display: inline-flex; align-items: center; gap: 5px; }
.hp-key i { width: 9px; height: 9px; border-radius: 3px; background: var(--c); }
.hp-key.fnone i { background: transparent; border: 1px dashed var(--st-todo); }
.hp-key b { color: var(--vp-c-text-1); font-variant-numeric: tabular-nums; }

.hp-lane { margin-top: 20px; }
.hp-lane-head { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.hp-lane-name { font-size: 14px; font-weight: 600; color: var(--vp-c-text-1); }
.hp-lane-num { font-size: 12px; font-family: var(--vp-font-family-mono); color: var(--wg-muted); }

.hp-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 8px; }
.hp-card { display: flex; gap: 10px; padding: 10px; border: 1px solid var(--wg-border); border-radius: 8px; background: var(--vp-c-bg); min-width: 0; transition: border-color 0.15s; }
.hp-card:hover { border-color: var(--vp-c-brand-1); }
.hp-card.next { border-color: var(--vp-c-brand-1); box-shadow: 0 0 0 2px var(--vp-c-brand-soft); }

.hp-sq { flex: none; display: grid; place-items: center; width: 40px; height: 40px; border-radius: 6px; font: 700 12px/1 var(--vp-font-family-mono); color: #fff; background: var(--c); cursor: help; }
.hp-sq.fnone { background: transparent; color: var(--wg-muted); border: 1.5px dashed var(--st-todo); }

.hp-body { min-width: 0; flex: 1; }
.hp-title { display: block; font-size: 14px; font-weight: 600; color: var(--vp-c-text-1); text-decoration: none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hp-title:hover { color: var(--vp-c-brand-1); }
.hp-meta { display: flex; align-items: center; gap: 8px; margin-top: 3px; font-size: 12px; color: var(--wg-muted); }
.hp-diff { display: inline-flex; gap: 2px; }
.hp-diff b { width: 4px; height: 10px; border-radius: 1px; background: var(--wg-border); }
.hp-diff b.on { background: var(--wg-muted); }
.hp-note.draft { color: var(--st-draft); }
.hp-note.reviewed { color: var(--st-reviewed); }

.hp-oj { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.hp-lg { display: inline-flex; align-items: center; gap: 4px; max-width: 100%; padding: 1px 6px; border-radius: 5px; border: 1px solid var(--wg-border); background: var(--wg-bg); font-size: 11.5px; color: var(--vp-c-text-2); text-decoration: none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hp-lg:hover { border-color: var(--vp-c-brand-1); color: var(--vp-c-brand-1); }
.hp-lg i { flex: none; width: 6px; height: 6px; border-radius: 50%; }
.hp-lg.easy i { background: var(--lg-easy); }
.hp-lg.medium i { background: var(--lg-medium); }
.hp-lg.hard i { background: var(--lg-hard); }

@media (max-width: 640px) {
  .hp { padding: 14px; }
  .hp-cards { grid-template-columns: 1fr; }
}
</style>
