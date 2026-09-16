<script setup lang="ts">
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import { data as topics } from '../../data/topics.data'
import { domains } from '../../domains'
import type { Topic } from '../../data/topics.data'

const W = 1200, H = 1000, CX = W / 2, CY = H / 2
const R_HUB = 190, R_LEAF = 400
const GAP_SLOTS = 1.6
const showRelated = ref(true)

interface LeafNode extends Topic { x: number; y: number; deg: number; color: string }
interface HubNode { dir: string; label: string; short: string; color: string; x: number; y: number; url: string; leaves: LeafNode[]; done: number; draft: number }

const byDomain = computed(() => {
  const m: Record<string, Topic[]> = {}
  for (const t of topics) {
    if (t.isIndex) continue
    ;(m[t.domain] ??= []).push(t)
  }
  return m
})

const layout = computed(() => {
  const ordered = [...domains].sort((a, b) => a.order - b.order)
  const leafCount = ordered.reduce((n, d) => n + (byDomain.value[d.dir]?.length ?? 0), 0)
  const totalSlots = leafCount + ordered.length * GAP_SLOTS
  const slot = (2 * Math.PI) / totalSlots
  let a = -Math.PI / 2 + (slot * GAP_SLOTS) / 2
  const hubs: HubNode[] = []
  for (const d of ordered) {
    const leaves = byDomain.value[d.dir] ?? []
    const start = a
    const nodes: LeafNode[] = leaves.map((t, i) => {
      const ang = start + slot * (i + 0.5)
      return { ...t, x: CX + R_LEAF * Math.cos(ang), y: CY + R_LEAF * Math.sin(ang), deg: (ang * 180) / Math.PI, color: d.color }
    })
    const mid = start + (slot * leaves.length) / 2
    hubs.push({
      dir: d.dir,
      label: d.label,
      short: d.short,
      color: d.color,
      x: CX + R_HUB * Math.cos(mid),
      y: CY + R_HUB * Math.sin(mid),
      url: `/${d.dir}/`,
      leaves: nodes,
      done: leaves.filter((t) => t.status === 'reviewed').length,
      draft: leaves.filter((t) => t.status === 'draft').length,
    })
    a = start + slot * (leaves.length + GAP_SLOTS)
  }
  return hubs
})

const leafByUrl = computed(() => {
  const m: Record<string, LeafNode> = {}
  for (const h of layout.value) for (const l of h.leaves) m[l.url] = l
  return m
})

const relatedEdges = computed(() => {
  const edges: { a: LeafNode; b: LeafNode }[] = []
  const seen = new Set<string>()
  for (const h of layout.value)
    for (const l of h.leaves)
      for (const r of l.related) {
        const target = leafByUrl.value[r] ?? leafByUrl.value[r.replace(/\/$/, '')]
        if (!target) continue
        const key = [l.url, target.url].sort().join('|')
        if (seen.has(key)) continue
        seen.add(key)
        edges.push({ a: l, b: target })
      }
  return edges
})

const totals = computed(() => {
  const all = topics.filter((t) => !t.isIndex)
  return {
    total: all.length,
    reviewed: all.filter((t) => t.status === 'reviewed').length,
    draft: all.filter((t) => t.status === 'draft').length,
  }
})

function labelTransform(n: LeafNode) {
  const flip = n.deg > 90 || n.deg < -90
  return `rotate(${n.deg} ${n.x} ${n.y}) ${flip ? `rotate(180 ${n.x} ${n.y})` : ''}`
}
function labelAnchor(n: LeafNode) {
  return n.deg > 90 || n.deg < -90 ? 'end' : 'start'
}
function labelDx(n: LeafNode) {
  return n.deg > 90 || n.deg < -90 ? -14 : 14
}
const statusText: Record<string, string> = { todo: '未写', draft: '草稿', reviewed: '已复习' }
</script>

<template>
  <div class="km">
    <div class="km-toolbar">
      <div class="km-legend">
        <span><i class="dot todo" />未写</span>
        <span><i class="dot draft" />草稿</span>
        <span><i class="dot reviewed" />已复习</span>
        <span class="km-total">{{ totals.reviewed }} / {{ totals.total }} 已复习，{{ totals.draft }} 草稿</span>
      </div>
      <label class="km-toggle"><input type="checkbox" v-model="showRelated" /> 显示 related 跨链</label>
    </div>

    <svg :viewBox="`0 0 ${W} ${H}`" class="km-svg" role="img" aria-label="知识地图">
      <g v-if="showRelated" class="km-related">
        <path v-for="(e, i) in relatedEdges" :key="i" :d="`M ${e.a.x} ${e.a.y} Q ${CX} ${CY} ${e.b.x} ${e.b.y}`" />
      </g>
      <g v-for="h in layout" :key="h.dir" class="km-domain">
        <line :x1="CX" :y1="CY" :x2="h.x" :y2="h.y" class="km-spoke" :stroke="h.color" />
        <line v-for="l in h.leaves" :key="l.url" :x1="h.x" :y1="h.y" :x2="l.x" :y2="l.y" class="km-edge" :stroke="h.color" />
        <a v-for="l in h.leaves" :key="'n' + l.url" :href="withBase(l.url)" class="km-leaf" :class="l.status">
          <title>{{ l.title }} · {{ statusText[l.status] }} · 难度 {{ l.difficulty }}</title>
          <circle :cx="l.x" :cy="l.y" :r="5 + l.difficulty * 1.2" :stroke="h.color" />
          <text :x="l.x + labelDx(l)" :y="l.y + 4" :text-anchor="labelAnchor(l)" :transform="labelTransform(l)">{{ l.title }}</text>
        </a>
        <a :href="withBase(h.url)" class="km-hub">
          <title>{{ h.label }} · {{ h.done }}/{{ h.leaves.length }} 已复习</title>
          <circle :cx="h.x" :cy="h.y" r="30" :fill="h.color" />
          <text :x="h.x" :y="h.y + 5" text-anchor="middle" class="km-hub-text">{{ h.short }}</text>
        </a>
      </g>
      <circle :cx="CX" :cy="CY" r="46" class="km-center" />
      <text :x="CX" :y="CY - 4" text-anchor="middle" class="km-center-text">AI Infra</text>
      <text :x="CX" :y="CY + 14" text-anchor="middle" class="km-center-sub">interview</text>
    </svg>

    <div class="km-progress">
      <a v-for="h in layout" :key="h.dir" :href="withBase(h.url)" class="km-pg">
        <span class="km-pg-label"><i class="dot" :style="{ background: h.color }" />{{ h.label }}</span>
        <span class="km-pg-bar">
          <i class="reviewed" :style="{ width: (h.done / Math.max(1, h.leaves.length)) * 100 + '%' }" />
          <i class="draft" :style="{ width: (h.draft / Math.max(1, h.leaves.length)) * 100 + '%' }" />
        </span>
        <span class="km-pg-num">{{ h.done }}/{{ h.leaves.length }}</span>
      </a>
    </div>

    <div class="km-list">
      <details v-for="h in layout" :key="h.dir" open>
        <summary><i class="dot" :style="{ background: h.color }" /><a :href="withBase(h.url)">{{ h.label }}</a> <span class="muted">{{ h.done }}/{{ h.leaves.length }}</span></summary>
        <ul>
          <li v-for="l in h.leaves" :key="l.url"><i class="dot" :class="l.status" /><a :href="withBase(l.url)">{{ l.title }}</a></li>
        </ul>
      </details>
    </div>
  </div>
</template>

<style scoped>
.km { max-width: 1152px; margin: 0 auto; padding: 24px 24px 48px; }
.km-toolbar { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 13px; color: var(--vp-c-text-2); margin-bottom: 8px; }
.km-legend { display: flex; gap: 14px; align-items: center; }
.km-total { margin-left: 8px; }
.km-toggle { cursor: pointer; display: flex; gap: 4px; align-items: center; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; background: var(--st-todo); }
.dot.todo { background: transparent; border: 2px solid var(--st-todo); }
.dot.draft { background: var(--st-draft); }
.dot.reviewed { background: var(--st-reviewed); }

.km-svg { width: 100%; height: auto; display: block; }
.km-related path { fill: none; stroke: var(--vp-c-text-3); stroke-width: 1; stroke-dasharray: 4 5; opacity: 0.5; }
.km-spoke { stroke-width: 3; opacity: 0.35; }
.km-edge { stroke-width: 1.2; opacity: 0.45; }
.km-leaf circle { fill: var(--vp-c-bg); stroke-width: 2; transition: r 0.15s; }
.km-leaf.draft circle { fill: var(--st-draft); }
.km-leaf.reviewed circle { fill: var(--st-reviewed); }
.km-leaf text { font-size: 13px; fill: var(--vp-c-text-1); }
.km-leaf:hover circle { stroke-width: 4; }
.km-leaf:hover text { font-weight: 600; fill: var(--vp-c-brand-1); }
.km-hub circle { stroke: var(--vp-c-bg); stroke-width: 3; }
.km-hub:hover circle { stroke: var(--vp-c-text-1); }
.km-hub-text { font-size: 14px; font-weight: 700; fill: #fff; }
.km-center { fill: var(--vp-c-bg-soft); stroke: var(--vp-c-divider); stroke-width: 2; }
.km-center-text { font-size: 16px; font-weight: 700; fill: var(--vp-c-text-1); }
.km-center-sub { font-size: 11px; fill: var(--vp-c-text-2); }

.km-progress { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 8px 24px; margin-top: 16px; }
.km-pg { display: grid; grid-template-columns: 150px 1fr 48px; align-items: center; gap: 8px; font-size: 13px; color: var(--vp-c-text-1); text-decoration: none; }
.km-pg:hover .km-pg-label { color: var(--vp-c-brand-1); }
.km-pg-bar { height: 8px; background: var(--vp-c-bg-soft); border-radius: 4px; overflow: hidden; display: flex; }
.km-pg-bar i { display: block; height: 100%; }
.km-pg-bar .reviewed { background: var(--st-reviewed); }
.km-pg-bar .draft { background: var(--st-draft); }
.km-pg-num { text-align: right; font-family: var(--vp-font-family-mono); font-size: 12px; color: var(--vp-c-text-2); }

.km-list { display: none; margin-top: 16px; }
.km-list summary { cursor: pointer; font-weight: 600; margin: 8px 0 4px; }
.km-list ul { list-style: none; padding-left: 16px; margin: 0; }
.km-list li { padding: 3px 0; font-size: 14px; }
.km-list a { color: var(--vp-c-text-1); text-decoration: none; }
.km-list a:hover { color: var(--vp-c-brand-1); }
.muted { color: var(--vp-c-text-2); font-weight: 400; font-size: 12px; }

@media (max-width: 640px) {
  .km { padding: 16px; }
  .km-svg, .km-progress, .km-toggle { display: none; }
  .km-list { display: block; }
}
</style>
