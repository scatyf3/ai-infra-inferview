<script setup lang="ts">
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import {
  FILTER_GROUPS,
  GROUP_LABEL,
  eras,
  filterByGroup,
  groupByEra,
  groupOfTag,
  milestones,
  releases,
  type FilterGroup,
  type Release,
  type ReleaseTag,
} from '@lib/releases'
import { renderMarks } from '@lib/annotate'
import StatCard from './ui/StatCard.vue'

const props = withDefaults(
  defineProps<{
    view?: 'timeline' | 'table'
    group?: FilterGroup | 'all'
    showEras?: boolean
    onlyMilestones?: boolean
    initialOpen?: string
  }>(),
  { view: 'timeline', group: 'all', showEras: true, onlyMilestones: false, initialOpen: undefined },
)

const mode = ref<'timeline' | 'table'>(props.view)
const activeGroup = ref<FilterGroup | 'all'>(props.group)
const milestoneOnly = ref(props.onlyMilestones)
const open = ref<string | null>(props.initialOpen ?? null)

// 与 domains.ts 的领域色板对齐，和全站领域色一致
const GROUP_COLOR: Record<FilterGroup, string> = {
  memory: '#3b82f6',
  parallel: '#8b5cf6',
  kernel: '#22c55e',
  architecture: '#f97316',
  scheduling: '#eab308',
  ecosystem: '#64748b',
}

const TAG_LABEL: Record<ReleaseTag, string> = {
  memory: 'memory',
  scheduling: 'scheduling',
  kernel: 'kernel',
  parallel: 'parallel',
  architecture: 'architecture',
  quantization: 'quant',
  models: 'models',
  serving: 'serving',
  hardware: 'hardware',
}

const visible = computed(() => {
  const byGroup = filterByGroup(releases, activeGroup.value)
  return milestoneOnly.value ? milestones(byGroup) : byGroup
})
const grouped = computed(() => groupByEra(visible.value, eras))
const totalMilestones = computed(() => milestones(releases).length)
const span = computed(() => {
  const first = releases[0]?.date.slice(0, 7) ?? ''
  const last = releases[releases.length - 1]?.date.slice(0, 7) ?? ''
  return `${first} → ${last}`
})

const tagColor = (t: ReleaseTag) => GROUP_COLOR[groupOfTag(t)]
const tagStyle = (t: ReleaseTag) => ({
  color: tagColor(t),
  borderColor: tagColor(t),
  background: `color-mix(in srgb, ${tagColor(t)} 14%, var(--vp-c-bg))`,
})
const toggle = (r: Release) => { open.value = open.value === r.version ? null : r.version }
const isOpen = (r: Release) => open.value === r.version
const hostOf = (url: string) => {
  const m = url.match(/^https?:\/\/([^/]+)\/(.*)$/)
  if (!m) return url
  const tag = m[2].match(/releases\/tag\/(v[\d.]+)/)
  return tag ? `release ${tag[1]}` : m[1].replace(/^www\./, '')
}
</script>

<template>
  <div class="widget rt">
    <h4>vLLM 版本演进：{{ releases.length }} 个 minor 版本 · {{ totalMilestones }} 个里程碑</h4>

    <div class="row controls">
      <div class="toggle-group">
        <button :class="{ active: activeGroup === 'all' }" @click="activeGroup = 'all'">全部</button>
        <button
          v-for="g in FILTER_GROUPS"
          :key="g"
          :class="{ active: activeGroup === g }"
          :style="activeGroup === g ? { background: GROUP_COLOR[g], color: '#fff' } : {}"
          @click="activeGroup = g"
        >{{ GROUP_LABEL[g] }}</button>
      </div>
      <div class="toggle-group">
        <button :class="{ active: mode === 'timeline' }" @click="mode = 'timeline'">时间线</button>
        <button :class="{ active: mode === 'table' }" @click="mode = 'table'">表格</button>
      </div>
      <label class="chk"><input v-model="milestoneOnly" type="checkbox" /> 仅里程碑</label>
    </div>

    <div class="grid stats">
      <StatCard label="minor 版本" :value="String(releases.length)" :sub="span" />
      <StatCard label="里程碑" :value="String(totalMilestones)" sub="带 observation / 场景 / 做法" tone="good" />
      <StatCard label="阶段" :value="String(eras.length)" sub="按瓶颈转移划分" />
      <StatCard label="当前筛选" :value="String(visible.length)" :sub="activeGroup === 'all' ? '未筛选' : GROUP_LABEL[activeGroup]" :tone="visible.length ? 'default' : 'warn'" />
    </div>

    <p v-if="!visible.length" class="muted">这个筛选下没有版本。</p>

    <!-- 时间线视图 -->
    <template v-else-if="mode === 'timeline'">
      <section v-for="g in grouped" :key="g.era.id" class="era">
        <header v-if="showEras" class="era-head">
          <div class="era-label">{{ g.era.label }}</div>
          <div class="muted era-sum">{{ g.era.summary }}</div>
        </header>
        <ol class="tl">
          <li v-for="r in g.releases" :key="r.version" :class="{ milestone: r.milestone, open: isOpen(r) }">
            <button class="tl-head" :aria-expanded="isOpen(r)" @click="toggle(r)">
              <i class="dot" />
              <span class="mono ver">{{ r.version }}</span>
              <span class="muted date">{{ r.date }}</span>
              <span class="head" v-html="renderMarks(r.headline)" />
              <span class="tags">
                <span v-for="t in r.tags" :key="t" class="tag" :style="tagStyle(t)">{{ TAG_LABEL[t] }}</span>
              </span>
            </button>
            <div v-if="isOpen(r)" class="tl-body">
              <template v-if="r.milestone">
                <div class="kv"><b>Observation</b><span v-html="renderMarks(r.observation ?? '')" /></div>
                <div class="kv"><b>场景</b><span v-html="renderMarks(r.scenario ?? '')" /></div>
                <div class="kv"><b>做法</b><span v-html="renderMarks(r.how ?? '')" /></div>
                <div v-if="r.related?.length" class="kv">
                  <b>关联页面</b>
                  <span class="links"><a v-for="p in r.related" :key="p" :href="withBase(p)">{{ p }}</a></span>
                </div>
              </template>
              <div v-if="r.sources?.length" class="kv">
                <b>来源</b>
                <span class="links">
                  <a v-for="s in r.sources" :key="s" :href="s" target="_blank" rel="noopener">{{ hostOf(s) }}</a>
                </span>
              </div>
            </div>
          </li>
        </ol>
      </section>
    </template>

    <!-- 表格视图 -->
    <div v-else class="tbl-wrap">
      <table class="tbl">
        <thead>
          <tr><th>版本</th><th>日期</th><th>加了什么</th><th>类别</th></tr>
        </thead>
        <tbody>
          <template v-for="r in visible" :key="r.version">
            <tr class="trow" :class="{ milestone: r.milestone, open: isOpen(r) }" @click="toggle(r)">
              <td class="mono ver"><span v-if="r.milestone" class="star">★</span>{{ r.version }}</td>
              <td class="mono date">{{ r.date }}</td>
              <td class="head" v-html="renderMarks(r.headline)" />
              <td class="tags"><span v-for="t in r.tags" :key="t" class="tag" :style="tagStyle(t)">{{ TAG_LABEL[t] }}</span></td>
            </tr>
            <tr v-if="isOpen(r)" class="detail">
              <td colspan="4">
                <div class="tl-body">
                  <template v-if="r.milestone">
                    <div class="kv"><b>Observation</b><span v-html="renderMarks(r.observation ?? '')" /></div>
                    <div class="kv"><b>场景</b><span v-html="renderMarks(r.scenario ?? '')" /></div>
                    <div class="kv"><b>做法</b><span v-html="renderMarks(r.how ?? '')" /></div>
                    <div v-if="r.related?.length" class="kv">
                      <b>关联页面</b>
                      <span class="links"><a v-for="p in r.related" :key="p" :href="withBase(p)">{{ p }}</a></span>
                    </div>
                  </template>
                  <div v-if="r.sources?.length" class="kv">
                    <b>来源</b>
                    <span class="links">
                      <a v-for="s in r.sources" :key="s" :href="s" target="_blank" rel="noopener">{{ hostOf(s) }}</a>
                    </span>
                  </div>
                </div>
              </td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>

    <p class="muted note">
      大圆点（表格里带 ★）是里程碑，点开看这一版观察到了什么瓶颈、优化谁、怎么做；小圆点是功能累积版本，只给一行摘要和 release note 链接。
      按类别筛选时看的是「这条线上的瓶颈怎么一步步转移」，比如选「调度」能看到 continuous batching → chunked prefill → multi-step → token-budget 统一调度 → async scheduling。
    </p>
  </div>
</template>

<style scoped>
.controls { gap: 8px 14px; margin-bottom: 12px; align-items: center; }
.chk { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; cursor: pointer; }
.stats { margin-bottom: 14px; }

.era { margin-top: 6px; }
.era-head { margin: 14px 0 6px; padding-left: 2px; }
.era-label { font-weight: 600; font-size: 14px; }
.era-sum { font-size: 12.5px; line-height: 1.5; }

.tl { list-style: none; margin: 0; padding: 0 0 0 14px; border-left: 2px solid var(--wg-border); }
.tl li { position: relative; margin: 0; padding: 2px 0; }
.tl-head {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 10px;
  width: 100%; text-align: left; padding: 6px 8px; border: none; border-radius: 6px;
  background: transparent; color: var(--wg-text); cursor: pointer; font-size: 13.5px; line-height: 1.45;
}
.tl-head:hover, .tl li.open .tl-head { background: var(--vp-c-bg); }
.dot {
  position: absolute; left: -20px; top: 13px; width: 8px; height: 8px; border-radius: 50%;
  background: var(--wg-muted); border: 2px solid var(--wg-bg);
}
.tl li.milestone .dot { left: -22px; top: 11px; width: 12px; height: 12px; background: var(--vp-c-brand-1); }
.ver { font-weight: 600; min-width: 46px; }
.tl li.milestone .ver { color: var(--vp-c-brand-1); }
.date { font-size: 12px; font-family: var(--vp-font-family-mono); }
.head { flex: 1 1 260px; }
.tl li.milestone .head { font-weight: 600; }
.tags { display: inline-flex; flex-wrap: wrap; gap: 4px; }
.tag { font-size: 10.5px; line-height: 1; padding: 3px 6px; border-radius: 999px; border: 1px solid; font-family: var(--vp-font-family-mono); white-space: nowrap; }

.tl-body { padding: 6px 8px 10px 8px; display: grid; gap: 6px; font-size: 13px; line-height: 1.6; }
.kv { display: grid; grid-template-columns: 84px 1fr; gap: 8px; }
.kv b { color: var(--wg-muted); font-weight: 600; font-size: 12px; padding-top: 2px; }
.links { display: flex; flex-wrap: wrap; gap: 4px 12px; }
.links a { font-family: var(--vp-font-family-mono); font-size: 12.5px; }

.tbl-wrap { overflow-x: auto; }
.tbl { width: 100%; border-collapse: collapse; }
.tbl th, .tbl td { padding: 6px 8px; vertical-align: top; }
.tbl th { text-align: left; font-size: 12px; color: var(--wg-muted); white-space: nowrap; }
.trow { cursor: pointer; }
.trow:hover, .trow.open { background: var(--vp-c-bg); }
.trow.milestone .head, .trow.milestone .ver { font-weight: 600; }
.trow.milestone .ver { color: var(--vp-c-brand-1); }
.star { font-size: 10px; margin-right: 3px; vertical-align: 1px; }
.tbl .date { white-space: nowrap; }
.tbl .head { min-width: 260px; }
.detail td { padding: 0 8px 8px 8px; background: var(--vp-c-bg); }

.note { margin: 14px 0 0; font-size: 12.5px; line-height: 1.6; }

@media (max-width: 560px) {
  .kv { grid-template-columns: 1fr; gap: 2px; }
  .tl-head { font-size: 13px; }
}
</style>
