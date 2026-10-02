<script setup lang="ts">
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import { data as topics } from '../../data/topics.data'
import { data as intros } from '../../data/stack.data'
import { layerOfTopic, layers } from '../../layers'
import type { Layer, StackTopic } from '../../layers'
import StackTopicPanel from './StackTopicPanel.vue'

// 首页的紧凑版分层图（论文插图风格）：点左侧层名展开这一层的总述，点格子展开小主题的介绍；都附相关文章

// key：小主题 id，或 `layer-N` 表示第 N 层的整层综述
const counts = new Map<string, number>()
for (const t of topics) {
  if (t.isIndex) continue
  for (const s of t.stack) {
    const key = typeof s === 'number' ? `layer-${s}` : s
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
}
const countOf = (key: string) => counts.get(key) ?? 0
const introUrl = new Map(intros.map((i) => [i.id, i.url]))

function groupTopics(l: Layer) {
  const groups: { name?: string; topics: StackTopic[] }[] = []
  for (const t of l.topics) {
    const last = groups[groups.length - 1]
    if (last && last.name === t.group) last.topics.push(t)
    else groups.push({ name: t.group, topics: [t] })
  }
  return groups
}
const rows = [...layers].reverse().map((l) => ({ ...l, groups: groupTopics(l) }))

const topicName = new Map(layers.flatMap((l) => l.topics.map((t) => [t.id, t.name] as const)))
const keyName = (key: string) => (key.startsWith('layer-') ? '本层总述' : topicName.get(key) ?? key)

const open = ref<string | null>(null)
const openLayer = computed(() => {
  const k = open.value
  if (!k) return null
  return k.startsWith('layer-') ? Number(k.slice(6)) : layerOfTopic.get(k) ?? null
})
const toggle = (key: string) => (open.value = open.value === key ? null : key)
</script>

<template>
  <figure class="sf">
    <div class="sf-fig">
      <div class="sf-rows">
        <div v-for="l in rows" :key="l.id" class="sf-row" :class="{ active: openLayer === l.id }" :style="{ '--lc': l.color }">
          <button
            class="sf-label"
            :class="{ open: open === `layer-${l.id}` }"
            :aria-expanded="open === `layer-${l.id}`"
            :title="`${l.name}：本层总述`"
            @click="toggle(`layer-${l.id}`)"
          >
            <span class="sf-id">{{ l.id }}</span>
            <span class="sf-label-text">
              <span class="sf-en">{{ l.en }}</span>
              <span class="sf-zh">{{ l.name }}</span>
            </span>
            <span class="sf-more" aria-hidden="true">›</span>
          </button>
          <div class="sf-boxes">
            <template v-for="(g, gi) in l.groups" :key="gi">
              <span v-if="gi > 0" class="sf-sep" aria-hidden="true" />
              <span v-if="g.name" class="sf-gname">{{ g.name }}</span>
              <button
                v-for="t in g.topics"
                :key="t.id"
                class="sf-box"
                :class="{ ph: !countOf(t.id), open: open === t.id }"
                :aria-expanded="open === t.id"
                @click="toggle(t.id)"
              >{{ t.name }}<sup v-if="countOf(t.id)">{{ countOf(t.id) }}</sup></button>
            </template>
          </div>

          <div v-if="openLayer === l.id && open" class="sf-panel">
            <div class="sf-panel-head">
              <b>L{{ l.id }} · {{ keyName(open) }}</b>
              <span class="sf-panel-actions">
                <a v-if="introUrl.get(open)" :href="withBase(introUrl.get(open)!)" class="sf-page">单独页面 ↗</a>
                <button class="sf-close" aria-label="收起" @click="toggle(open)">✕</button>
              </span>
            </div>
            <StackTopicPanel :id="open" compact />
          </div>
        </div>
      </div>

      <div class="sf-rail" aria-hidden="true">
        <div class="sf-arrow down"><span>request</span></div>
        <div class="sf-arrow up"><span>response</span></div>
      </div>
    </div>

  </figure>
</template>

<style scoped>
.sf {
  --sf-line: color-mix(in srgb, var(--vp-c-text-1) 22%, transparent);
  max-width: 1152px;
  margin: 40px auto 48px;
  padding: 0 24px;
  font-family: Inter, 'Helvetica Neue', Arial, var(--vp-font-family-base);
}
.sf-fig {
  display: grid;
  grid-template-columns: 1fr 64px;
  gap: 0 10px;
  padding: 16px;
  border: 1px solid var(--sf-line);
  border-radius: 4px;
  background: var(--vp-c-bg);
}
.sf-rows { display: flex; flex-direction: column; gap: 4px; min-width: 0; }

.sf-row {
  display: grid;
  grid-template-columns: 196px 1fr;
  align-items: center;
  border: 1px solid color-mix(in srgb, var(--lc) 35%, transparent);
  border-radius: 3px;
  background: color-mix(in srgb, var(--lc) 7%, var(--vp-c-bg));
  transition: border-color 0.12s;
}
.sf-row.active { border-color: color-mix(in srgb, var(--lc) 75%, transparent); }
.sf-label {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 5px 8px 5px 10px;
  align-self: stretch;
  border-right: 1px solid color-mix(in srgb, var(--lc) 35%, transparent);
  text-align: left;
  cursor: pointer;
  transition: background 0.12s;
}
.sf-label:hover, .sf-label.open { background: color-mix(in srgb, var(--lc) 16%, var(--vp-c-bg)); }
.sf-more { margin-left: auto; font-size: 16px; line-height: 1; color: var(--vp-c-text-3); transition: transform 0.15s; }
.sf-label:hover .sf-more { color: var(--vp-c-text-1); }
.sf-label.open .sf-more { transform: rotate(90deg); color: var(--vp-c-text-1); }
.sf-id {
  flex: none;
  display: grid;
  place-items: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  border: 1.5px solid color-mix(in srgb, var(--lc) 75%, var(--vp-c-text-1));
  font: 700 11px/1 var(--vp-font-family-mono);
  color: color-mix(in srgb, var(--lc) 70%, var(--vp-c-text-1));
}
.sf-label-text { display: flex; flex-direction: column; min-width: 0; line-height: 1.25; }
.sf-en { font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--vp-c-text-1); white-space: nowrap; }
.sf-zh { font-size: 11px; color: var(--vp-c-text-2); white-space: nowrap; }

.sf-boxes { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; padding: 5px 6px; min-width: 0; }
.sf-box {
  flex: 1 1 auto;
  position: relative;
  padding: 4px 14px 4px 9px;
  border: 1px solid color-mix(in srgb, var(--lc) 60%, var(--vp-c-text-1));
  border-radius: 2px;
  background: var(--vp-c-bg);
  font-size: 12px;
  line-height: 1.4;
  text-align: center;
  white-space: nowrap;
  color: var(--vp-c-text-1);
  cursor: pointer;
  transition: background 0.12s;
}
.sf-box:hover, .sf-box.open { background: color-mix(in srgb, var(--lc) 18%, var(--vp-c-bg)); }
.sf-box.open { box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--lc) 60%, var(--vp-c-text-1)); }
.sf-box sup {
  position: absolute;
  top: 1px;
  right: 3px;
  font: 600 9px/1 var(--vp-font-family-mono);
  color: var(--vp-c-text-2);
}
.sf-box.ph {
  border-style: dashed;
  border-color: var(--sf-line);
  color: var(--vp-c-text-3);
  background: repeating-linear-gradient(
    135deg,
    transparent 0 4px,
    color-mix(in srgb, var(--vp-c-text-3) 16%, transparent) 4px 5px
  );
}
.sf-box.ph:hover, .sf-box.ph.open { color: var(--vp-c-text-1); }
.sf-sep { align-self: stretch; width: 0; margin: 0 4px; border-left: 1px dashed color-mix(in srgb, var(--lc) 55%, transparent); }
.sf-gname { font-size: 10.5px; font-style: italic; color: var(--vp-c-text-2); }

/* 展开面板：贴在那一层下面 */
.sf-panel { grid-column: 1 / -1; padding: 10px 14px 12px; border-top: 1px dashed color-mix(in srgb, var(--lc) 55%, transparent); background: var(--vp-c-bg); }
.sf-panel-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-size: 13.5px; }
.sf-panel-actions { display: inline-flex; align-items: center; gap: 12px; }
.sf-page { font-size: 12px; color: var(--vp-c-brand-1); text-decoration: none; }
.sf-page:hover { text-decoration: underline; }
.sf-close { padding: 0 4px; font-size: 12px; color: var(--vp-c-text-3); cursor: pointer; }
.sf-close:hover { color: var(--vp-c-text-1); }

/* 右侧请求 / 响应箭头 */
.sf-rail { display: flex; justify-content: space-around; padding: 4px 0; }
.sf-arrow { position: relative; display: flex; justify-content: center; width: 24px; }
.sf-arrow::before { content: ''; position: absolute; top: 0; bottom: 0; left: 50%; border-left: 1.5px solid var(--vp-c-text-2); }
.sf-arrow::after { content: ''; position: absolute; left: calc(50% - 4.25px); border: 5px solid transparent; }
.sf-arrow.down::after { bottom: -6px; border-top: 7px solid var(--vp-c-text-2); }
.sf-arrow.up::after { top: -6px; border-bottom: 7px solid var(--vp-c-text-2); }
.sf-arrow span {
  position: relative;
  align-self: center;
  padding: 6px 0;
  writing-mode: vertical-rl;
  font-size: 11px;
  font-style: italic;
  letter-spacing: 0.04em;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg);
}
.sf-arrow.up span { transform: rotate(180deg); }


@media (max-width: 768px) {
  .sf { padding: 0 16px; margin-top: 24px; }
  .sf-fig { grid-template-columns: 1fr; padding: 10px; }
  .sf-rail { display: none; }
  .sf-row { grid-template-columns: 1fr; }
  .sf-label { border-right: none; border-bottom: 1px solid color-mix(in srgb, var(--lc) 35%, transparent); }
  .sf-box { white-space: normal; }
}
</style>
