<script setup lang="ts">
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import { data as topics } from '../../data/topics.data'
import type { Topic } from '../../data/topics.data'
import { domainByDir } from '../../domains'
import { layerOfTopic, layers } from '../../layers'
import { stages } from '../../roadmap'

// 首页学习路线：上面一条阶段轨道，点阶段在下面展开它的目标、小主题、文章和过关标准

const topicByUrl = new Map(topics.map((t) => [t.url, t]))
const topicName = new Map(layers.flatMap((l) => l.topics.map((t) => [t.id, t.name] as const)))

const view = stages.map((s) => {
  const arts = s.articles.map((u) => topicByUrl.get(u)).filter((t): t is Topic => !!t)
  const touched = new Set(s.topics.map((id) => layerOfTopic.get(id)!))
  // 小主题按层从下往上排，和分层图的阅读方向一致
  const subs = [...s.topics].sort((a, b) => layerOfTopic.get(a)! - layerOfTopic.get(b)!)
  return {
    ...s,
    arts,
    subs,
    touched,
    reviewed: arts.filter((t) => t.status === 'reviewed').length,
    layerTip: [...touched].sort((a, b) => a - b).map((n) => `L${n} ${layers[n].name}`).join('\n'),
  }
})

const sel = ref(1)
const cur = computed(() => view.find((s) => s.id === sel.value)!)

const layerColor = (id: string) => layers[layerOfTopic.get(id)!].color
const colorOf = (t: Topic) => domainByDir[t.domain]?.color ?? 'var(--st-todo)'
const statusText = { todo: '未写', draft: '草稿', reviewed: '已复习' } as const
const strip = [...layers].reverse()
</script>

<template>
  <section class="rm">
    <header class="rm-head">
      <h2>学习路线</h2>
      <span>从一次前向开始，逐层往外扩</span>
    </header>

    <div class="rm-frame">
      <ol class="rm-track">
        <li v-for="s in view" :key="s.id">
          <button class="rm-node" :class="{ on: sel === s.id }" :aria-pressed="sel === s.id" @click="sel = s.id">
            <span class="rm-dot">{{ s.id }}</span>
            <span class="rm-stage">Stage {{ s.id }}</span>
            <span class="rm-name">{{ s.title }}</span>
            <span class="rm-subt">{{ s.subtitle }}</span>
            <span class="rm-meta">
              <span class="rm-strip" :title="s.layerTip">
                <i v-for="l in strip" :key="l.id" :style="s.touched.has(l.id) ? { background: l.color } : {}" />
              </span>
              <span class="rm-count">{{ s.topics.length }} 个小主题<br />{{ s.reviewed }}/{{ s.arts.length }} 篇已复习</span>
            </span>
          </button>
        </li>
      </ol>

      <div class="rm-detail">
        <p class="rm-goal"><b>Stage {{ cur.id }} · {{ cur.title }}</b>{{ cur.goal }}</p>
        <div class="rm-cols">
          <div>
            <div class="rm-h">小主题</div>
            <div class="rm-chips">
              <a
                v-for="id in cur.subs"
                :key="id"
                :href="withBase(`/stack/${id}`)"
                class="rm-chip"
                :style="{ '--lc': layerColor(id) }"
              ><em>L{{ layerOfTopic.get(id) }}</em>{{ topicName.get(id) }}</a>
            </div>
          </div>
          <div>
            <div class="rm-h">读 / 手撕</div>
            <ul class="rm-arts">
              <li v-for="a in cur.arts" :key="a.url">
                <a :href="withBase(a.url)" :style="{ '--dc': colorOf(a) }">
                  <i class="rm-st" :class="a.status" :title="statusText[a.status]" />{{ a.title }}
                </a>
              </li>
            </ul>
          </div>
          <div>
            <div class="rm-h">过关标准</div>
            <ul class="rm-checks">
              <li v-for="c in cur.checks" :key="c">{{ c }}</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.rm {
  --rm-line: color-mix(in srgb, var(--vp-c-text-1) 22%, transparent);
  max-width: 1152px;
  margin: 0 auto 64px;
  padding: 0 24px;
  font-family: Inter, 'Helvetica Neue', Arial, var(--vp-font-family-base);
}
.rm-head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 10px; }
.rm-head h2 { margin: 0; font-size: 18px; font-weight: 700; color: var(--vp-c-text-1); }
.rm-head span { font-size: 13px; color: var(--vp-c-text-2); }
.rm-frame { border: 1px solid var(--rm-line); border-radius: 4px; background: var(--vp-c-bg); }

/* ---- 阶段轨道 ---- */
.rm-track { position: relative; display: grid; grid-template-columns: repeat(5, 1fr); margin: 0; padding: 18px 8px 12px; list-style: none; }
.rm-track::before {
  content: '';
  position: absolute;
  top: 31px;
  left: 10%;
  right: 10%;
  border-top: 1.5px solid var(--vp-c-text-3);
}
.rm-track::after {
  content: '';
  position: absolute;
  top: 27px;
  right: calc(10% - 9px);
  border: 4.5px solid transparent;
  border-left: 7px solid var(--vp-c-text-3);
}
.rm-track li { display: flex; justify-content: center; min-width: 0; }
.rm-node {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  width: 100%;
  padding: 0 6px 8px;
  border-radius: 4px;
  text-align: center;
  color: var(--vp-c-text-1);
  cursor: pointer;
  transition: background 0.12s;
}
.rm-node:hover { background: var(--vp-c-default-soft); }
.rm-node.on { background: color-mix(in srgb, var(--vp-c-brand-1) 8%, var(--vp-c-bg)); }
.rm-dot {
  position: relative;
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  margin-bottom: 6px;
  border: 1.5px solid var(--vp-c-text-2);
  border-radius: 50%;
  background: var(--vp-c-bg);
  font: 700 13px/1 var(--vp-font-family-mono);
}
.rm-node.on .rm-dot { border-color: var(--vp-c-brand-1); background: var(--vp-c-brand-1); color: #fff; }
.rm-stage { font-size: 10.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--vp-c-text-3); }
.rm-name { font-size: 14px; font-weight: 600; line-height: 1.35; }
.rm-subt { font-size: 11.5px; color: var(--vp-c-text-2); }
.rm-meta { display: flex; align-items: center; gap: 8px; margin-top: 6px; }
.rm-strip { display: flex; flex-direction: column; gap: 1px; width: 22px; }
.rm-strip i { height: 3px; border-radius: 1px; background: color-mix(in srgb, var(--vp-c-text-3) 22%, transparent); }
.rm-count { font: 10.5px/1.35 var(--vp-font-family-mono); color: var(--vp-c-text-3); text-align: left; }

/* ---- 阶段详情 ---- */
.rm-detail { padding: 14px 18px 16px; border-top: 1px dashed var(--rm-line); }
.rm-goal { margin: 0 0 12px; font-size: 13.5px; line-height: 1.7; color: var(--vp-c-text-2); }
.rm-goal b { margin-right: 10px; color: var(--vp-c-text-1); }
.rm-cols { display: grid; grid-template-columns: 1fr 1.3fr 1.2fr; gap: 20px; }
.rm-h { margin-bottom: 6px; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; color: var(--vp-c-text-3); }

.rm-chips { display: flex; flex-wrap: wrap; gap: 5px; }
.rm-chip {
  display: inline-flex;
  align-items: baseline;
  gap: 5px;
  padding: 2px 8px;
  border: 1px solid color-mix(in srgb, var(--lc) 60%, var(--vp-c-text-1));
  border-radius: 2px;
  background: color-mix(in srgb, var(--lc) 7%, var(--vp-c-bg));
  font-size: 12px;
  color: var(--vp-c-text-1);
  text-decoration: none;
}
.rm-chip:hover { background: color-mix(in srgb, var(--lc) 18%, var(--vp-c-bg)); }
.rm-chip em { font: normal 600 10px var(--vp-font-family-mono); color: var(--vp-c-text-3); }

.rm-arts, .rm-checks { margin: 0; padding: 0; list-style: none; }
.rm-arts a {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 2px 6px;
  border-left: 2px solid var(--dc);
  font-size: 12.5px;
  line-height: 1.6;
  color: var(--vp-c-text-1);
  text-decoration: none;
}
.rm-arts a:hover { background: color-mix(in srgb, var(--dc) 10%, transparent); }
.rm-st { flex: none; width: 7px; height: 7px; border-radius: 50%; }
.rm-st.todo { border: 1.5px solid var(--st-todo); }
.rm-st.draft { background: var(--st-draft); }
.rm-st.reviewed { background: var(--st-reviewed); }
.rm-checks li { position: relative; padding: 2px 0 2px 20px; font-size: 12.5px; line-height: 1.6; color: var(--vp-c-text-1); }
.rm-checks li::before {
  content: '';
  position: absolute;
  left: 2px;
  top: 7px;
  width: 9px;
  height: 9px;
  border: 1.5px solid var(--vp-c-text-3);
  border-radius: 2px;
}

@media (max-width: 860px) {
  .rm-cols { grid-template-columns: 1fr; gap: 14px; }
}
@media (max-width: 768px) {
  .rm { padding: 0 16px; }
  .rm-track { grid-template-columns: 1fr; gap: 4px; padding: 10px; }
  .rm-track::before, .rm-track::after { display: none; }
  .rm-node { flex-direction: row; flex-wrap: wrap; justify-content: flex-start; gap: 4px 8px; padding: 6px; text-align: left; }
  .rm-dot { margin: 0; }
  .rm-meta { display: none; }
}
</style>
