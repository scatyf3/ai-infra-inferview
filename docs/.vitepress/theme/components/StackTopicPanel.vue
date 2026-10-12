<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { data as topics } from '../../data/topics.data'
import type { Topic } from '../../data/topics.data'
import { data as intros } from '../../data/stack.data'
import { domainByDir } from '../../domains'
import { keysOfArticle, locateKey } from '../../layers'

/**
 * 分层图里一个格子的内容：基础介绍 + 相关文章。
 * key 是小主题 id（如 kv-paged、tr-p-pp），或 `layer-N` / `train-layer-N` 表示整层综述。
 * 首页分层图的展开面板和 /stack/<id> 独立页底部都用它。
 */
// 注意：boolean prop 没传时 Vue 会给 false，所以用「隐藏」而不是「显示」做开关
const props = defineProps<{ id: string; compact?: boolean; hideIntro?: boolean }>()

const introById = new Map(intros.map((i) => [i.id, i]))

const intro = computed(() => introById.get(props.id))
const articles = computed(() => topics.filter((t) => !t.isIndex && keysOfArticle(t).includes(props.id)))

// 「also」只列同一张图上的其他格子，推理栈的面板里不混进训练栈的位置，反之亦然
const keyLabel = (key: string) => {
  const loc = locateKey(key)!
  return `${loc.stack.prefix}${loc.layer.id} ${loc.topic?.name ?? '整层'}`
}
const stackOf = (key: string) => locateKey(key)?.stack.id
const elsewhere = (t: Topic) =>
  keysOfArticle(t)
    .filter((k) => k !== props.id && stackOf(k) === stackOf(props.id))
    .map(keyLabel)

const colorOf = (t: Topic) => domainByDir[t.domain]?.color ?? 'var(--st-todo)'
const statusText = { todo: '未写', draft: '草稿', reviewed: '已复习' } as const
</script>

<template>
  <div class="stp" :class="{ compact }">
    <div v-if="!hideIntro && intro" class="stp-intro" v-html="intro.html" />

    <div class="stp-sec">
      <span class="stp-h">知识库文章</span>
      <span class="stp-n">{{ articles.length || '暂无' }}</span>
    </div>
    <ul v-if="articles.length" class="stp-list">
      <li v-for="a in articles" :key="a.url">
        <a :href="withBase(a.url)" :style="{ '--dc': colorOf(a) }">
          <i class="stp-dot" :class="a.status" :title="statusText[a.status]" />
          <span class="stp-title">{{ a.title }}</span>
          <span v-if="elsewhere(a).length" class="stp-also">also {{ elsewhere(a).join(' · ') }}</span>
        </a>
      </li>
    </ul>
    <p v-else class="stp-empty">
      还没有深入的文章。写好后在文章 frontmatter 的 <code>stack</code> 里加上 <code>{{ id }}</code> 就会出现在这里。
    </p>
  </div>
</template>

<style scoped>
.stp { font-size: 13.5px; line-height: 1.7; color: var(--vp-c-text-1); }

/* 介绍正文：v-html 进来的内容，用 :deep 套样式 */
.stp-intro { max-width: 860px; }
.stp-intro :deep(p) { margin: 0 0 8px; }
.stp-intro :deep(ul) { margin: 0 0 8px; padding-left: 18px; }
.stp-intro :deep(li) { margin: 2px 0; }
.stp-intro :deep(li::marker) { color: var(--vp-c-text-3); }
.stp-intro :deep(strong) { font-weight: 600; }
.stp-intro :deep(code) { padding: 1px 4px; border-radius: 3px; background: var(--vp-c-default-soft); font-size: 0.9em; }
.stp-intro :deep(a) { color: var(--vp-c-brand-1); text-decoration: none; }
.stp-intro :deep(a:hover) { text-decoration: underline; }
.stp-intro :deep(p:last-child) { color: var(--vp-c-text-2); font-size: 12.5px; }
.compact .stp-intro { font-size: 13px; }

.stp-sec { display: flex; align-items: baseline; gap: 8px; margin: 10px 0 4px; padding-top: 8px; border-top: 1px dashed var(--wg-border); }
.stp-h { font-size: 12px; font-weight: 600; letter-spacing: 0.04em; color: var(--vp-c-text-2); }
.stp-n { font: 11px var(--vp-font-family-mono); color: var(--vp-c-text-3); }

.stp-list { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 2px 16px; }
.stp-list a { display: flex; align-items: baseline; gap: 7px; padding: 3px 6px; border-left: 2px solid var(--dc); border-radius: 0 2px 2px 0; font-size: 13px; color: var(--vp-c-text-1); text-decoration: none; }
.stp-list a:hover { background: color-mix(in srgb, var(--dc) 10%, transparent); }
.stp-title { flex: none; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.stp-also { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; font-style: italic; color: var(--vp-c-text-3); }
.stp-dot { flex: none; align-self: center; width: 7px; height: 7px; border-radius: 50%; }
.stp-dot.todo { border: 1.5px solid var(--st-todo); }
.stp-dot.draft { background: var(--st-draft); }
.stp-dot.reviewed { background: var(--st-reviewed); }
.stp-empty { margin: 0; font-size: 12.5px; color: var(--vp-c-text-2); }
.stp-empty code { font-size: 11.5px; }

@media (max-width: 768px) {
  .stp-list { grid-template-columns: 1fr; }
  .stp-also { display: none; }
}
</style>
