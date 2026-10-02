<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { LAYER_IDS, layerOfTopic, layers } from '../../layers'
import StackTopicPanel from './StackTopicPanel.vue'

// /stack/<id> 独立页的上下文：顶部是「在推理栈的哪一层」，底部是相关的知识库文章；
// 层总述页（/stack/layer-N）底部再列出这一层的所有小主题
const props = defineProps<{ where: 'before' | 'after' }>()
const { page } = useData()

const id = computed(() => {
  const m = page.value.relativePath.match(/^stack\/(.+)\.md$/)
  if (!m) return null
  const lm = m[1].match(/^layer-(\d+)$/)
  if (lm) return LAYER_IDS.has(Number(lm[1])) ? m[1] : null
  return layerOfTopic.has(m[1]) ? m[1] : null
})
const isLayerPage = computed(() => !!id.value?.startsWith('layer-'))
const layer = computed(() => {
  if (!id.value) return null
  const n = isLayerPage.value ? Number(id.value.slice(6)) : layerOfTopic.get(id.value)!
  return layers[n]
})
</script>

<template>
  <div v-if="id && layer && props.where === 'before'" class="spx-crumb" :style="{ '--lc': layer.color }">
    <a :href="withBase('/')">推理栈</a>
    <span>/</span>
    <a v-if="!isLayerPage" :href="withBase(`/stack/layer-${layer.id}`)" class="spx-layer"><i>{{ layer.id }}</i>{{ layer.name }}</a>
    <span v-else class="spx-layer"><i>{{ layer.id }}</i>{{ layer.en }}</span>
  </div>

  <div v-else-if="id && layer && props.where === 'after'" class="spx-after">
    <template v-if="isLayerPage">
      <div class="spx-h">本层小主题</div>
      <div class="spx-subs" :style="{ '--lc': layer.color }">
        <a v-for="t in layer.topics" :key="t.id" :href="withBase(`/stack/${t.id}`)">{{ t.name }}</a>
      </div>
    </template>
    <StackTopicPanel :id="id" hide-intro />
  </div>
</template>

<style scoped>
.spx-crumb { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; font-size: 13px; color: var(--vp-c-text-3); }
.spx-crumb a { color: var(--vp-c-text-2); text-decoration: none; }
.spx-crumb a:hover { color: var(--vp-c-brand-1); }
.spx-layer { display: inline-flex; align-items: center; gap: 6px; color: var(--vp-c-text-2); }
.spx-layer i {
  display: grid;
  place-items: center;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 1.5px solid var(--lc);
  font: normal 700 10px/1 var(--vp-font-family-mono);
  color: var(--vp-c-text-1);
}
.spx-after { margin-top: 32px; }
.spx-h { margin-bottom: 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.04em; color: var(--vp-c-text-2); }
.spx-subs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 4px; }
.spx-subs a {
  padding: 3px 10px;
  border: 1px solid color-mix(in srgb, var(--lc) 60%, var(--vp-c-text-1));
  border-radius: 2px;
  font-size: 13px;
  color: var(--vp-c-text-1);
  text-decoration: none;
}
.spx-subs a:hover { background: color-mix(in srgb, var(--lc) 16%, var(--vp-c-bg)); }
</style>
