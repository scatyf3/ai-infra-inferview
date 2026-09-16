<script setup lang="ts">
defineProps<{ label: string; min?: number; max?: number; step?: number; hint?: string; width?: string }>()
const model = defineModel<number>({ required: true })
</script>

<template>
  <label class="nf" :style="{ width: width ?? '120px' }">
    <span class="nf-label">{{ label }}<span v-if="hint" class="nf-hint" :title="hint">?</span></span>
    <input type="number" v-model.number="model" :min="min" :max="max" :step="step ?? 1" />
  </label>
</template>

<style scoped>
.nf { display: flex; flex-direction: column; gap: 3px; }
.nf-label { font-size: 12px; color: var(--wg-muted); display: flex; gap: 4px; align-items: center; }
.nf-hint { display: inline-flex; width: 14px; height: 14px; border-radius: 50%; border: 1px solid var(--wg-border); font-size: 10px; align-items: center; justify-content: center; cursor: help; }
input { width: 100%; padding: 4px 8px; border: 1px solid var(--wg-border); border-radius: 6px; background: var(--vp-c-bg); color: var(--wg-text); font-family: var(--vp-font-family-mono); font-size: 13px; }
input:focus { outline: none; border-color: var(--vp-c-brand-1); }
</style>
