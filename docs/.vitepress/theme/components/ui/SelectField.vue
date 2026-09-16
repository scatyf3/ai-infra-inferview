<script setup lang="ts">
defineProps<{ label: string; options: { value: string; label: string }[]; width?: string }>()
const model = defineModel<string>({ required: true })
</script>

<template>
  <label class="sf" :style="{ width: width ?? '160px' }">
    <span class="sf-label">{{ label }}</span>
    <select v-model="model">
      <option v-for="o in options" :key="o.value" :value="o.value">{{ o.label }}</option>
    </select>
  </label>
</template>

<style scoped>
.sf { display: flex; flex-direction: column; gap: 3px; }
.sf-label { font-size: 12px; color: var(--wg-muted); }
select { width: 100%; padding: 4px 8px; border: 1px solid var(--wg-border); border-radius: 6px; background: var(--vp-c-bg); color: var(--wg-text); font-size: 13px; }
select:focus { outline: none; border-color: var(--vp-c-brand-1); }
</style>
