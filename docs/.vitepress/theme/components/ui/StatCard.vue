<script setup lang="ts">
defineProps<{ label: string; value: string; sub?: string; tone?: 'default' | 'good' | 'warn' | 'bad'; color?: string }>()
</script>

<template>
  <div class="sc" :class="tone ?? 'default'" :style="color ? { borderLeftColor: color } : {}">
    <div class="sc-label">{{ label }}</div>
    <div class="sc-value">{{ value }}</div>
    <div v-if="sub" class="sc-sub">{{ sub }}</div>
  </div>
</template>

<style scoped>
.sc { border: 1px solid var(--wg-border); border-left: 4px solid var(--vp-c-brand-1); border-radius: 8px; padding: 8px 12px; background: var(--vp-c-bg); }
.sc.good { border-left-color: #22c55e; }
.sc.warn { border-left-color: #f59e0b; }
.sc.bad { border-left-color: #ef4444; }
.sc-label { font-size: 12px; color: var(--wg-muted); }
.sc-value { font-size: 18px; font-weight: 600; font-family: var(--vp-font-family-mono); margin-top: 2px; }
.sc-sub { font-size: 12px; color: var(--wg-muted); margin-top: 2px; }
</style>
