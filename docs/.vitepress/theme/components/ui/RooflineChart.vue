<script setup lang="ts">
import { computed } from 'vue'

interface Pt { label: string; ai: number; color: string }
const props = defineProps<{ peak: number; bw: number; points: Pt[]; peakLabel?: string }>()

const W = 560, H = 300, PL = 64, PR = 20, PT = 20, PB = 40
const xMin = -1, xMax = 4 // log10 AI: 0.1 .. 10^4

const yMax = computed(() => Math.log10(props.peak) + 0.3)
const yMin = computed(() => Math.log10(props.peak) - 4)

const x = (ai: number) => PL + ((Math.log10(ai) - xMin) / (xMax - xMin)) * (W - PL - PR)
const y = (flops: number) => PT + ((yMax.value - Math.log10(flops)) / (yMax.value - yMin.value)) * (H - PT - PB)
const roof = (ai: number) => Math.min(props.peak, props.bw * ai)
const ridge = computed(() => props.peak / props.bw)

const roofPath = computed(() => {
  const a0 = Math.pow(10, xMin), a1 = Math.pow(10, xMax)
  const r = ridge.value
  return `M ${x(a0)} ${y(roof(a0))} L ${x(r)} ${y(roof(r))} L ${x(a1)} ${y(props.peak)}`
})
const xTicks = [-1, 0, 1, 2, 3, 4]
const yTicks = computed(() => {
  const t: number[] = []
  for (let e = Math.ceil(yMin.value); e <= Math.floor(yMax.value); e++) t.push(e)
  return t
})
const fmtPow = (e: number) => (e === 0 ? '1' : e === 1 ? '10' : `10^${e}`)
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="roofline" role="img" aria-label="roofline chart">
    <g class="grid">
      <line v-for="t in xTicks" :key="'x' + t" :x1="x(10 ** t)" :x2="x(10 ** t)" :y1="PT" :y2="H - PB" />
      <line v-for="t in yTicks" :key="'y' + t" :x1="PL" :x2="W - PR" :y1="y(10 ** t)" :y2="y(10 ** t)" />
    </g>
    <path :d="roofPath" class="roof" />
    <line :x1="x(ridge)" :x2="x(ridge)" :y1="PT" :y2="H - PB" class="ridge" />
    <text :x="x(ridge) + 4" :y="H - PB - 6" class="lbl">ridge {{ ridge.toFixed(0) }}</text>
    <text :x="W - PR" :y="y(peak) - 6" text-anchor="end" class="lbl">{{ peakLabel ?? 'peak' }}</text>
    <text :x="PL + 6" :y="PT + 14" class="lbl memory">memory-bound</text>
    <text :x="W - PR - 6" :y="H - PB - 30" text-anchor="end" class="lbl compute">compute-bound</text>
    <g v-for="p in points" :key="p.label">
      <line :x1="x(p.ai)" :x2="x(p.ai)" :y1="y(roof(p.ai))" :y2="H - PB" class="drop" :stroke="p.color" />
      <circle :cx="x(p.ai)" :cy="y(roof(p.ai))" r="6" :fill="p.color" />
      <text :x="x(p.ai) + 9" :y="y(roof(p.ai)) - 8" class="lbl" :fill="p.color">{{ p.label }} (AI≈{{ p.ai < 10 ? p.ai.toFixed(1) : p.ai.toFixed(0) }})</text>
    </g>
    <g class="axis">
      <text v-for="t in xTicks" :key="'xl' + t" :x="x(10 ** t)" :y="H - PB + 16" text-anchor="middle" class="tick">{{ fmtPow(t) }}</text>
      <text v-for="t in yTicks" :key="'yl' + t" :x="PL - 6" :y="y(10 ** t) + 4" text-anchor="end" class="tick">{{ fmtPow(t) }}</text>
      <text :x="(PL + W - PR) / 2" :y="H - 4" text-anchor="middle" class="tick">Arithmetic Intensity (FLOP / Byte)</text>
      <text :x="12" :y="(PT + H - PB) / 2" text-anchor="middle" class="tick" :transform="`rotate(-90 12 ${(PT + H - PB) / 2})`">FLOP/s</text>
    </g>
  </svg>
</template>

<style scoped>
.roofline { width: 100%; height: auto; background: var(--vp-c-bg); border-radius: 8px; border: 1px solid var(--wg-border); }
.grid line { stroke: var(--wg-border); stroke-width: 1; }
.roof { fill: none; stroke: var(--vp-c-brand-1); stroke-width: 2.5; }
.ridge { stroke: var(--wg-muted); stroke-dasharray: 4 4; }
.drop { stroke-dasharray: 3 3; stroke-width: 1; }
.lbl { font-size: 11px; fill: var(--wg-text); }
.lbl.memory { fill: #f59e0b; }
.lbl.compute { fill: #22c55e; }
.tick { font-size: 10px; fill: var(--wg-muted); }
</style>
