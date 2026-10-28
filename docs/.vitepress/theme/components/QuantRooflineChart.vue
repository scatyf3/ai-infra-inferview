<script setup lang="ts">
import { computed, ref } from 'vue'

// H100 SXM 上 K = N = 8192 的线性层，roofline：t = max(权重字节 / 带宽, 2BKN / 峰值)
const K = 8192, N = 8192, BW = 3.35e12
const fmts = [
  { id: 'bf16', label: 'bf16', bytes: 2, peak: 989e12 },
  { id: 'w4a16', label: 'W4A16', bytes: 0.5, peak: 989e12 },
  { id: 'fp8', label: 'FP8 W8A8', bytes: 1, peak: 1979e12 },
]
const memUs = (f: (typeof fmts)[number]) => (f.bytes * K * N) / BW * 1e6
const calcUs = (f: (typeof fmts)[number], b: number) => (2 * b * K * N) / f.peak * 1e6
const tUs = (f: (typeof fmts)[number], b: number) => Math.max(memUs(f), calcUs(f, b))
const knee = (f: (typeof fmts)[number]) => (f.peak / BW) * (f.bytes / 2)

const W = 640, H = 320, PL = 46, PR = 16, PT = 16, PB = 40
const bMin = 1, bMax = 4096, tMin = 5, tMax = 800
const x = (b: number) => PL + (Math.log2(b) / Math.log2(bMax)) * (W - PL - PR)
const y = (t: number) => PT + (1 - (Math.log10(t) - Math.log10(tMin)) / (Math.log10(tMax) - Math.log10(tMin))) * (H - PT - PB)
const xTicks = [1, 4, 16, 64, 256, 1024, 4096]
const yTicks = [10, 20, 50, 100, 200, 500]
const fmtB = (b: number) => (b >= 1024 ? `${b / 1024}K` : `${b}`)

// 每格 2^(1/16) 采样，拐点处再补一个点，折线才是尖角
const samples = (f: (typeof fmts)[number]) => {
  const bs: number[] = []
  for (let i = 0; i <= 16 * Math.log2(bMax); i++) bs.push(2 ** (i / 16))
  bs.push(knee(f))
  return bs.sort((a, b) => a - b)
}
const path = (f: (typeof fmts)[number]) =>
  samples(f).map((b, i) => `${i ? 'L' : 'M'}${x(b).toFixed(1)},${y(tUs(f, b)).toFixed(1)}`).join('')

// 十字线：吸附到整数 B
const hover = ref<number | null>(null)
const svgEl = ref<SVGSVGElement | null>(null)
function onMove(e: PointerEvent) {
  const r = svgEl.value!.getBoundingClientRect()
  const px = ((e.clientX - r.left) / r.width) * W
  const frac = Math.min(1, Math.max(0, (px - PL) / (W - PL - PR)))
  hover.value = Math.round(2 ** (frac * Math.log2(bMax)))
}
const readout = computed(() => {
  const b = hover.value
  if (b === null) return []
  const base = tUs(fmts[0], b)
  return fmts.map((f) => ({ f, t: tUs(f, b), bound: memUs(f) >= calcUs(f, b) ? '读权重' : '算', speedup: base / tUs(f, b) }))
})
const tipLeft = computed(() => (hover.value === null ? 0 : (x(hover.value) / W) * 100))
</script>

<template>
  <div class="widget qr">
    <h4>H100 上一个 8192 × 8192 线性层的耗时（roofline 估算）</h4>
    <div class="legend">
      <span v-for="f in fmts" :key="f.id" class="key">
        <svg width="18" height="10"><line x1="1" y1="5" x2="17" y2="5" :stroke="`var(--qr-${f.id})`" stroke-width="2.5" :stroke-dasharray="f.id === 'w4a16' ? '4 3' : undefined" /></svg>
        {{ f.label }}
      </span>
    </div>
    <div class="plot">
      <svg ref="svgEl" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="三种格式的耗时随 token 数 B 变化的曲线"
        @pointermove="onMove" @pointerleave="hover = null">
        <g class="grid">
          <line v-for="t in yTicks" :key="'y' + t" :x1="PL" :x2="W - PR" :y1="y(t)" :y2="y(t)" />
        </g>
        <g v-for="b in [74, 295]" :key="'k' + b">
          <line :x1="x(b)" :x2="x(b)" :y1="PT" :y2="H - PB" class="knee" />
          <text :x="x(b) + 4" :y="PT + 10" class="lbl">B = {{ b }}</text>
        </g>
        <g class="axis">
          <text v-for="t in yTicks" :key="'yl' + t" :x="PL - 6" :y="y(t) + 4" text-anchor="end">{{ t }}</text>
          <text v-for="t in xTicks" :key="'xl' + t" :x="x(t)" :y="H - PB + 16" text-anchor="middle">{{ fmtB(t) }}</text>
          <text :x="(PL + W - PR) / 2" :y="H - 4" text-anchor="middle">这一步的 token 数 B（对数刻度）</text>
          <text x="12" :y="(PT + H - PB) / 2" text-anchor="middle" :transform="`rotate(-90 12 ${(PT + H - PB) / 2})`">耗时 μs（对数）</text>
        </g>
        <line v-if="hover !== null" :x1="x(hover)" :x2="x(hover)" :y1="PT" :y2="H - PB" class="cross" />
        <g v-for="f in fmts" :key="f.id">
          <path :d="path(f)" class="line" :stroke="`var(--qr-${f.id})`" :stroke-dasharray="f.id === 'w4a16' ? '6 4' : undefined" />
          <text :x="x(bMin) + 6" :y="y(memUs(f)) - 6" class="lbl">{{ f.label }}</text>
          <circle v-if="hover !== null" :cx="x(hover)" :cy="y(tUs(f, hover))" r="4.5" :fill="`var(--qr-${f.id})`" class="dot" />
        </g>
      </svg>
      <div v-if="hover !== null" class="tip" :style="{ left: tipLeft + '%' }" :class="{ flip: tipLeft > 55 }">
        <div class="tip-h">B = {{ hover }}</div>
        <div v-for="r in readout" :key="r.f.id" class="tip-r">
          <span class="sw" :style="{ background: `var(--qr-${r.f.id})` }" />
          <span class="tip-l">{{ r.f.label }}（{{ r.bound }}）</span>
          <b>{{ r.t.toFixed(r.t < 100 ? 1 : 0) }} μs</b>
          <span class="tip-s">{{ r.speedup.toFixed(2) }}×</span>
        </div>
      </div>
    </div>
    <p class="muted small">
      t = max(权重字节 / 3.35 TB/s, 2BKN / 峰值算力)，峰值 bf16 989、FP8 1979 TFLOP/s。理想上限：权重只从 HBM 读一次，激活、输出和 scale 的字节忽略，带宽和算力都跑满。悬停看各格式耗时和相对 bf16 的加速比。
    </p>
  </div>
</template>

<style scoped>
.qr { --qr-bf16: #2a78d6; --qr-w4a16: #eb6834; --qr-fp8: #1baf7a; }
:root.dark .qr { --qr-bf16: #3987e5; --qr-w4a16: #d95926; --qr-fp8: #199e70; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-bottom: 6px; }
.key { display: inline-flex; align-items: center; gap: 5px; font-size: 12.5px; }
.plot { position: relative; }
.plot svg { width: 100%; height: auto; display: block; background: var(--vp-c-bg); border-radius: 8px; border: 1px solid var(--wg-border); touch-action: pan-y; }
.grid line { stroke: var(--wg-border); stroke-width: 1; }
.knee { stroke: var(--wg-muted); stroke-width: 1; stroke-dasharray: 4 4; }
.cross { stroke: var(--wg-muted); stroke-width: 1; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.dot { stroke: var(--vp-c-bg); stroke-width: 2; }
.lbl { font-size: 11px; fill: var(--wg-muted); }
.axis text { font-size: 10.5px; fill: var(--wg-muted); font-variant-numeric: tabular-nums; }
.tip {
  position: absolute; top: 12px; transform: translateX(12px); pointer-events: none;
  background: var(--vp-c-bg-elv); border: 1px solid var(--wg-border); border-radius: 8px;
  padding: 6px 10px; font-size: 12px; box-shadow: var(--vp-shadow-2); min-width: 210px;
}
.tip.flip { transform: translateX(calc(-100% - 12px)); }
.tip-h { font-weight: 600; margin-bottom: 3px; }
.tip-r { display: flex; align-items: center; gap: 6px; line-height: 1.6; }
.tip-l { flex: 1; color: var(--wg-muted); }
.tip-r b { font-variant-numeric: tabular-nums; }
.tip-s { width: 42px; text-align: right; color: var(--wg-muted); font-variant-numeric: tabular-nums; }
.sw { width: 10px; height: 10px; border-radius: 2px; flex: none; }
.small { font-size: 12px; margin: 8px 0 0; }
</style>
