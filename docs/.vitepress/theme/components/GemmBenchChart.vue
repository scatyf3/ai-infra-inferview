<script setup lang="ts">
import { computed, ref } from 'vue'
import data from '../../../../bench/triton-gemm/results/rtx5070ti.json'

interface Pt { n: number; ms: number; tflops: number }
interface Series { id: string; label: string; points: Pt[] }

// which = 'ladder'：v0–v5 + cuBLAS；which = 'group'：两种 tile × 行优先 / 分组
const props = withDefaults(defineProps<{ which?: 'ladder' | 'group' }>(), { which: 'ladder' })

// 实测负载频率下的 bf16 Tensor Core 峰值：70 SM × 512 FLOP/clk × f
const peak = (data.env.sms * 512 * data.load.sm_clock_mhz * 1e6) / 1e12
const series = (props.which === 'ladder' ? data.ladder : data.group.series) as Series[]
const sizes = (props.which === 'ladder' ? data.sizes : data.group.sizes) as number[]
const colorVar = (id: string) => `var(--gb-${id})`
// 虚线：cuBLAS 参照线，以及 group 图里的行优先（group 图颜色区分 tile 大小）
const dashed = (id: string) => id === 'cublas' || id.endsWith('g1')
const title = props.which === 'ladder'
  ? `${data.env.gpu} 上的 bf16 GEMM（M = N = K = n）`
  : '程序编号顺序：行优先 vs 分组（同一个 v4 kernel，只换 GROUP_M）'

const hidden = ref(new Set<string>())
const scale = ref<'lin' | 'log'>('lin')
const showTable = ref(false)
function toggle(id: string) {
  const s = new Set(hidden.value)
  s.has(id) ? s.delete(id) : s.add(id)
  hidden.value = s
}

const W = 640, H = 340, PL = 46, PR = 16, PT = 16, PB = 40
const lx0 = Math.log2(sizes[0]), lx1 = Math.log2(sizes[sizes.length - 1])
const x = (n: number) => PL + ((Math.log2(n) - lx0) / (lx1 - lx0)) * (W - PL - PR)
const yMaxLin = Math.ceil(peak / 20) * 20
const y = (v: number) => {
  if (scale.value === 'lin') return PT + (1 - v / yMaxLin) * (H - PT - PB)
  const a = Math.log10(0.05), b = Math.log10(yMaxLin)
  return PT + (1 - (Math.log10(Math.max(v, 0.05)) - a) / (b - a)) * (H - PT - PB)
}
const yTicks = computed(() => (scale.value === 'lin' ? [0, 20, 40, 60, 80, 100].filter((t) => t <= yMaxLin) : [0.1, 1, 10, 100]))
const xTicks = [256, 512, 1024, 2048, 4096, 8192, 16384].filter((t) => t >= sizes[0] && t <= sizes[sizes.length - 1])
const fmtN = (n: number) => (n >= 1024 ? `${n / 1024}K` : `${n}`)
const fmtT = (v: number) => (v < 1 ? v.toFixed(2) : v < 10 ? v.toFixed(1) : v.toFixed(0))

const visible = computed(() => series.filter((s) => !hidden.value.has(s.id)))
const path = (s: Series) => s.points.map((p, i) => `${i ? 'L' : 'M'}${x(p.n).toFixed(1)},${y(p.tflops).toFixed(1)}`).join('')

// 十字线：吸附到最近的 n
const hover = ref<number | null>(null)
const svgEl = ref<SVGSVGElement | null>(null)
function onMove(e: PointerEvent) {
  const r = svgEl.value!.getBoundingClientRect()
  const px = ((e.clientX - r.left) / r.width) * W
  let best = sizes[0]
  for (const n of sizes) if (Math.abs(x(n) - px) < Math.abs(x(best) - px)) best = n
  hover.value = best
}
const readout = computed(() => {
  const n = hover.value
  if (n === null) return []
  return visible.value
    .map((s) => ({ s, p: s.points.find((p) => p.n === n) }))
    .filter((r) => r.p)
    .sort((a, b) => b.p!.tflops - a.p!.tflops)
})
const tipLeft = computed(() => (hover.value === null ? 0 : (x(hover.value) / W) * 100))
</script>

<template>
  <div class="widget gb">
    <h4>{{ title }}</h4>
    <div class="row controls">
      <div class="toggle-group">
        <button :class="{ active: scale === 'lin' }" @click="scale = 'lin'">线性 y</button>
        <button :class="{ active: scale === 'log' }" @click="scale = 'log'">对数 y</button>
      </div>
      <button class="btn" @click="showTable = !showTable">{{ showTable ? '看图' : '看表' }}</button>
    </div>

    <div class="legend">
      <button v-for="s in series" :key="s.id" class="key" :class="{ off: hidden.has(s.id) }" @click="toggle(s.id)">
        <svg width="18" height="10"><line x1="1" y1="5" x2="17" y2="5" :stroke="colorVar(s.id)" stroke-width="2.5" :stroke-dasharray="dashed(s.id) ? '4 3' : undefined" /></svg>
        {{ s.label }}
      </button>
    </div>

    <div v-if="!showTable" class="plot">
      <svg ref="svgEl" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="TFLOP/s 随矩阵边长变化的曲线"
        @pointermove="onMove" @pointerleave="hover = null">
        <g class="grid">
          <line v-for="t in yTicks" :key="'y' + t" :x1="PL" :x2="W - PR" :y1="y(t)" :y2="y(t)" />
        </g>
        <line :x1="PL" :x2="W - PR" :y1="y(peak)" :y2="y(peak)" class="peak" />
        <text :x="PL + 4" :y="y(peak) - 5" class="lbl">峰值 ≈ {{ peak.toFixed(0) }} TFLOP/s（@ {{ (data.load.sm_clock_mhz / 1000).toFixed(2) }} GHz）</text>
        <g class="axis">
          <text v-for="t in yTicks" :key="'yl' + t" :x="PL - 6" :y="y(t) + 4" text-anchor="end">{{ t }}</text>
          <text v-for="t in xTicks" :key="'xl' + t" :x="x(t)" :y="H - PB + 16" text-anchor="middle">{{ fmtN(t) }}</text>
          <text :x="(PL + W - PR) / 2" :y="H - 4" text-anchor="middle">矩阵边长 n（对数刻度）</text>
          <text x="12" :y="(PT + H - PB) / 2" text-anchor="middle" :transform="`rotate(-90 12 ${(PT + H - PB) / 2})`">TFLOP/s</text>
        </g>
        <line v-if="hover !== null" :x1="x(hover)" :x2="x(hover)" :y1="PT" :y2="H - PB" class="cross" />
        <g v-for="s in visible" :key="s.id">
          <path :d="path(s)" class="line" :stroke="colorVar(s.id)" :stroke-dasharray="dashed(s.id) ? '5 4' : undefined" />
          <circle v-for="p in s.points.filter((p) => p.n === hover)" :key="p.n" :cx="x(p.n)" :cy="y(p.tflops)" r="4.5"
            :fill="colorVar(s.id)" class="dot" />
        </g>
      </svg>
      <div v-if="hover !== null" class="tip" :style="{ left: tipLeft + '%' }" :class="{ flip: tipLeft > 55 }">
        <div class="tip-h">n = {{ hover }}</div>
        <div v-for="r in readout" :key="r.s.id" class="tip-r">
          <span class="sw" :style="{ background: colorVar(r.s.id) }" />
          <span class="tip-l">{{ r.s.label }}</span>
          <b>{{ fmtT(r.p!.tflops) }}</b>
        </div>
      </div>
    </div>

    <div v-else class="tbl">
      <table>
        <thead><tr><th>n</th><th v-for="s in series" :key="s.id">{{ s.label }}</th></tr></thead>
        <tbody>
          <tr v-for="n in sizes" :key="n">
            <td>{{ n }}</td>
            <td v-for="s in series" :key="s.id">{{ s.points.find((p) => p.n === n) ? fmtT(s.points.find((p) => p.n === n)!.tflops) : '—' }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <p class="muted small">
      单位 TFLOP/s = 2n³ / 时间。每个点是 <code>triton.testing.do_bench</code> 的中位数（每次前清 L2）。
      {{ data.env.date }}，driver {{ data.env.driver }}，torch {{ data.env.torch }}，Triton {{ data.env.triton }}。点图例可隐藏曲线。
    </p>
  </div>
</template>

<style scoped>
.gb {
  --gb-v0: #2a78d6; --gb-v1: #eb6834; --gb-v2: #1baf7a; --gb-v3: #eda100; --gb-v4: #e87ba4; --gb-v5: #008300;
  --gb-cublas: var(--vp-c-text-2);
  --gb-t64g1: #2a78d6; --gb-t64g8: #2a78d6; --gb-t128g1: #eb6834; --gb-t128g8: #eb6834;
}
:root.dark .gb {
  --gb-v0: #3987e5; --gb-v1: #d95926; --gb-v2: #199e70; --gb-v3: #c98500; --gb-v4: #d55181; --gb-v5: #008300;
  --gb-t64g1: #3987e5; --gb-t64g8: #3987e5; --gb-t128g1: #d95926; --gb-t128g8: #d95926;
}
.controls { margin-bottom: 8px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-bottom: 6px; }
.key { display: inline-flex; align-items: center; gap: 5px; font-size: 12.5px; color: var(--wg-text); background: none; border: 0; padding: 2px 0; cursor: pointer; }
.key.off { opacity: 0.35; }
.plot { position: relative; }
.plot svg { width: 100%; height: auto; display: block; background: var(--vp-c-bg); border-radius: 8px; border: 1px solid var(--wg-border); touch-action: pan-y; }
.grid line { stroke: var(--wg-border); stroke-width: 1; }
.peak { stroke: var(--wg-muted); stroke-width: 1; stroke-dasharray: 4 4; }
.cross { stroke: var(--wg-muted); stroke-width: 1; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.dot { stroke: var(--vp-c-bg); stroke-width: 2; }
.lbl { font-size: 11px; fill: var(--wg-muted); }
.axis text { font-size: 10.5px; fill: var(--wg-muted); font-variant-numeric: tabular-nums; }
.tip {
  position: absolute; top: 12px; transform: translateX(12px); pointer-events: none;
  background: var(--vp-c-bg-elv); border: 1px solid var(--wg-border); border-radius: 8px;
  padding: 6px 10px; font-size: 12px; box-shadow: var(--vp-shadow-2); min-width: 170px;
}
.tip.flip { transform: translateX(calc(-100% - 12px)); }
.tip-h { font-weight: 600; margin-bottom: 3px; }
.tip-r { display: flex; align-items: center; gap: 6px; line-height: 1.6; }
.tip-l { flex: 1; color: var(--wg-muted); }
.tip-r b { font-variant-numeric: tabular-nums; }
.sw { width: 10px; height: 10px; border-radius: 2px; flex: none; }
.tbl { overflow-x: auto; }
.tbl table { font-size: 12px; font-variant-numeric: tabular-nums; margin: 0; }
.small { font-size: 12px; margin: 8px 0 0; }
</style>
