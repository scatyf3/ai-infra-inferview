<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { cdiv, launchOrder, waveUsage } from '@lib/triton'

const n = ref(4) // C 切成 n × n 个 tile
const wave = ref(4) // 一波同时跑的 block 数（≈ SM 数）
const groupM = ref(2)
const w = ref(0) // 当前第几波
const total = computed(() => n.value * n.value)
const numWaves = computed(() => cdiv(total.value, wave.value))
const waveStart = computed(() => w.value * wave.value)
const blocksInWave = computed(() => Math.min(wave.value, total.value - waveStart.value))
const noL2Cum = computed(() => 2 * Math.min(total.value, waveStart.value + wave.value))

// 鼠标停在哪个 tile 上：高亮它读的两条条带
const hov = ref<{ key: string; m: number; n: number } | null>(null)

const views = computed(() =>
  [
    { key: 'row', title: '行优先', g: 1 },
    { key: 'grp', title: `分组（GROUP_M = ${groupM.value}）`, g: groupM.value },
  ].map(({ key, title, g }) => {
    const order = launchOrder(n.value, n.value, g)
    const use = (wi: number) => waveUsage(order, wi * wave.value, wave.value)
    const cur = use(w.value)
    const rows = Array.from({ length: n.value }, (_, i) => cur.rows.get(i) ?? 0)
    const cols = Array.from({ length: n.value }, (_, j) => cur.cols.get(j) ?? 0)
    let cum = 0
    for (let wi = 0; wi <= w.value; wi++) { const u = use(wi); cum += u.rows.size + u.cols.size }
    const tileState = (pid: number) => (pid < waveStart.value ? 'done' : pid < waveStart.value + wave.value ? 'run' : 'todo')
    return { key, title, order, rows, cols, tileState, a: cur.rows.size, b: cur.cols.size, cum }
  }),
)
const stripClass = (k: number) => (k === 0 ? '' : k === 1 ? 'solo' : 'shared')
const focused = (key: string, axis: 'm' | 'n', i: number) => hov.value?.key === key && hov.value[axis] === i

// SVG 布局：B 的列条带在上，A 的行条带在左，C 在右下
const C = 144, T = 44, GAP = 6
const u = computed(() => C / n.value)
const font = computed(() => Math.min(11, u.value * 0.42))
const W = T + GAP + C

const timer = ref<number | null>(null)
function stop() { if (timer.value !== null) { clearInterval(timer.value); timer.value = null } }
function next() { if (w.value + 1 < numWaves.value) w.value++; else stop() }
function prev() { if (w.value > 0) w.value-- }
function toggle() {
  if (timer.value !== null) return stop()
  if (w.value + 1 >= numWaves.value) w.value = 0
  timer.value = window.setInterval(next, 1500)
}
function reset() { stop(); w.value = 0 }
watch([n, wave, groupM], reset)
onUnmounted(stop)
</script>

<template>
  <div class="widget l2v">
    <h4>一波 block 同时跑：每条条带只从 HBM 读一次，其余命中 L2</h4>

    <div class="row controls">
      <div class="field">
        <span class="flabel">C 的 tile 数</span>
        <div class="toggle-group">
          <button v-for="s in [4, 8]" :key="s" :class="{ active: n === s }" @click="n = s">{{ s }} × {{ s }}</button>
        </div>
      </div>
      <div class="field">
        <span class="flabel">一波 block 数（≈ SM 数）</span>
        <div class="toggle-group">
          <button v-for="x in [4, 8]" :key="x" :class="{ active: wave === x }" @click="wave = x">{{ x }}</button>
        </div>
      </div>
      <div class="field">
        <span class="flabel">右边的 GROUP_M</span>
        <div class="toggle-group">
          <button v-for="g in [2, 4]" :key="g" :class="{ active: groupM === g }" @click="groupM = g">{{ g }}</button>
        </div>
      </div>
    </div>

    <div class="row controls">
      <button class="btn" @click="prev" :disabled="w === 0">上一波</button>
      <button class="btn" @click="next" :disabled="w + 1 >= numWaves">下一波</button>
      <button class="btn" @click="toggle">{{ timer === null ? '播放' : '暂停' }}</button>
      <button class="btn" @click="reset">重置</button>
      <span class="muted small">第 {{ w + 1 }} / {{ numWaves }} 波：block {{ waveStart }}–{{ waveStart + blocksInWave - 1 }} 同时在 SM 上跑</span>
    </div>

    <div class="panels">
      <div v-for="v in views" :key="v.key" class="panel">
        <div class="ptitle">{{ v.title }}</div>
        <svg :viewBox="`0 0 ${W} ${W}`" role="img" :aria-label="`${v.title}：第 ${w + 1} 波读到的条带`" @mouseleave="hov = null">
          <g v-for="(k, j) in v.cols" :key="'b' + j" class="strip" :class="[stripClass(k), { focus: focused(v.key, 'n', j) }]">
            <rect :x="T + GAP + j * u" :y="0" :width="u" :height="T" />
            <text :x="T + GAP + (j + 0.5) * u" :y="T / 2 - font * 0.65" :font-size="font" text-anchor="middle" dominant-baseline="central">B{{ j }}</text>
            <text v-if="k" class="k" :x="T + GAP + (j + 0.5) * u" :y="T / 2 + font * 0.75" :font-size="font" text-anchor="middle" dominant-baseline="central">×{{ k }}</text>
          </g>
          <g v-for="(k, i) in v.rows" :key="'a' + i" class="strip" :class="[stripClass(k), { focus: focused(v.key, 'm', i) }]">
            <rect :x="0" :y="T + GAP + i * u" :width="T" :height="u" />
            <text :x="T / 2" :y="T + GAP + (i + 0.5) * u" :font-size="font" text-anchor="middle" dominant-baseline="central">
              A{{ i }}<tspan v-if="k" class="k"> ×{{ k }}</tspan>
            </text>
          </g>
          <g
            v-for="t in v.order"
            :key="'t' + t.pid"
            class="tile"
            :class="v.tileState(t.pid)"
            @mouseenter="hov = { key: v.key, m: t.pidM, n: t.pidN }"
            @click="hov = { key: v.key, m: t.pidM, n: t.pidN }"
          >
            <rect :x="T + GAP + t.pidN * u" :y="T + GAP + t.pidM * u" :width="u" :height="u" />
            <text
              :x="T + GAP + (t.pidN + 0.5) * u"
              :y="T + GAP + (t.pidM + 0.5) * u"
              :font-size="font"
              text-anchor="middle"
              dominant-baseline="central"
            >{{ t.pid }}</text>
          </g>
        </svg>

        <div class="counts">
          这一波从 HBM 读 <b>{{ v.a }} + {{ v.b }} = {{ v.a + v.b }}</b> 条
          <span class="muted">（不靠 L2：{{ blocksInWave }} × 2 = {{ 2 * blocksInWave }}）</span>
        </div>
        <div class="counts">
          累计 <b>{{ v.cum }}</b> 条 <span class="muted">（不靠 L2：{{ noL2Cum }}）</span>
        </div>
      </div>
    </div>

    <div class="legend">
      <span><i class="sw run" />这一波在跑的 block</span>
      <span><i class="sw done" />前几波已跑完</span>
      <span><i class="sw shared" />读一次，×k 个 block 共用</span>
      <span><i class="sw solo" />读一次，只有 1 个 block 用</span>
    </div>

    <p class="muted note">
      C 里的数字是 block 编号。硬件按编号从小到大一次派一波，这一波的 block 同时在跑；block 算 C(i, j) 要读 A 的第 i 条行条带和 B 的第 j 条列条带（鼠标停在 tile 上看它读哪两条）。
      这一波要的每条条带只从 HBM 读一次进 L2，×k 表示这一波有 k 个 block 读它，其中 k − 1 次命中 L2。
      所以一波从 HBM 读的条带数 = 覆盖的行数 + 列数。左右两边只差在「编号 → tile」的映射，算的东西完全一样。
      简化：换波时把上一波的条带当作已被挤出 L2。
    </p>
  </div>
</template>

<style scoped>
.controls { margin-bottom: 10px; }
.field { display: flex; flex-direction: column; gap: 3px; }
.flabel { font-size: 12px; color: var(--wg-muted); }
.small { font-size: 12px; }

.panels { display: flex; flex-wrap: wrap; gap: 16px 24px; justify-content: center; margin: 6px 0 10px; }
.panel { flex: 1 1 240px; max-width: 320px; display: flex; flex-direction: column; gap: 4px; }
.ptitle { font-weight: 600; font-size: 13px; text-align: center; }
.panel svg { width: 100%; height: auto; }

.strip rect { fill: var(--vp-c-bg); stroke: var(--wg-muted); stroke-width: 0.8; transition: fill 0.3s; }
.strip.shared rect { fill: color-mix(in srgb, #22c55e 55%, transparent); }
.strip.solo rect { fill: color-mix(in srgb, #ef4444 50%, transparent); }
.strip.focus rect { stroke: var(--vp-c-brand-1); stroke-width: 2.5; }
.strip text, .tile text { font-family: var(--vp-font-family-mono); pointer-events: none; }
.widget .strip .k { font-weight: 700; }

.tile { cursor: pointer; }
.tile rect { fill: var(--vp-c-bg); stroke: var(--wg-muted); stroke-width: 0.8; transition: fill 0.3s; }
.tile.done rect { fill: color-mix(in srgb, var(--wg-muted) 18%, transparent); }
.tile.run rect { fill: var(--vp-c-brand-1); stroke: var(--vp-c-brand-1); }
.widget .tile.run text { fill: #fff; font-weight: 700; }
.tile.todo text, .tile.done text { opacity: 0.5; }
.tile:hover rect { stroke: var(--vp-c-brand-1); stroke-width: 2; }

.counts { font-size: 12.5px; }
.counts b { font-family: var(--vp-font-family-mono); }

.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--wg-muted); }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; border: 1px solid var(--wg-border); }
.sw.run { background: var(--vp-c-brand-1); }
.sw.done { background: color-mix(in srgb, var(--wg-muted) 18%, transparent); }
.sw.shared { background: color-mix(in srgb, #22c55e 55%, transparent); }
.sw.solo { background: color-mix(in srgb, #ef4444 50%, transparent); }
.note { font-size: 12.5px; line-height: 1.75; margin-top: 10px; }
</style>
