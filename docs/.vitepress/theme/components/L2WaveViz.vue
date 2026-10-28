<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { l2Replay, launchOrder } from '@lib/triton'

const n = ref(4) // C 切成 n × n 个 tile
const wave = ref(4) // 一波同时跑的 block 数（≈ SM 数）
const groupM = ref(2)
const step = ref(0) // 已经派发了几个 block
const total = computed(() => n.value * n.value)

const panels = computed(() => [
  { key: 'row', title: '行优先', events: l2Replay(launchOrder(n.value, n.value, 1), wave.value) },
  { key: 'grp', title: `分组（GROUP_M = ${groupM.value}）`, events: l2Replay(launchOrder(n.value, n.value, groupM.value), wave.value) },
])

type StripState = '' | 'l2' | 'hit' | 'hbm'
const views = computed(() =>
  panels.value.map(({ key, title, events }) => {
    const cur = step.value > 0 ? events[step.value - 1] : null
    const ws = cur ? cur.wave * wave.value : 0
    const rows: StripState[] = Array(n.value).fill('')
    const cols: StripState[] = Array(n.value).fill('')
    for (const e of events.slice(ws, step.value)) { rows[e.pidM] = 'l2'; cols[e.pidN] = 'l2' }
    if (cur) { rows[cur.pidM] = cur.aHit ? 'hit' : 'hbm'; cols[cur.pidN] = cur.bHit ? 'hit' : 'hbm' }
    const tileState = (i: number) => (i >= step.value ? 'todo' : i === step.value - 1 ? 'cur' : i < ws ? 'done' : 'run')
    const before = ws > 0 ? events[ws - 1].hbm : 0
    return { key, title, events, rows, cols, tileState, cur, waveHbm: cur ? cur.hbm - before : 0, hbm: cur?.hbm ?? 0 }
  }),
)

// SVG 布局：B 的列条带在上，A 的行条带在左，C 在右下
const C = 144, T = 34, GAP = 6
const u = computed(() => C / n.value)
const font = computed(() => Math.min(12, u.value * 0.42))
const W = T + GAP + C

const timer = ref<number | null>(null)
function stop() { if (timer.value !== null) { clearInterval(timer.value); timer.value = null } }
function next() { if (step.value < total.value) step.value++; else stop() }
function prev() { if (step.value > 0) step.value-- }
function nextWave() { step.value = Math.min(total.value, (Math.floor(step.value / wave.value) + 1) * wave.value) }
function toggle() {
  if (timer.value !== null) return stop()
  if (step.value >= total.value) step.value = 0
  timer.value = window.setInterval(next, 900)
}
function reset() { stop(); step.value = 0 }
watch([n, wave, groupM], reset)
onUnmounted(stop)
</script>

<template>
  <div class="widget l2v">
    <h4>同一波 block 读同一条条带：第一次走 HBM，之后命中 L2</h4>

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
          <button v-for="w in [4, 8]" :key="w" :class="{ active: wave === w }" @click="wave = w">{{ w }}</button>
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
      <button class="btn" @click="prev" :disabled="step === 0">上一步</button>
      <button class="btn" @click="next" :disabled="step === total">下一步</button>
      <button class="btn" @click="nextWave" :disabled="step === total">跳到本波结束</button>
      <button class="btn" @click="toggle">{{ timer === null ? '播放' : '暂停' }}</button>
      <button class="btn" @click="reset">重置</button>
      <span class="muted small">已派发 {{ step }} / {{ total }} 个 block</span>
    </div>

    <div class="panels">
      <div v-for="v in views" :key="v.key" class="panel">
        <div class="ptitle">{{ v.title }}</div>
        <svg :viewBox="`0 0 ${W} ${W}`" role="img" :aria-label="`${v.title}的 block 派发和条带读取`">
          <g v-for="(s, j) in v.cols" :key="'b' + j" class="strip" :class="s">
            <rect :x="T + GAP + j * u" :y="0" :width="u" :height="T" />
            <text :x="T + GAP + (j + 0.5) * u" :y="T / 2" :font-size="font" text-anchor="middle" dominant-baseline="central">B{{ j }}</text>
          </g>
          <g v-for="(s, i) in v.rows" :key="'a' + i" class="strip" :class="s">
            <rect :x="0" :y="T + GAP + i * u" :width="T" :height="u" />
            <text :x="T / 2" :y="T + GAP + (i + 0.5) * u" :font-size="font" text-anchor="middle" dominant-baseline="central">A{{ i }}</text>
          </g>
          <g v-for="e in v.events" :key="'t' + e.pid" class="tile" :class="v.tileState(e.pid)">
            <rect :x="T + GAP + e.pidN * u" :y="T + GAP + e.pidM * u" :width="u" :height="u" />
            <text
              :x="T + GAP + (e.pidN + 0.5) * u"
              :y="T + GAP + (e.pidM + 0.5) * u"
              :font-size="font"
              text-anchor="middle"
              dominant-baseline="central"
            >{{ e.pid }}</text>
          </g>
        </svg>

        <div class="log">
          <template v-if="v.cur">
            block {{ v.cur.pid }} → C({{ v.cur.pidM }}, {{ v.cur.pidN }})：
            A{{ v.cur.pidM }} <b :class="v.cur.aHit ? 'hit' : 'hbm'">{{ v.cur.aHit ? '命中 L2' : '从 HBM 读' }}</b>，
            B{{ v.cur.pidN }} <b :class="v.cur.bHit ? 'hit' : 'hbm'">{{ v.cur.bHit ? '命中 L2' : '从 HBM 读' }}</b>
          </template>
          <span v-else class="muted">点「下一步」派发 block 0</span>
        </div>
        <div class="counts">
          本波从 HBM 读 <b>{{ v.waveHbm }}</b> 条 · 累计 <b>{{ v.hbm }}</b> 条
          <span class="muted">（不靠 L2：{{ 2 * step }}）</span>
        </div>
      </div>
    </div>

    <div class="legend">
      <span><i class="sw hbm" />这个 block 从 HBM 读</span>
      <span><i class="sw hit" />这个 block 命中 L2</span>
      <span><i class="sw l2" />本波已读进 L2</span>
      <span><i class="sw run" />本波在跑的 block</span>
      <span><i class="sw done" />上一波已跑完</span>
    </div>

    <p class="muted note">
      C 里的数字是 block 编号，硬件按编号从小到大派发，一次一波。block 算 C(i, j) 要读 A 的第 i 条行条带和 B 的第 j 条列条带。
      本波第一个读某条条带的 block 去 HBM，之后读同一条的命中 L2。所以一波从 HBM 读的条带数 = 这一波覆盖的行数 + 列数。
      左右两边只差在「编号 → tile」的映射，算的东西完全一样。
      简化：同一波的 block 实际是同时跑的，这里一个个派发只是为了看清谁先读；换波时把上一波的条带当作已被挤出 L2。
    </p>
  </div>
</template>

<style scoped>
.controls { margin-bottom: 10px; }
.field { display: flex; flex-direction: column; gap: 3px; }
.flabel { font-size: 12px; color: var(--wg-muted); }
.small { font-size: 12px; }

.panels { display: flex; flex-wrap: wrap; gap: 16px 24px; justify-content: center; margin: 6px 0 10px; }
.panel { flex: 1 1 240px; max-width: 320px; display: flex; flex-direction: column; gap: 6px; }
.ptitle { font-weight: 600; font-size: 13px; text-align: center; }
.panel svg { width: 100%; height: auto; }

.strip rect { fill: var(--vp-c-bg); stroke: var(--wg-muted); stroke-width: 0.8; transition: fill 0.25s; }
.strip.l2 rect { fill: color-mix(in srgb, #f59e0b 30%, transparent); }
.strip.hit rect { fill: color-mix(in srgb, #22c55e 60%, transparent); }
.strip.hbm rect { fill: color-mix(in srgb, #ef4444 60%, transparent); }
.strip text, .tile text { font-family: var(--vp-font-family-mono); pointer-events: none; }

.tile rect { fill: var(--vp-c-bg); stroke: var(--wg-muted); stroke-width: 0.8; transition: fill 0.25s; }
.tile.done rect { fill: color-mix(in srgb, var(--wg-muted) 18%, transparent); }
.tile.run rect { fill: color-mix(in srgb, #f59e0b 30%, transparent); }
.tile.cur rect { fill: var(--vp-c-brand-1); stroke: var(--vp-c-brand-1); }
.widget .tile.cur text { fill: #fff; font-weight: 700; }
.tile.todo text, .tile.done text { opacity: 0.5; }

.log { font-size: 12.5px; min-height: 2.6em; line-height: 1.6; }
.log b.hbm { color: #ef4444; }
.log b.hit { color: #16a34a; }
.counts { font-size: 12.5px; }
.counts b { font-family: var(--vp-font-family-mono); }

.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--wg-muted); }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; border: 1px solid var(--wg-border); }
.sw.hbm { background: color-mix(in srgb, #ef4444 60%, transparent); }
.sw.hit { background: color-mix(in srgb, #22c55e 60%, transparent); }
.sw.l2, .sw.run { background: color-mix(in srgb, #f59e0b 30%, transparent); }
.sw.done { background: color-mix(in srgb, var(--wg-muted) 18%, transparent); }
.note { font-size: 12.5px; line-height: 1.75; margin-top: 10px; }
</style>
