<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { SECTOR_FLOATS, transposeSteps, transposeTotals, type Cell, type TransposeMode } from '@lib/transpose'
import StatCard from './ui/StatCard.vue'

// 缩小版：矩阵 16×16，tile 8×8（2×2 个 block），一个 warp 8 个线程，shared memory 8 个 bank
const N = 16, T = 8
const props = withDefaults(defineProps<{ mode?: TransposeMode }>(), { mode: 'smem' })
const mode = ref<TransposeMode>(props.mode)
const bx = ref(1), by = ref(0)
const x0 = computed(() => bx.value * T), y0 = computed(() => by.value * T)
const pitch = computed(() => (mode.value === 'padded' ? T + 1 : T))

const steps = computed(() => transposeSteps(mode.value, N, T, bx.value, by.value))
const totals = computed(() => transposeTotals(steps.value))
const cur = ref(0)
watch(steps, () => { cur.value = 0 })
const step = computed(() => steps.value[cur.value])

const key = (r: number, c: number) => `${r},${c}`
const laneColor = (t: number) => `hsl(${t * 45} 75% 55%)`
const idIn = (r: number, c: number) => r * N + c

// 回放到当前步：tile 里存了哪些格、out 里写了哪些格
const done = computed(() => {
  const tile = new Set<string>(), out = new Set<string>()
  for (const s of steps.value.slice(0, cur.value + 1)) {
    if (s.smem?.op === 'store') s.smem.cells.forEach(([r, c]) => tile.add(key(r, c)))
    if (s.hbm?.mat === 'out') s.hbm.cells.forEach(([r, c]) => out.add(key(r, c)))
  }
  return { tile, out }
})

// 当前步每个格子是哪个 lane 在访问；被搬进来但没用到的 sector 格
const laneMap = (cells?: Cell[]) => new Map((cells ?? []).map(([r, c], t) => [key(r, c), t]))
const hbmLanes = computed(() => laneMap(step.value.hbm?.cells))
const smemLanes = computed(() => laneMap(step.value.smem?.cells))
const sectorCells = computed(() => {
  const s = new Set<string>()
  for (const [r, c] of step.value.hbm?.cells ?? []) {
    const start = Math.floor((r * N + c) / SECTOR_FLOATS) * SECTOR_FLOATS
    for (let i = start; i < start + SECTOR_FLOATS; i++) s.add(key(Math.floor(i / N), i % N))
  }
  return s
})

function hbmCellStyle(mat: 'in' | 'out', r: number, c: number) {
  if (step.value.hbm?.mat !== mat) return {}
  const t = hbmLanes.value.get(key(r, c))
  return t === undefined ? {} : { background: laneColor(t), color: '#fff' }
}
const inBlock = (r: number, c: number, ox: number, oy: number) => r >= oy && r < oy + T && c >= ox && c < ox + T
const bankOf = (r: number, c: number) => (r * pitch.value + c) % T

// out 的这个块对应 in 的转置位置
const outX0 = computed(() => y0.value), outY0 = computed(() => x0.value)

const codeLines = computed(() =>
  mode.value === 'naive'
    ? [
        { k: 'read', code: 'float v = in[y0 + ty][x0 + tx];' },
        { k: 'write', code: 'out[x0 + tx][y0 + ty] = v;' },
      ]
    : [
        { k: 'decl', code: `__shared__ float tile[${T}][${pitch.value}];` },
        { k: 'read', code: 'tile[ty][tx] = in[y0 + ty][x0 + tx];' },
        { k: 'sync', code: '__syncthreads();' },
        { k: 'write', code: 'out[x0 + ty][y0 + tx] = tile[tx][ty];' },
      ],
)

const stepText = computed(() => {
  const s = step.value, w = s.warp
  if (s.kind === 'sync') return '整个 block 在这里等：所有 warp 都把自己那行存进 tile，才能开始阶段 2'
  if (mode.value === 'naive')
    return s.kind === 'read'
      ? `warp ${w}：线程 t 读 in[${y0.value + w}][${x0.value}+t]，同一行连续 ${T} 个`
      : `warp ${w}：线程 t 把刚读的值写到 out[${x0.value}+t][${y0.value + w}]，是 out 的一列`
  return s.kind === 'read'
    ? `阶段 1 · warp ${w}：读 in 第 ${y0.value + w} 行，存进 tile 第 ${w} 行`
    : `阶段 2 · warp ${w}：线程 t 取 tile[t][${w}]（一列，是别的 warp 存的），写到 out 第 ${x0.value + w} 行`
})

const timer = ref<number | null>(null)
const next = () => { cur.value = (cur.value + 1) % steps.value.length }
const prev = () => { cur.value = (cur.value - 1 + steps.value.length) % steps.value.length }
function stop() { if (timer.value !== null) { clearInterval(timer.value); timer.value = null } }
function toggle() { if (timer.value !== null) return stop(); timer.value = window.setInterval(next, 900) }
function reset() { stop(); cur.value = 0 }
function pickBlock(r: number, c: number) { bx.value = Math.floor(c / T); by.value = Math.floor(r / T) }
onUnmounted(stop)
</script>

<template>
  <div class="widget tv">
    <div class="row head">
      <h4>矩阵转置：一个 block 的访存</h4>
      <div class="toggle-group">
        <button :class="{ active: mode === 'naive' }" @click="mode = 'naive'">直接转置</button>
        <button :class="{ active: mode === 'smem' }" @click="mode = 'smem'">shared memory 中转</button>
        <button :class="{ active: mode === 'padded' }" @click="mode = 'padded'">+ padding</button>
      </div>
    </div>

    <div class="row controls">
      <button class="btn" @click="prev">上一步</button>
      <button class="btn" @click="next">下一步</button>
      <button class="btn" @click="toggle">{{ timer === null ? '播放' : '暂停' }}</button>
      <button class="btn" @click="reset">重置</button>
      <span class="muted small">第 {{ cur + 1 }} / {{ steps.length }} 步 · 点 in 里的块换一个 block</span>
    </div>

    <div class="steptext">{{ stepText }}</div>

    <div class="code">
      <div v-for="l in codeLines" :key="l.k" class="ln" :class="{ hl: l.k === step.kind }"><code>{{ l.code }}</code></div>
    </div>

    <div class="mats">
      <!-- in -->
      <div class="panel">
        <div class="ptitle">in（HBM，{{ N }}×{{ N }}）</div>
        <div class="mgrid" :style="{ gridTemplateColumns: `repeat(${N}, 1fr)` }">
          <template v-for="r in N" :key="'ir' + r">
            <div
              v-for="c in N"
              :key="'i' + r + ',' + c"
              class="mc"
              :class="{
                blk: inBlock(r - 1, c - 1, x0, y0),
                sector: step.hbm?.mat === 'in' && sectorCells.has(key(r - 1, c - 1)),
              }"
              :style="hbmCellStyle('in', r - 1, c - 1)"
              @click="pickBlock(r - 1, c - 1)"
            >{{ idIn(r - 1, c - 1) }}</div>
          </template>
        </div>
      </div>

      <!-- shared memory tile -->
      <div class="panel">
        <div class="ptitle">tile（shared memory，{{ T }}×{{ pitch }}）</div>
        <template v-if="mode !== 'naive'">
          <div class="mgrid tgrid" :style="{ gridTemplateColumns: `repeat(${pitch}, 1fr)` }">
            <template v-for="r in T" :key="'tr' + r">
              <div
                v-for="c in pitch"
                :key="'t' + r + ',' + c"
                class="tc"
                :class="{ pad: c - 1 >= T }"
                :style="smemLanes.has(key(r - 1, c - 1)) ? { background: laneColor(smemLanes.get(key(r - 1, c - 1))!), color: '#fff' } : {}"
              >
                <span v-if="c - 1 < T && done.tile.has(key(r - 1, c - 1))">{{ idIn(y0 + r - 1, x0 + c - 1) }}</span>
                <sub>{{ bankOf(r - 1, c - 1) }}</sub>
              </div>
            </template>
          </div>
          <div class="banks">
            <div v-for="(n, b) in step.banks.length ? step.banks : Array(T).fill(0)" :key="'b' + b" class="bank" :class="{ hot: n > 1 }">
              <div class="bar" :style="{ height: `${(n / T) * 100}%` }" />
              <span>{{ b }}</span>
            </div>
          </div>
          <div class="muted small">格子右下角是 bank 编号；下面的柱子是这一步每个 bank 被几个不同地址访问</div>
        </template>
        <div v-else class="muted small nosmem">直接转置不经过 shared memory：线程把读进寄存器的值直接写出去</div>
      </div>

      <!-- out -->
      <div class="panel">
        <div class="ptitle">out（HBM，{{ N }}×{{ N }}）</div>
        <div class="mgrid" :style="{ gridTemplateColumns: `repeat(${N}, 1fr)` }">
          <template v-for="r in N" :key="'or' + r">
            <div
              v-for="c in N"
              :key="'o' + r + ',' + c"
              class="mc"
              :class="{
                blk: inBlock(r - 1, c - 1, outX0, outY0),
                sector: step.hbm?.mat === 'out' && sectorCells.has(key(r - 1, c - 1)),
              }"
              :style="hbmCellStyle('out', r - 1, c - 1)"
            >{{ done.out.has(key(r - 1, c - 1)) ? idIn(c - 1, r - 1) : '' }}</div>
          </template>
        </div>
      </div>
    </div>

    <div class="grid stats">
      <StatCard
        label="这一步 HBM"
        :value="step.hbm ? `${step.sectors} 个 sector` : '—'"
        :sub="step.hbm ? `搬了 ${step.sectors * 32} B，有用的 ${T * 4} B` : '这一步不访问 HBM'"
        :tone="step.hbm ? (step.sectors > 1 ? 'bad' : 'good') : 'default'"
      />
      <StatCard
        label="这一步 shared memory"
        :value="step.smem ? (step.ways > 1 ? `${step.ways} 路冲突` : '无冲突') : '—'"
        :sub="step.smem ? `要分 ${step.ways} 次完成` : '这一步不访问 shared memory'"
        :tone="step.smem ? (step.ways > 1 ? 'bad' : 'good') : 'default'"
      />
      <StatCard
        label="整个 block 合计"
        :value="`${totals.sectors} 个 sector`"
        :sub="totals.smemAccesses ? `shared memory 访问 ${totals.smemAccesses} 次，实际 ${totals.smemPasses} 次` : '不用 shared memory'"
      />
    </div>

    <p class="muted note">
      颜色是 lane（warp 里的第几个线程），同一颜色是同一个线程。格子里的数是元素在 in 里的编号 <code>r × {{ N }} + c</code>，写进 out 后能看出 <code>out[c][r] = in[r][c]</code>。
      HBM 上的浅色底是这一步被整块搬进来、但没用到的 sector（32 B = 8 个 float）。
      这里只单步演示一个 block；实际 launch 时 {{ (N / T) ** 2 }} 个 block 在不同 SM 上同时跑，各搬各的块，没有预处理。
      为了能看清，规模缩小了：一个 warp {{ T }} 个线程、{{ T }} 个 bank。真实的 32×32 tile 是按行读 4 个 sector、按列写 32 个，按列读 tile 是 32 路冲突，比例和这里一样。
    </p>
  </div>
</template>

<style scoped>
.head { justify-content: space-between; align-items: center; margin-bottom: 12px; }
.head h4 { margin: 0; }
.controls { margin-bottom: 8px; align-items: center; }
.small { font-size: 12px; }
.steptext { font-size: 13px; font-weight: 600; margin: 4px 0 8px; min-height: 1.5em; }

.code { border: 1px solid var(--wg-border); border-radius: 8px; background: var(--vp-c-bg); padding: 6px 10px; font-size: 12px; margin-bottom: 12px; }
.ln { padding: 2px 0; }
.ln code { background: none; padding: 0; font-size: 12px; }
.ln.hl { background: color-mix(in srgb, var(--vp-c-brand-1) 14%, transparent); margin: 0 -10px; padding: 2px 10px; font-weight: 600; }

.mats { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.panel { flex: 1 1 220px; max-width: 300px; min-width: 0; }
.ptitle { font-size: 12px; font-weight: 600; margin-bottom: 4px; }
.mgrid { display: grid; gap: 1px; background: var(--wg-border); border: 1px solid var(--wg-border); }
.mc { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; font-size: 7px; font-family: var(--vp-font-family-mono); background: var(--vp-c-bg); color: var(--wg-muted); cursor: pointer; overflow: hidden; }
.mc.blk { background: color-mix(in srgb, var(--vp-c-brand-1) 8%, var(--vp-c-bg)); color: var(--wg-text); }
.mc.sector { background: color-mix(in srgb, #ef4444 22%, var(--vp-c-bg)); }

.tgrid { gap: 2px; }
.tc { position: relative; aspect-ratio: 1; display: flex; align-items: center; justify-content: center; font-size: 9px; font-family: var(--vp-font-family-mono); background: var(--vp-c-bg); }
.tc sub { position: absolute; right: 2px; bottom: 1px; font-size: 7px; opacity: 0.55; }
.tc.pad { background: repeating-linear-gradient(45deg, transparent 0 3px, color-mix(in srgb, var(--wg-muted) 25%, transparent) 3px 5px); }
.banks { display: flex; gap: 3px; height: 44px; margin: 8px 0 4px; align-items: flex-end; }
.bank { flex: 1; height: 100%; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; font-size: 9px; font-family: var(--vp-font-family-mono); }
.bar { width: 100%; background: #22c55e; border-radius: 2px 2px 0 0; min-height: 1px; }
.bank.hot .bar { background: #ef4444; }
.nosmem { padding: 12px 0; }

.stats { margin: 14px 0 6px; }
.note { font-size: 12.5px; line-height: 1.75; margin-top: 10px; }
</style>
