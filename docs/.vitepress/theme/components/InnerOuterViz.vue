<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { tiledGemmCost } from '@lib/regtile'

// C = A·B，M = N = K = 8；把 C 切成 T × T 的块：T = 1 是内积，T = 8 是外积
const M = 8, N = 8, K = 8
const options = [
  { t: 1, label: '1 × 1（内积）' },
  { t: 2, label: '2 × 2' },
  { t: 4, label: '4 × 4' },
  { t: 8, label: '8 × 8（外积）' },
]
const T = ref(1)
const step = ref(0) // 第几步；一步 = 某个块的某个 k
const tilesPerSide = computed(() => M / T.value)
const totalSteps = computed(() => tilesPerSide.value ** 2 * K)
const tileIdx = computed(() => Math.floor(step.value / K))
const k = computed(() => step.value % K)
const ti = computed(() => Math.floor(tileIdx.value / tilesPerSide.value))
const tj = computed(() => tileIdx.value % tilesPerSide.value)
const rows = computed(() => [ti.value * T.value, (ti.value + 1) * T.value])
const cols = computed(() => [tj.value * T.value, (tj.value + 1) * T.value])
const inRows = (i: number) => i >= rows.value[0] && i < rows.value[1]
const inCols = (j: number) => j >= cols.value[0] && j < cols.value[1]
const range = (n: number) => Array.from({ length: n }, (_, i) => i)
const slice = (r: number[]) => (r[1] - r[0] === 1 ? `${r[0]}` : `${r[0]}:${r[1]}`)

const total = computed(() => tiledGemmCost(M, N, K, T.value, T.value))
const done = computed(() => step.value + 1)
const allCosts = options.map((o) => ({ ...o, c: tiledGemmCost(M, N, K, o.t, o.t) }))

const aClass = (i: number, kk: number) => (inRows(i) ? (kk === k.value ? 'load' : kk < k.value ? 'used' : '') : '')
const bClass = (kk: number, j: number) => (inCols(j) ? (kk === k.value ? 'load' : kk < k.value ? 'used' : '') : '')
function cClass(i: number, j: number) {
  const idx = Math.floor(i / T.value) * tilesPerSide.value + Math.floor(j / T.value)
  return idx < tileIdx.value ? 'done' : idx === tileIdx.value ? 'acc' : ''
}

// SVG 布局：B 在右上，A 在左下，C 在右下
const S = 17, GAP = 8, TOP = 14
const ax = 0, bx = K * S + GAP, by = TOP, ay = TOP + K * S + GAP
const W = bx + N * S, H = ay + M * S + 4

const timer = ref<number | null>(null)
const speed = computed(() => ({ 1: 70, 2: 160, 4: 450, 8: 700 })[T.value] ?? 300)
function stop() { if (timer.value !== null) { clearInterval(timer.value); timer.value = null } }
function next() { if (step.value + 1 < totalSteps.value) step.value++; else stop() }
function prev() { if (step.value > 0) step.value-- }
function nextTile() { step.value = Math.min(totalSteps.value - 1, (tileIdx.value + 1) * K) }
function toggle() {
  if (timer.value !== null) return stop()
  if (step.value + 1 >= totalSteps.value) step.value = 0
  timer.value = window.setInterval(next, speed.value)
}
function reset() { stop(); step.value = 0 }
watch(T, reset)
onUnmounted(stop)
</script>

<template>
  <div class="widget iov">
    <h4>同一个 GEMM，块从 1 × 1 调到 8 × 8：从内积走到外积</h4>

    <div class="row controls">
      <div class="field">
        <span class="flabel">块大小 TM × TN（M = N = K = 8）</span>
        <div class="toggle-group">
          <button v-for="o in options" :key="o.t" :class="{ active: T === o.t }" @click="T = o.t">{{ o.label }}</button>
        </div>
      </div>
    </div>
    <div class="row controls">
      <button class="btn" @click="prev" :disabled="step === 0">上一步</button>
      <button class="btn" @click="next" :disabled="step + 1 >= totalSteps">下一步</button>
      <button class="btn" @click="nextTile" :disabled="tileIdx + 1 >= tilesPerSide ** 2">下一块</button>
      <button class="btn" @click="toggle">{{ timer === null ? '播放' : '暂停' }}</button>
      <button class="btn" @click="reset">重置</button>
      <span class="muted small">第 {{ done }} / {{ totalSteps }} 步</span>
    </div>

    <div class="body">
      <svg :viewBox="`0 0 ${W} ${H}`" class="mats" role="img" aria-label="A、B、C 上这一步读哪些元素、累加哪些输出">
        <text class="lbl" :x="bx" :y="10">B（K × N）</text>
        <text class="lbl" :x="ax" :y="ay - 4">A（M × K）</text>
        <g v-for="kk in range(K)" :key="'b' + kk">
          <rect v-for="j in range(N)" :key="j" class="cell" :class="bClass(kk, j)" :x="bx + j * S" :y="by + kk * S" :width="S" :height="S" />
        </g>
        <g v-for="i in range(M)" :key="'a' + i">
          <rect v-for="kk in range(K)" :key="kk" class="cell" :class="aClass(i, kk)" :x="ax + kk * S" :y="ay + i * S" :width="S" :height="S" />
        </g>
        <g v-for="i in range(M)" :key="'c' + i">
          <rect
            v-for="j in range(N)"
            :key="j"
            class="cell"
            :class="cClass(i, j)"
            :style="cClass(i, j) === 'acc' ? { fillOpacity: 0.2 + (0.8 * (k + 1)) / K } : {}"
            :x="bx + j * S"
            :y="ay + i * S"
            :width="S"
            :height="S"
          />
        </g>
        <line v-for="x in tilesPerSide - 1" :key="'h' + x" class="tline" :x1="bx" :x2="bx + N * S" :y1="ay + x * T * S" :y2="ay + x * T * S" />
        <line v-for="x in tilesPerSide - 1" :key="'v' + x" class="tline" :y1="ay" :y2="ay + M * S" :x1="bx + x * T * S" :x2="bx + x * T * S" />
        <rect class="frame" :x="bx" :y="ay" :width="N * S" :height="M * S" />
      </svg>

      <div class="code">
        <div class="ln"><code>for I in tiles(M, {{ T }}):</code><span>I = {{ slice(rows) }}</span></div>
        <div class="ln"><code>&nbsp;&nbsp;for J in tiles(N, {{ T }}):</code><span>J = {{ slice(cols) }}</span></div>
        <div class="ln"><code>&nbsp;&nbsp;&nbsp;&nbsp;acc = zeros({{ T }}, {{ T }})</code><span>{{ T * T }} 个累加器</span></div>
        <div class="ln hl"><code>&nbsp;&nbsp;&nbsp;&nbsp;for k in range(K):</code><span>k = {{ k }}</span></div>
        <div class="ln hl"><code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a = A[I, k]</code><span>读 {{ T }} 个</span></div>
        <div class="ln hl"><code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b = B[k, J]</code><span>读 {{ T }} 个</span></div>
        <div class="ln hl"><code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;acc += outer(a, b)</code><span>{{ T * T }} 次 MAC</span></div>
        <div class="ln"><code>&nbsp;&nbsp;&nbsp;&nbsp;C[I, J] = acc</code><span>写回一次</span></div>
        <div class="tag">
          <template v-if="T === 1">块是 1 × 1：外两层就是 i、j，outer 退化成一次乘法，<b>k 在最内层的内积</b>。</template>
          <template v-else-if="T === M">块是整个 C：外两层只转一次，<b>k 成了最外层的外积</b>，C 的部分和全程都在。</template>
          <template v-else>块间按内积走（一块沿 k 累加完再换下一块），块内按外积走。</template>
        </div>
      </div>
    </div>

    <div class="counts">
      到这一步：读了 <b>{{ done * 2 * T }}</b> 个数，做了 <b>{{ done * T * T }}</b> 次 MAC，同时活着 <b>{{ T * T }}</b> 个累加器
    </div>

    <table class="cmp">
      <thead>
        <tr><th>块</th><th>累加器（空间）</th><th>整个 GEMM 读多少个数（load）</th><th>每读 1 个数做几次 MAC</th></tr>
      </thead>
      <tbody>
        <tr v-for="o in allCosts" :key="o.t" :class="{ cur: o.t === T }" @click="T = o.t">
          <td>{{ o.label }}</td>
          <td>{{ o.c.accum }}</td>
          <td>{{ o.c.loads }}</td>
          <td>{{ o.c.macPerLoad }}</td>
        </tr>
      </tbody>
    </table>

    <p class="muted note">
      橙色是这一步读的 A 的一列、B 的一行（各 {{ T }} 个），浅色是这一块已经用过的；C 里蓝色是正在累加的块，颜色越深累加的 k 越多，灰色是已经写回的块。
      总 MAC 永远是 {{ total.macs }}：块每边放大 2 倍，load 减半、累加器变 4 倍。
    </p>
  </div>
</template>

<style scoped>
.controls { margin-bottom: 10px; }
.field { display: flex; flex-direction: column; gap: 3px; }
.flabel { font-size: 12px; color: var(--wg-muted); }
.small { font-size: 12px; }

.body { display: flex; flex-wrap: wrap; gap: 14px 24px; align-items: flex-start; justify-content: center; }
.mats { width: 100%; max-width: 300px; height: auto; flex: 1 1 220px; }
.cell { fill: var(--vp-c-bg); stroke: var(--wg-border); stroke-width: 0.7; transition: fill 0.2s; }
.cell.used { fill: color-mix(in srgb, #f59e0b 22%, transparent); }
.cell.load { fill: #f59e0b; }
.cell.acc { fill: var(--vp-c-brand-1); }
.cell.done { fill: color-mix(in srgb, var(--wg-muted) 22%, transparent); }
.tline, .frame { stroke: var(--wg-text); stroke-width: 1.2; opacity: 0.55; fill: none; }
.lbl { font-size: 10px; font-weight: 600; }

.code { flex: 1 1 280px; max-width: 420px; border: 1px solid var(--wg-border); border-radius: 8px; background: var(--vp-c-bg); padding: 8px 10px; font-size: 12px; }
.ln { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, auto); gap: 4px 12px; padding: 2px 0; align-items: baseline; }
.ln code { background: none; padding: 0; font-size: 12px; white-space: pre-wrap; }
.ln span { color: var(--vp-c-brand-1); font-family: var(--vp-font-family-mono); text-align: right; }
.ln.hl { background: color-mix(in srgb, var(--vp-c-brand-1) 7%, transparent); margin: 0 -10px; padding: 2px 10px; }
.tag { margin-top: 8px; padding-top: 6px; border-top: 1px dashed var(--wg-border); font-size: 12px; line-height: 1.6; }

.counts { font-size: 13px; margin: 12px 0 6px; }
.counts b { font-family: var(--vp-font-family-mono); }
.cmp { width: 100%; display: table; font-size: 12.5px; }
.cmp tr { cursor: pointer; }
.cmp tr.cur td { background: color-mix(in srgb, var(--vp-c-brand-1) 12%, transparent); font-weight: 600; }
.note { font-size: 12.5px; line-height: 1.75; margin-top: 8px; }
</style>
