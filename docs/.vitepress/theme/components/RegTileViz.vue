<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { blockSmemReads, regTileCost } from '@lib/regtile'
import StatCard from './ui/StatCard.vue'

// 缩小版 block tile：C 是 BM × BN，As 是 BM × BK，Bs 是 BK × BN
const BM = 16, BN = 16, BK = 8
const tiles = [1, 2, 4, 8]
const t = ref(8) // TM = TN = t
const vec = ref<1 | 4>(1)
const kk = ref(0)
const ty = ref(0), tx = ref(1) // 选中的线程
const range = (n: number) => Array.from({ length: n }, (_, i) => i)

const threadsPerSide = computed(() => BM / t.value)
watch(t, (v) => {
  if (v < 4) vec.value = 1
  ty.value = Math.min(ty.value, BM / v - 1)
  tx.value = Math.min(tx.value, BN / v - 1)
})
const cost = computed(() => regTileCost(t.value, t.value, vec.value))
const smem = computed(() => blockSmemReads(BM, BN, t.value, t.value))
const pct = (x: number) => `${Math.round(x * 100)}%`

const rowOf = (i: number) => ty.value * t.value + i // a[i] 来自 As 的哪一行
const colOf = (j: number) => tx.value * t.value + j // b[j] 来自 Bs 的哪一列
const mineM = (m: number) => Math.floor(m / t.value) === ty.value
const mineN = (n: number) => Math.floor(n / t.value) === tx.value
const asClass = (m: number, k: number) => (k === kk.value ? (mineM(m) ? 'sel' : 'now') : k < kk.value ? 'past' : '')
const bsClass = (k: number, n: number) => (k === kk.value ? (mineN(n) ? 'sel' : 'now') : k < kk.value ? 'past' : '')
function pick(m: number, n: number) { ty.value = Math.floor(m / t.value); tx.value = Math.floor(n / t.value) }

// SVG 布局：Bs 在右上，As 在左下，C 在右下
const S = 11, GAP = 8, TOP = 14
const ax = 0, bx = BK * S + GAP
const by = TOP, ay = TOP + BK * S + GAP
const W = bx + BN * S, H = ay + BM * S + 16

// 寄存器面板的悬停：a[i] / b[j] / acc[i][j]
const hi = ref<{ a?: number; b?: number } | null>(null)
const accHi = (i: number, j: number) => !!hi.value && (hi.value.a === undefined || hi.value.a === i) && (hi.value.b === undefined || hi.value.b === j)
const caption = computed(() => {
  const h = hi.value, T = t.value
  if (!h) return null
  if (h.a !== undefined && h.b !== undefined) return `acc[${h.a}][${h.b}] += a[${h.a}] * b[${h.b}]`
  if (h.a !== undefined) return `a[${h.a}] = As[${rowOf(h.a)}][${kk.value}]，被 acc 第 ${h.a} 行的 ${T} 个元素各用一次`
  return `b[${h.b}] = Bs[${kk.value}][${colOf(h.b!)}]，被 acc 第 ${h.b} 列的 ${T} 个元素各用一次`
})

const timer = ref<number | null>(null)
function stop() { if (timer.value !== null) { clearInterval(timer.value); timer.value = null } }
function next() { kk.value = (kk.value + 1) % BK }
function prev() { kk.value = (kk.value - 1 + BK) % BK }
function toggle() { if (timer.value !== null) return stop(); timer.value = window.setInterval(next, 1000) }
onUnmounted(stop)
</script>

<template>
  <div class="widget rtv">
    <h4>寄存器 tile：一个线程在每个 kk 里读什么、算什么</h4>

    <div class="row controls">
      <div class="field">
        <span class="flabel">每线程输出 TM × TN</span>
        <div class="toggle-group">
          <button v-for="x in tiles" :key="x" :class="{ active: t === x }" @click="t = x">{{ x }} × {{ x }}</button>
        </div>
      </div>
      <div class="field">
        <span class="flabel">shared memory load</span>
        <div class="toggle-group">
          <button :class="{ active: vec === 1 }" @click="vec = 1">标量</button>
          <button :class="{ active: vec === 4 }" :disabled="t < 4" @click="vec = 4">float4</button>
        </div>
      </div>
      <button class="btn" @click="prev">上一步</button>
      <button class="btn" @click="next">下一步</button>
      <button class="btn" @click="toggle">{{ timer === null ? '播放' : '暂停' }}</button>
      <span class="muted small">kk = {{ kk }} / {{ BK }}</span>
    </div>

    <div class="body">
      <svg :viewBox="`0 0 ${W} ${H}`" class="smem" role="img" aria-label="As、Bs 和 C tile 的线程划分">
        <text class="lbl" :x="bx" :y="10">Bs（BK × BN = {{ BK }} × {{ BN }}）</text>
        <text class="lbl" :x="ax" :y="ay - 4">As（{{ BM }} × {{ BK }}）</text>
        <g v-for="k in range(BK)" :key="'bs' + k">
          <rect v-for="n in range(BN)" :key="n" class="cell" :class="bsClass(k, n)" :x="bx + n * S" :y="by + k * S" :width="S" :height="S" />
        </g>
        <g v-for="m in range(BM)" :key="'as' + m">
          <rect v-for="k in range(BK)" :key="k" class="cell" :class="asClass(m, k)" :x="ax + k * S" :y="ay + m * S" :width="S" :height="S" />
        </g>
        <g v-for="m in range(BM)" :key="'c' + m">
          <rect
            v-for="n in range(BN)"
            :key="n"
            class="cell c"
            :class="{ mine: mineM(m) && mineN(n) }"
            :style="mineM(m) && mineN(n) ? { fillOpacity: 0.25 + (0.75 * (kk + 1)) / BK } : {}"
            :x="bx + n * S"
            :y="ay + m * S"
            :width="S"
            :height="S"
            @click="pick(m, n)"
          />
        </g>
        <!-- 线程边界 -->
        <line v-for="i in threadsPerSide - 1" :key="'h' + i" class="tline" :x1="bx" :x2="bx + BN * S" :y1="ay + i * t * S" :y2="ay + i * t * S" />
        <line v-for="i in threadsPerSide - 1" :key="'v' + i" class="tline" :y1="ay" :y2="ay + BM * S" :x1="bx + i * t * S" :x2="bx + i * t * S" />
        <rect class="cframe" :x="bx" :y="ay" :width="BN * S" :height="BM * S" />
        <text class="lbl" :x="bx" :y="H - 3">C tile（{{ BM }} × {{ BN }}），{{ smem.threads }} 个线程，点格子换线程</text>
      </svg>

      <div class="regs">
        <div class="rtitle">线程 ({{ ty }}, {{ tx }}) 的寄存器</div>
        <div class="rgrid" :style="{ gridTemplateColumns: `repeat(${t + 1}, minmax(0, 26px))` }" @mouseleave="hi = null">
          <span class="corner" />
          <span v-for="j in range(t)" :key="'b' + j" class="r b" :class="{ on: hi && (hi.b === j) }" @mouseenter="hi = { b: j }" @click="hi = { b: j }">b{{ j }}</span>
          <template v-for="i in range(t)" :key="'row' + i">
            <span class="r a" :class="{ on: hi && hi.a === i }" @mouseenter="hi = { a: i }" @click="hi = { a: i }">a{{ i }}</span>
            <span
              v-for="j in range(t)"
              :key="j"
              class="r acc"
              :class="{ on: accHi(i, j) }"
              :style="{ '--fill': 0.15 + (0.6 * (kk + 1)) / BK }"
              @mouseenter="hi = { a: i, b: j }"
              @click="hi = { a: i, b: j }"
            />
          </template>
        </div>
        <div class="cap">
          <code v-if="caption">{{ caption }}</code>
          <span v-else class="muted">鼠标停在 a、b 或 acc 上；acc 越深，表示已经累加了越多轮 kk</span>
        </div>
      </div>
    </div>

    <div class="instr">
      <div class="flabel">这个线程每个 kk 发的指令</div>
      <div class="boxes">
        <i v-for="x in cost.lds" :key="'l' + x" class="ld" />
        <i v-for="x in cost.fma" :key="'f' + x" class="fma" />
      </div>
      <div class="legend muted"><span><i class="ld" />shared memory load</span><span><i class="fma" />FMA</span></div>
    </div>

    <div class="grid stats">
      <StatCard label="每个 kk 的 load / FMA" :value="`${cost.lds} / ${cost.fma}`" :sub="vec === 4 ? `float4：${t} 个数 ${t / 4} 条指令` : `读 ${t} + ${t} 个数，做 ${t} × ${t} 次 MAC`" />
      <StatCard label="FMA 占发射槽" :value="pct(cost.fmaShare)" :tone="cost.fmaShare > 0.9 ? 'good' : cost.fmaShare > 0.7 ? 'default' : 'warn'" sub="warp 调度器每周期发 1 条" />
      <StatCard label="每线程寄存器" :value="`${cost.regs}`" :sub="`acc ${t * t} + a ${t} + b ${t}`" />
      <StatCard label="整个 block 每个 kk 读 smem" :value="`${smem.reads} 个数`" :sub="`其中不同的只有 ${smem.distinct} 个`" />
    </div>

    <p class="muted note">
      每个 kk，所有线程都读 As 的第 kk 列和 Bs 的第 kk 行（浅色），选中的线程只取其中自己那 TM 个和 TN 个（深色），读进寄存器 a、b，再做一次外积累加到 acc。
      a[i] 被 acc 的一整行用 TN 次，b[j] 被一整列用 TM 次：TM、TN 越大，一次 load 换来的 FMA 越多，但 acc 占的寄存器按 TM·TN 涨。
      float4 一条指令读 4 个连续的数：b 在 Bs 的一行里本来就连续，a 要连续得把 As 在 shared memory 里转置存（As[kk][m]）。
      这里的 16 × 16 是缩小版，实际常见 BM = BN = 128、256 个线程、每线程 8 × 8。
    </p>
  </div>
</template>

<style scoped>
.controls { margin-bottom: 12px; }
.field { display: flex; flex-direction: column; gap: 3px; }
.flabel { font-size: 12px; color: var(--wg-muted); }
.small { font-size: 12px; }
.widget .toggle-group button:disabled { opacity: 0.4; cursor: not-allowed; }

.body { display: flex; flex-wrap: wrap; gap: 16px 28px; justify-content: center; align-items: flex-start; }
.smem { width: 100%; max-width: 340px; height: auto; flex: 1 1 260px; }
.cell { fill: var(--vp-c-bg); stroke: var(--wg-border); stroke-width: 0.6; transition: fill 0.25s; }
.cell.past { fill: color-mix(in srgb, var(--wg-muted) 14%, transparent); }
.cell.now { fill: color-mix(in srgb, #f59e0b 28%, transparent); }
.cell.sel { fill: #f59e0b; }
.cell.c { cursor: pointer; }
.cell.c.mine { fill: var(--vp-c-brand-1); }
.cell.c:hover { stroke: var(--vp-c-brand-1); stroke-width: 1.4; }
.tline { stroke: var(--wg-text); stroke-width: 1.2; opacity: 0.55; pointer-events: none; }
.cframe { fill: none; stroke: var(--wg-text); stroke-width: 1.2; opacity: 0.55; pointer-events: none; }
.lbl { font-size: 9.5px; font-weight: 600; }

.regs { flex: 0 1 auto; display: flex; flex-direction: column; gap: 6px; align-items: center; }
.rtitle { font-size: 13px; font-weight: 600; }
.rgrid { display: grid; gap: 2px; }
.r { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; font-family: var(--vp-font-family-mono); font-size: 10px; border-radius: 3px; border: 1px solid var(--wg-border); cursor: pointer; }
.r.a, .r.b { background: color-mix(in srgb, #f59e0b 35%, var(--vp-c-bg)); }
.r.acc { background: color-mix(in srgb, var(--vp-c-brand-1) calc(var(--fill) * 100%), var(--vp-c-bg)); }
.r.on { outline: 2px solid #ef4444; outline-offset: -1px; }
.r.a.on, .r.b.on { background: #f59e0b; color: #fff; }
.cap { font-size: 12px; min-height: 2.8em; max-width: 300px; text-align: center; line-height: 1.5; }
.cap code { font-size: 12px; white-space: normal; }

.instr { margin-top: 14px; }
.boxes { display: flex; flex-wrap: wrap; gap: 2px; margin: 6px 0; }
.boxes i, .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }
.ld { background: #f59e0b; }
.fma { background: var(--vp-c-brand-1); }
.legend { display: flex; gap: 16px; font-size: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.stats { margin: 12px 0 4px; }
.note { font-size: 12.5px; line-height: 1.75; margin-top: 10px; }
</style>
