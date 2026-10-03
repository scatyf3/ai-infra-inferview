<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { cdiv, launchOrder, tileRange, vectorPrograms, waveFootprint } from '@lib/triton'
import NumberField from './ui/NumberField.vue'
import StatCard from './ui/StatCard.vue'

const props = withDefaults(defineProps<{ mode?: 'vector' | 'matmul' }>(), { mode: 'vector' })
const mode = ref(props.mode)
const clampInt = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(Number(v) || lo)))
const rangeText = (start: number, end: number) => (end - start <= 1 ? `${start}` : `${start} … ${end - 1}`)

// ---------------- 1D：program / offs / mask ----------------
const nRaw = ref(50)
const block = ref(16)
const view = ref<'triton' | 'cuda'>('triton')
const n = computed(() => clampInt(nRaw.value, 1, 160))
const programs = computed(() => vectorPrograms(n.value, block.value))
const selPid = ref(2)
watch(programs, (ps) => { if (selPid.value >= ps.length) selPid.value = ps.length - 1 })
const sel = computed(() => programs.value[selPid.value])
const maskedLanes = computed(() => programs.value.length * block.value - n.value)
const progPalette = ['#3b82f6', '#f97316', '#22c55e', '#ec4899', '#8b5cf6', '#14b8a6']
const progColor = (pid: number) => progPalette[pid % progPalette.length]

// ---------------- 2D：matmul tile / K 循环 / launch 顺序 ----------------
const mRaw = ref(80), nmRaw = ref(128), kRaw = ref(64)
const BM = ref(32), BN = ref(32), BK = ref(16)
const blockRefs = [['BLOCK_M', BM], ['BLOCK_N', BN], ['BLOCK_K', BK]] as const
const groupM = ref(2)
const waveRaw = ref(4)
const M = computed(() => clampInt(mRaw.value, 8, 256))
const N = computed(() => clampInt(nmRaw.value, 8, 256))
const K = computed(() => clampInt(kRaw.value, 8, 256))
const numPidM = computed(() => cdiv(M.value, BM.value))
const numPidN = computed(() => cdiv(N.value, BN.value))
const kSteps = computed(() => cdiv(K.value, BK.value))
const order = computed(() => launchOrder(numPidM.value, numPidN.value, groupM.value))
const total = computed(() => order.value.length)
const wave = computed(() => clampInt(waveRaw.value, 1, total.value))

const pid = ref(0)
const k = ref(0)
watch([total, kSteps], () => {
  if (pid.value >= total.value) pid.value = 0
  if (k.value >= kSteps.value) k.value = 0
})
const cur = computed(() => order.value[pid.value])
const rowR = computed(() => tileRange(cur.value.pidM, BM.value, M.value))
const colR = computed(() => tileRange(cur.value.pidN, BN.value, N.value))
const kR = computed(() => tileRange(k.value, BK.value, K.value))

const waveStart = computed(() => Math.floor(pid.value / wave.value) * wave.value)
const waveTiles = computed(() => order.value.slice(waveStart.value, waveStart.value + wave.value))
const inWave = computed(() => new Set(waveTiles.value.map((t) => t.pid)))
const waveRows = computed(() => [...new Set(waveTiles.value.map((t) => t.pidM))])
const waveCols = computed(() => [...new Set(waveTiles.value.map((t) => t.pidN))])
const fp = computed(() => waveFootprint(order.value, waveStart.value, wave.value))
const fpRow = computed(() => waveFootprint(launchOrder(numPidM.value, numPidN.value, 1), waveStart.value, wave.value))

// SVG 布局：B 在右上，A 在左下，C 在右下（经典的 matmul 摆法），越界的 tile 也要画得下
const geo = computed(() => {
  const Kp = kSteps.value * BK.value, Mp = numPidM.value * BM.value, Np = numPidN.value * BN.value
  const s = Math.min(3.2, 540 / (Kp + Np))
  const gap = 16, pad = 20
  const ax = 4, bx = ax + Kp * s + gap
  const by = pad, ay = by + Kp * s + gap
  return { s, ax, ay, bx, by, w: bx + Np * s + 4, h: ay + Mp * s + 22 }
})
const rect = (x0: number, y0: number, r0: number, r1: number, c0: number, c1: number) => {
  const s = geo.value.s
  return { x: x0 + c0 * s, y: y0 + r0 * s, width: Math.max(0, (c1 - c0) * s), height: Math.max(0, (r1 - r0) * s) }
}
const tileFontPx = computed(() => Math.min(11, Math.min(BM.value, BN.value) * geo.value.s * 0.42))

// 播放：先走完当前 program 的 K 循环，再按 launch 顺序换下一个 program
const timer = ref<number | null>(null)
function next() {
  if (k.value + 1 < kSteps.value) return void k.value++
  k.value = 0
  pid.value = (pid.value + 1) % total.value
}
function prev() {
  if (k.value > 0) return void k.value--
  pid.value = (pid.value - 1 + total.value) % total.value
  k.value = kSteps.value - 1
}
function pick(p: number) { pid.value = p; k.value = 0 }
function stop() { if (timer.value !== null) { clearInterval(timer.value); timer.value = null } }
function toggle() { if (timer.value !== null) return stop(); timer.value = window.setInterval(next, 650) }
function reset() { stop(); pid.value = 0; k.value = 0 }
onUnmounted(stop)
</script>

<template>
  <div class="widget tpv">
    <div class="row head">
      <h4>Triton 编程模型：你写到 program 这一级</h4>
      <div class="toggle-group">
        <button :class="{ active: mode === 'vector' }" @click="mode = 'vector'">1D · offs / mask</button>
        <button :class="{ active: mode === 'matmul' }" @click="mode = 'matmul'">2D · matmul tile</button>
      </div>
    </div>

    <!-- ========== 1D ========== -->
    <template v-if="mode === 'vector'">
      <div class="row controls">
        <NumberField v-model="nRaw" label="n（元素数）" :min="1" :max="160" width="110px" />
        <div class="field">
          <span class="flabel">BLOCK（constexpr）</span>
          <div class="toggle-group">
            <button v-for="b in [4, 8, 16, 32]" :key="b" :class="{ active: block === b }" @click="block = b">{{ b }}</button>
          </div>
        </div>
        <div class="field">
          <span class="flabel">视角</span>
          <div class="toggle-group">
            <button :class="{ active: view === 'triton' }" @click="view = 'triton'">Triton</button>
            <button :class="{ active: view === 'cuda' }" @click="view = 'cuda'">CUDA</button>
          </div>
        </div>
      </div>

      <div class="grid stats">
        <StatCard label="grid" :value="`(${programs.length},)`" :sub="`cdiv(${n}, ${block})`" />
        <StatCard label="被 mask 的 lane" :value="String(maskedLanes)" :sub="maskedLanes ? '只出现在最后一个 program' : 'n 恰好整除 BLOCK'" :tone="maskedLanes ? 'warn' : 'good'" />
        <StatCard label="你要管的单位" :value="view === 'triton' ? `${programs.length} 个 program` : `${programs.length * block} 个线程`" :sub="view === 'triton' ? '每个 program 拿一个下标向量' : '每个线程算自己的标量下标'" />
      </div>

      <div class="lanes" :class="view">
        <div
          v-for="p in programs"
          :key="p.pid"
          class="prog"
          :class="{ sel: p.pid === selPid }"
          :style="{ '--pc': progColor(p.pid) }"
          @click="selPid = p.pid"
        >
          <div class="plabel">{{ view === 'triton' ? `pid ${p.pid}` : `blockIdx.x = ${p.pid}` }}</div>
          <div class="cells">
            <span v-for="(o, i) in p.offs" :key="i" class="cell" :class="{ off: !p.mask[i] }" :title="p.mask[i] ? `offs=${o}` : `offs=${o} ≥ n，被 mask`">
              {{ view === 'triton' ? o : `t${i}` }}
            </span>
          </div>
        </div>
      </div>

      <div class="code" v-if="sel">
        <template v-if="view === 'triton'">
          <div class="ln"><code>pid  = tl.program_id(0)</code><span>{{ sel.pid }}</span></div>
          <div class="ln"><code>offs = pid * BLOCK + tl.arange(0, BLOCK)</code><span>[{{ rangeText(sel.offs[0], sel.offs[0] + block) }}]，长度 {{ block }} 的向量</span></div>
          <div class="ln"><code>mask = offs &lt; n</code><span>{{ sel.valid }} 个 True<template v-if="block - sel.valid"> · {{ block - sel.valid }} 个 False</template></span></div>
          <div class="ln"><code>x = tl.load(x_ptr + offs, mask=mask)</code><span>整块一起读；False 的 lane 不访存</span></div>
          <div class="ln"><code>tl.store(out_ptr + offs, x + y, mask=mask)</code><span>块内哪个线程算哪几个元素：编译器定</span></div>
        </template>
        <template v-else>
          <div class="ln"><code>int b = blockIdx.x;</code><span>{{ sel.pid }}</span></div>
          <div class="ln"><code>int t = threadIdx.x;</code><span>0 … {{ block - 1 }}，每个线程各跑一遍这段代码</span></div>
          <div class="ln"><code>int i = b * blockDim.x + t;</code><span>{{ rangeText(sel.offs[0], sel.offs[0] + block) }}，每线程一个标量</span></div>
          <div class="ln"><code>if (i &lt; n) out[i] = x[i] + y[i];</code><span>{{ block - sel.valid }} 个线程空转</span></div>
          <div class="ln"><code>// shared mem / __syncthreads / 合并访存</code><span>都要自己写</span></div>
        </template>
      </div>

      <p class="muted note">
        点任意一块看它的变量。Triton 视角下一块是<b>一个 program 手里的一整个向量</b>，你写的是对这个向量的操作；
        CUDA 视角下同样的数据拆成 {{ block }} 个线程，每个线程算自己的标量下标。
        Triton 里没有 <code>threadIdx</code>：一个 program 用 <code>num_warps × 32</code> 个线程执行，每个线程分到哪几个元素（layout）由编译器决定。
      </p>
    </template>

    <!-- ========== 2D matmul ========== -->
    <template v-else>
      <div class="row controls">
        <NumberField v-model="mRaw" label="M" :min="8" :max="256" :step="16" width="72px" />
        <NumberField v-model="nmRaw" label="N" :min="8" :max="256" :step="16" width="72px" />
        <NumberField v-model="kRaw" label="K" :min="8" :max="256" :step="16" width="72px" />
        <div class="field" v-for="[name, r] in blockRefs" :key="name">
          <span class="flabel">{{ name }}</span>
          <div class="toggle-group">
            <button v-for="b in [16, 32, 64]" :key="b" :class="{ active: r.value === b }" @click="r.value = b; reset()">{{ b }}</button>
          </div>
        </div>
        <div class="field">
          <span class="flabel">GROUP_M（1 = 行优先）</span>
          <div class="toggle-group">
            <button v-for="g in [1, 2, 4]" :key="g" :class="{ active: groupM === g }" @click="groupM = g">{{ g }}</button>
          </div>
        </div>
        <NumberField v-model="waveRaw" label="一波并发 program" :min="1" :max="total" width="120px" hint="同时驻留在 SM 上的 program 数（≈ SM 数）。同一波读到的 A/B 条带可以在 L2 里复用" />
      </div>

      <div class="row controls">
        <button class="btn" @click="prev">上一步</button>
        <button class="btn" @click="next">下一步</button>
        <button class="btn" @click="toggle">{{ timer === null ? '播放' : '暂停' }}</button>
        <button class="btn" @click="reset">重置</button>
        <span class="muted small">点 C 里任意 tile 切换 program；数字是 launch 顺序（pid）</span>
      </div>

      <div class="mm">
        <svg :viewBox="`0 0 ${geo.w} ${geo.h}`" class="mmsvg" role="img" aria-label="matmul tile 划分">
          <!-- 矩阵底 -->
          <rect class="mat" v-bind="rect(geo.ax, geo.ay, 0, M, 0, K)" />
          <rect class="mat" v-bind="rect(geo.bx, geo.by, 0, K, 0, N)" />
          <rect class="mat" v-bind="rect(geo.bx, geo.ay, 0, M, 0, N)" />

          <!-- 同一波其他 program 要读的条带 -->
          <rect v-for="r in waveRows" :key="'wr' + r" class="wstrip" v-bind="rect(geo.ax, geo.ay, r * BM, Math.min((r + 1) * BM, M), 0, K)" />
          <rect v-for="c in waveCols" :key="'wc' + c" class="wstrip" v-bind="rect(geo.bx, geo.by, 0, K, c * BN, Math.min((c + 1) * BN, N))" />

          <!-- 当前 program 的 A 行条带、B 列条带，以及第 k 步的 tile -->
          <rect class="strip" v-bind="rect(geo.ax, geo.ay, rowR.start, rowR.end, 0, K)" />
          <rect class="strip" v-bind="rect(geo.bx, geo.by, 0, K, colR.start, colR.end)" />
          <rect class="ktile" v-bind="rect(geo.ax, geo.ay, rowR.start, rowR.end, kR.start, kR.end)" />
          <rect class="ktile" v-bind="rect(geo.bx, geo.by, kR.start, kR.end, colR.start, colR.end)" />
          <rect v-if="rowR.masked || kR.masked" class="ghost" v-bind="rect(geo.ax, geo.ay, rowR.start, rowR.start + BM, kR.start, kR.start + BK)" />
          <rect v-if="colR.masked || kR.masked" class="ghost" v-bind="rect(geo.bx, geo.by, kR.start, kR.start + BK, colR.start, colR.start + BN)" />

          <!-- K 维切分线 -->
          <line v-for="j in kSteps - 1" :key="'ka' + j" class="kline" :x1="geo.ax + j * BK * geo.s" :x2="geo.ax + j * BK * geo.s" :y1="geo.ay" :y2="geo.ay + M * geo.s" />
          <line v-for="j in kSteps - 1" :key="'kb' + j" class="kline" :y1="geo.by + j * BK * geo.s" :y2="geo.by + j * BK * geo.s" :x1="geo.bx" :x2="geo.bx + N * geo.s" />

          <!-- C 的 tile = program -->
          <g v-for="t in order" :key="'t' + t.pid" class="tile" :class="{ sel: t.pid === pid, inwave: inWave.has(t.pid) }" @click="pick(t.pid)">
            <rect
              v-bind="rect(geo.bx, geo.ay, t.pidM * BM, Math.min((t.pidM + 1) * BM, M), t.pidN * BN, Math.min((t.pidN + 1) * BN, N))"
              :style="t.pid === pid ? { fillOpacity: 0.3 + 0.7 * (k + 1) / kSteps } : {}"
            />
            <text
              :x="geo.bx + (t.pidN * BN + Math.min(BN, N - t.pidN * BN) / 2) * geo.s"
              :y="geo.ay + (t.pidM * BM + Math.min(BM, M - t.pidM * BM) / 2) * geo.s"
              :font-size="tileFontPx"
              dominant-baseline="central"
              text-anchor="middle"
            >{{ t.pid }}</text>
          </g>
          <rect v-if="rowR.masked || colR.masked" class="ghost" v-bind="rect(geo.bx, geo.ay, rowR.start, rowR.start + BM, colR.start, colR.start + BN)" />

          <text class="lbl" :x="geo.ax" :y="geo.ay - 5">A（{{ M }}×{{ K }}）</text>
          <text class="lbl" :x="geo.bx" :y="geo.by - 6">B（{{ K }}×{{ N }}）</text>
          <text class="lbl" :x="geo.bx" :y="geo.h - 4">C（{{ M }}×{{ N }}） · grid = ({{ total }},)</text>
        </svg>

        <div class="code mmcode">
          <div class="ln"><code>pid = tl.program_id(0)</code><span>{{ pid }} / {{ total }}</span></div>
          <div class="ln"><code>pid_m, pid_n = swizzle(pid, GROUP_M)</code><span>({{ cur.pidM }}, {{ cur.pidN }})</span></div>
          <div class="ln"><code>offs_m = pid_m*BM + tl.arange(0, BM)</code><span>{{ rangeText(rowR.start, rowR.start + BM) }}<em v-if="rowR.masked"> · 后 {{ rowR.masked }} 行 mask</em></span></div>
          <div class="ln"><code>offs_n = pid_n*BN + tl.arange(0, BN)</code><span>{{ rangeText(colR.start, colR.start + BN) }}<em v-if="colR.masked"> · 后 {{ colR.masked }} 列 mask</em></span></div>
          <div class="ln"><code>acc = tl.zeros((BM, BN), tl.float32)</code><span>累加器在寄存器里</span></div>
          <div class="ln hl"><code>for k in range(tl.cdiv(K, BK)):</code><span>k = {{ k }} / {{ kSteps }}</span></div>
          <div class="ln hl"><code>&nbsp;&nbsp;offs_k = k*BK + tl.arange(0, BK)</code><span>{{ rangeText(kR.start, kR.start + BK) }}<em v-if="kR.masked"> · 后 {{ kR.masked }} 个 mask</em></span></div>
          <div class="ln hl"><code>&nbsp;&nbsp;a = tl.load(a_ptrs, mask=…)</code><span>({{ BM }}, {{ BK }})</span></div>
          <div class="ln hl"><code>&nbsp;&nbsp;b = tl.load(b_ptrs, mask=…)</code><span>({{ BK }}, {{ BN }})</span></div>
          <div class="ln hl"><code>&nbsp;&nbsp;acc += tl.dot(a, b)</code><span>tensor core</span></div>
          <div class="ln"><code>tl.store(c_ptrs, acc, mask=…)</code><span>每个 C 元素只写一次</span></div>
          <div class="ptrs">a_ptrs = a + offs_m[:, None]*stride_am + offs_k[None, :]*stride_ak<br />每步 a_ptrs += BK*stride_ak，b_ptrs += BK*stride_bk</div>
        </div>
      </div>

      <div class="grid stats">
        <StatCard label="program 数" :value="`${numPidM} × ${numPidN}`" :sub="`每个 program 跑 ${kSteps} 步 K 循环`" />
        <StatCard
          label="这一波要读的条带"
          :value="`${fp.rowsA} + ${fp.colsB}`"
          :sub="`A 行条带 + B 列条带（pid ${waveStart}–${waveStart + fp.programs - 1}）`"
          :tone="fp.rowsA + fp.colsB < fpRow.rowsA + fpRow.colsB ? 'good' : 'default'"
        />
        <StatCard label="行优先时" :value="`${fpRow.rowsA} + ${fpRow.colsB}`" sub="同一波、GROUP_M = 1" />
      </div>

      <p class="muted note">
        <b>一个 program 负责 C 的一个 tile</b>：沿 K 一步步把 A 的 (BM, BK) 块和 B 的 (BK, BN) 块读进来，<code>tl.dot</code> 累到寄存器里的 acc，最后写回一次。
        蓝色是当前 program 的读取，深蓝是第 k 步正在读的块，红色虚线是越界、靠 mask 补齐的部分，橙色是同一波其他 program 要读的条带。
        GROUP_M 改的只是 pid → (pid_m, pid_n) 的映射：同一波 program 挤在 GROUP_M 行里，读到的 A/B 条带更少、更容易命中 L2。总计算量不变，HBM 流量变小。
      </p>
    </template>
  </div>
</template>

<style scoped>
.head { justify-content: space-between; align-items: center; margin-bottom: 12px; }
.head h4 { margin: 0; }
.controls { margin-bottom: 10px; }
.field { display: flex; flex-direction: column; gap: 3px; }
.flabel { font-size: 12px; color: var(--wg-muted); }
.small { font-size: 12px; }
.stats { margin: 10px 0; }

/* 1D */
.lanes { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 12px; }
.prog { border: 2px solid var(--pc); border-radius: 8px; padding: 4px 6px 6px; cursor: pointer; background: var(--vp-c-bg); transition: box-shadow 0.15s; }
.prog.sel { box-shadow: 0 0 0 3px color-mix(in srgb, var(--pc) 40%, transparent); }
.plabel { font-size: 11px; font-family: var(--vp-font-family-mono); color: var(--pc); font-weight: 600; margin-bottom: 4px; }
.cells { display: flex; flex-wrap: wrap; max-width: 330px; }
.cell { min-width: 20px; height: 20px; padding: 0 2px; font-size: 10px; font-family: var(--vp-font-family-mono); display: inline-flex; align-items: center; justify-content: center; background: color-mix(in srgb, var(--pc) 22%, var(--vp-c-bg)); }
/* Triton：一整条向量；CUDA：一格一个线程 */
.lanes.triton .cells { border-radius: 4px; overflow: hidden; }
.lanes.cuda .cells { gap: 2px; }
.lanes.cuda .cell { border-radius: 50%; width: 22px; height: 22px; font-size: 9px; }
.cell.off { background: repeating-linear-gradient(45deg, transparent 0 3px, color-mix(in srgb, #ef4444 35%, transparent) 3px 5px); color: var(--wg-muted); text-decoration: line-through; }

/* 代码面板 */
.code { border: 1px solid var(--wg-border); border-radius: 8px; background: var(--vp-c-bg); padding: 8px 10px; font-size: 12px; }
.ln { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, auto); gap: 4px 14px; padding: 2px 0; align-items: baseline; }
.ln code { background: none; padding: 0; font-size: 12px; white-space: pre-wrap; }
.ln span { color: var(--vp-c-brand-1); font-family: var(--vp-font-family-mono); text-align: right; }
.ln em { color: #ef4444; font-style: normal; }
.ln.hl { background: color-mix(in srgb, var(--vp-c-brand-1) 7%, transparent); margin: 0 -10px; padding: 2px 10px; }
.ptrs { margin-top: 6px; padding-top: 6px; border-top: 1px dashed var(--wg-border); font-family: var(--vp-font-family-mono); font-size: 11px; color: var(--wg-muted); line-height: 1.7; }

/* 2D */
.mm { display: flex; flex-direction: column; gap: 12px; }
.mmsvg { width: 100%; max-width: 560px; height: auto; align-self: center; }
.mat { fill: var(--vp-c-bg); stroke: var(--wg-muted); stroke-width: 1; }
.wstrip { fill: color-mix(in srgb, #f59e0b 22%, transparent); }
.strip { fill: color-mix(in srgb, var(--vp-c-brand-1) 22%, transparent); }
.ktile { fill: var(--vp-c-brand-1); fill-opacity: 0.75; transition: x 0.2s, y 0.2s; }
.ghost { fill: none; stroke: #ef4444; stroke-width: 1.2; stroke-dasharray: 3 2; }
.kline { stroke: var(--wg-muted); stroke-width: 0.6; stroke-dasharray: 2 2; opacity: 0.6; }
.tile { cursor: pointer; }
.tile rect { fill: transparent; stroke: var(--wg-muted); stroke-width: 0.8; }
.tile.inwave rect { fill: color-mix(in srgb, #f59e0b 22%, transparent); }
.tile.sel rect { fill: var(--vp-c-brand-1); stroke: var(--vp-c-brand-1); }
.tile:hover rect { stroke: var(--vp-c-brand-1); stroke-width: 1.6; }
.tile text { font-family: var(--vp-font-family-mono); pointer-events: none; }
.widget .tile.sel text { fill: #fff; font-weight: 700; }
.lbl { font-size: 11px; font-weight: 600; }
.note { font-size: 12.5px; line-height: 1.75; margin-top: 12px; }
</style>
