<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { BANKS, WORD_BYTES, canSwizzle, gcd, laneCells, warpAccess, type Access } from '@lib/bank'
import StatCard from './ui/StatCard.vue'

// 真实规模：32 个 bank、一个 warp 32 个线程、tile 32 行。pad 表示多出来的列只是 padding，不存数据
interface Preset { pitch: number; pad?: boolean }
const PRESETS: Record<2 | 4, Preset[]> = {
  4: [{ pitch: 32 }, { pitch: 33, pad: true }, { pitch: 48 }, { pitch: 64 }],
  2: [{ pitch: 64 }, { pitch: 65, pad: true }, { pitch: 66, pad: true }, { pitch: 96 }],
}
const ROW_BYTES = BANKS * WORD_BYTES

const props = withDefaults(defineProps<{ access?: Access }>(), { access: 'col' })
const elemBytes = ref<2 | 4>(4)
const pi = ref(0)
const access = ref<Access>(props.access)
const k = ref(0)
const swz = ref(false)

const ty = computed(() => (elemBytes.value === 4 ? 'float' : 'half'))
const preset = computed(() => PRESETS[elemBytes.value][pi.value])
const pitch = computed(() => preset.value.pitch)
const rowBytes = computed(() => pitch.value * elemBytes.value)
const swizzleOk = computed(() => canSwizzle({ elemBytes: elemBytes.value, pitch: pitch.value }))
const layout = computed(() => ({ elemBytes: elemBytes.value, pitch: pitch.value, swizzle: swz.value && swizzleOk.value }))
const kMax = computed(() => (access.value === 'row' ? 31 : ROW_BYTES / elemBytes.value - 1))
watch(kMax, (m) => { if (k.value > m) k.value = m })

const acc = computed(() => warpAccess(layout.value, laneCells(access.value, k.value)))
const used = computed(() => acc.value.banks.filter((n) => n > 0).length)

// bank 表：每行 32 个字（128 B），列就是 bank
const nRows = computed(() => Math.ceil((32 * rowBytes.value) / ROW_BYTES))
const lanesAt = computed(() => {
  const m = new Map<number, number[]>()
  acc.value.words.forEach((w, t) => m.set(w, [...(m.get(w) ?? []), t]))
  return m
})
const rows = computed(() =>
  Array.from({ length: nRows.value }, (_, R) =>
    Array.from({ length: BANKS }, (_, b) => {
      const w = R * BANKS + b, byte = w * WORD_BYTES
      const lr = Math.floor(byte / rowBytes.value)
      return {
        w, lr,
        out: lr >= 32,
        pad: !!preset.value.pad && byte - lr * rowBytes.value >= ROW_BYTES,
        lanes: lanesAt.value.get(w),
      }
    }),
  ),
)
const laneColor = (t: number) => `hsl(${(t * 360) / BANKS} 70% 48%)`

const codeLines = computed(() => {
  const decl = `__shared__ ${ty.value} tile[32][${pitch.value}];`
  const half = elemBytes.value === 2
  let read: string
  if (!layout.value.swizzle) read = access.value === 'row' ? `v = tile[${k.value}][t];` : `v = tile[t][${k.value}];`
  else if (access.value === 'row')
    read = half ? `v = tile[${k.value}][(((t >> 1) ^ ${k.value}) << 1) | (t & 1)];` : `v = tile[${k.value}][t ^ ${k.value}];`
  else read = half ? `v = tile[t][(((${k.value} >> 1) ^ t) << 1) | ${k.value & 1}];` : `v = tile[t][${k.value} ^ t];`
  return [decl, `${read}   // t = lane 0..31`]
})

const explain = computed(() => {
  const n = acc.value.ways
  const verdict = n === 1 ? '32 个线程各占一个 bank（或同一个字广播），一次完成。' : `最挤的 bank 上排了 ${n} 个不同的字，${n} 路冲突。`
  if (layout.value.swizzle)
    return `swizzle 只在一行内部挪位置：第 r 行的第 j 个字存到第 j ^ r 个位置。同一列在不同行里被换到不同的 bank，不用 padding。${verdict}`
  if (access.value === 'row')
    return elemBytes.value === 2
      ? `相邻两个 lane 读同一个字里的两个 half，算广播；32 个 lane 只用到 16 个字、16 个 bank。${verdict}`
      : `相邻 lane 的地址差 1 个字，正好铺满一行 32 个 bank。${verdict}`
  const s = rowBytes.value / WORD_BYTES
  if (Number.isInteger(s)) {
    const g = gcd(s, BANKS)
    return `相邻 lane 差一整行 = ${s} 个字，lane t 落在 bank (${s}t + ${Math.floor((k.value * elemBytes.value) / WORD_BYTES)}) mod 32。gcd(${s}, 32) = ${g}，所以只用到 32 / ${g} = ${BANKS / g} 个 bank。${verdict}`
  }
  return `一行 ${rowBytes.value} B = ${s} 个字，不是整数，相邻两行只错开半个字：lane 2m 和 2m+1 落在同一个 bank 的不同字上。${verdict}`
})
</script>

<template>
  <div class="widget bc">
    <div class="row head">
      <h4>shared memory：一个 warp 的一次访问落在哪些 bank</h4>
    </div>

    <div class="row controls">
      <div class="toggle-group">
        <button :class="{ active: elemBytes === 4 }" @click="elemBytes = 4">float</button>
        <button :class="{ active: elemBytes === 2 }" @click="elemBytes = 2">half</button>
      </div>
      <div class="toggle-group">
        <button v-for="(p, i) in PRESETS[elemBytes]" :key="p.pitch" :class="{ active: pi === i }" @click="pi = i">[32][{{ p.pitch }}]</button>
      </div>
    </div>
    <div class="row controls">
      <div class="toggle-group">
        <button :class="{ active: access === 'row' }" @click="access = 'row'">按行 tile[k][t]</button>
        <button :class="{ active: access === 'col' }" @click="access = 'col'">按列 tile[t][k]</button>
      </div>
      <label class="small">k = <input v-model.number="k" type="range" min="0" :max="kMax" /> {{ k }}</label>
      <label class="small" :class="{ muted: !swizzleOk }" :title="swizzleOk ? '' : '一行要正好是 128 B 的整数倍'">
        <input v-model="swz" type="checkbox" :disabled="!swizzleOk" /> XOR swizzle
      </label>
    </div>

    <div class="code">
      <div v-for="l in codeLines" :key="l" class="ln"><code>{{ l }}</code></div>
    </div>
    <div class="steptext">{{ explain }}</div>

    <div class="table">
      <div class="g hdr">
        <div class="rl">行</div>
        <div v-for="b in BANKS" :key="'h' + b" class="bl">{{ (b - 1) % 8 === 0 || b === BANKS ? b - 1 : '' }}</div>
      </div>
      <div class="scroll">
        <div v-for="(cells, R) in rows" :key="'r' + R" class="g">
          <div class="rl">{{ R }}</div>
          <div
            v-for="c in cells"
            :key="c.w"
            class="c"
            :class="{ odd: !c.out && c.lr % 2 === 1, pad: c.pad, out: c.out, hit: c.lanes }"
            :style="c.lanes ? { background: laneColor(c.lanes[0]) } : {}"
            :title="c.lanes ? `字 ${c.w} · bank ${c.w % BANKS} · lane ${c.lanes.join(', ')}` : `字 ${c.w} · bank ${c.w % BANKS}${c.out ? '' : ` · tile 第 ${c.lr} 行`}`"
          >{{ c.lanes ? c.lanes[0] : '' }}</div>
        </div>
      </div>
      <div class="g hist">
        <div class="rl">次数</div>
        <div v-for="(n, b) in acc.banks" :key="'n' + b" class="bank" :class="{ hot: n > 1 }">
          <span v-if="n">{{ n }}</span>
          <div class="bar" :style="{ height: n ? `max(2px, ${(n / BANKS) * 100}%)` : '0' }" />
        </div>
      </div>
    </div>

    <div class="grid stats">
      <StatCard
        label="这次访问"
        :value="acc.ways > 1 ? `${acc.ways} 路冲突` : '无冲突'"
        :sub="`要分 ${acc.ways} 次完成，有效带宽 1/${acc.ways}`"
        :tone="acc.ways > 1 ? 'bad' : 'good'"
      />
      <StatCard label="用到的 bank" :value="`${used} / ${BANKS}`" :sub="`${new Set(acc.words).size} 个不同的字`" />
      <StatCard
        label="这块 tile 占的空间"
        :value="`${32 * rowBytes} B`"
        :sub="preset.pad ? `其中 padding ${32 * (rowBytes - ROW_BYTES)} B` : '没有 padding'"
      />
    </div>

    <p class="muted note">
      表格就是 shared memory 的 bank 表：每行 128 B = 32 个字，第 b 列是 bank b。彩色格是这次被访问的字，数字是 lane 号（同一个字被几个 lane 读时显示最小的那个，鼠标悬停看全部）。
      深浅交替的底色是 tile 的逻辑行，斜线格是 padding：一行正好 128 B 时逻辑行和 bank 表的行对齐，按列读就全挤在一列；加了 padding 后每个逻辑行往右错开，同一列就斜着铺开了。
      下面的柱子是每个 bank 上被访问的不同字数，最高的那根就是几路冲突。这里只算 4 B 及以下的标量访问；<code>float2</code> / <code>float4</code> 这类 8 B / 16 B 访问，硬件会把 warp 拆成 2 / 4 个 phase 分别算冲突。
    </p>
  </div>
</template>

<style scoped>
.head { margin-bottom: 8px; }
.head h4 { margin: 0; }
.controls { margin-bottom: 8px; align-items: center; gap: 12px; flex-wrap: wrap; }
.small { font-size: 12px; display: inline-flex; align-items: center; gap: 4px; }
.small input[type='range'] { width: 96px; }
.code { border: 1px solid var(--wg-border); border-radius: 8px; background: var(--vp-c-bg); padding: 6px 10px; font-size: 12px; margin: 4px 0 8px; overflow-x: auto; }
.ln { padding: 2px 0; white-space: pre; }
.ln code { background: none; padding: 0; font-size: 12px; }
.steptext { font-size: 13px; line-height: 1.6; margin-bottom: 10px; min-height: 3.2em; }

.table { border: 1px solid var(--wg-border); border-radius: 8px; background: var(--vp-c-bg); padding: 6px; }
.g { display: grid; grid-template-columns: 26px repeat(32, minmax(0, 1fr)); gap: 1px; }
.scroll { max-height: 420px; overflow-y: auto; display: flex; flex-direction: column; gap: 1px; }
.rl { font-size: 9px; font-family: var(--vp-font-family-mono); color: var(--wg-muted); text-align: right; padding-right: 4px; line-height: 13px; }
.bl { font-size: 9px; font-family: var(--vp-font-family-mono); color: var(--wg-muted); white-space: nowrap; overflow: visible; }
.hdr { margin-bottom: 2px; }
.c { height: 13px; background: color-mix(in srgb, var(--wg-muted) 8%, var(--vp-c-bg)); font-size: 7px; line-height: 13px; text-align: center; font-family: var(--vp-font-family-mono); color: #fff; overflow: hidden; }
.c.odd { background: color-mix(in srgb, var(--vp-c-brand-1) 18%, var(--vp-c-bg)); }
.c.pad { background: repeating-linear-gradient(45deg, transparent 0 2px, color-mix(in srgb, var(--wg-muted) 35%, transparent) 2px 3px); }
.c.out { background: transparent; }
.c.hit { font-weight: 600; }

.hist { height: 60px; margin-top: 6px; align-items: end; }
.hist .rl { align-self: end; }
.bank { height: 100%; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; font-size: 8px; font-family: var(--vp-font-family-mono); }
.bar { width: 100%; background: #22c55e; border-radius: 2px 2px 0 0; }
.bank.hot { color: #ef4444; }
.bank.hot .bar { background: #ef4444; }

.stats { margin: 14px 0 6px; }
.note { font-size: 12.5px; line-height: 1.75; margin-top: 10px; }
</style>
