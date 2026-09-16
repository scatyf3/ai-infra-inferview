<script setup lang="ts">
import { computed, ref } from 'vue'
import { formatShape, kvElemsPerTokenPerLayer, numel, shapeFlow, type Dims } from '@lib/shapes'
import type { AttentionKind, Phase } from '@lib/types'

const props = withDefaults(defineProps<{ variant?: AttentionKind; phase?: Phase }>(), { variant: 'gqa', phase: 'prefill' })
const variant = ref<AttentionKind>(props.variant)
const phase = ref<Phase>(props.phase)
const numeric = ref(true)

const dims = ref<Dims>({ B: 4, S: 2048, S_kv: 2048, d: 8192, H: 64, H_kv: 8, h_d: 128, d_ff: 28672, d_c: 512, d_r: 64 })

const effDims = computed<Dims>(() => {
  const d = { ...dims.value }
  if (variant.value === 'mha') d.H_kv = d.H
  if (variant.value === 'mqa') d.H_kv = 1
  d.h_d = Math.round(d.d / d.H)
  if (phase.value === 'prefill') d.S_kv = d.S
  return d
})

const steps = computed(() => shapeFlow({ variant: variant.value, phase: phase.value }))
const show = computed(() => (numeric.value ? effDims.value : undefined))

const kvCompare = computed(() =>
  (['mha', 'gqa', 'mqa', 'mla'] as AttentionKind[]).map((v) => {
    const d = { ...effDims.value }
    if (v === 'mha') d.H_kv = d.H
    if (v === 'mqa') d.H_kv = 1
    if (v === 'gqa') d.H_kv = dims.value.H_kv
    const elems = kvElemsPerTokenPerLayer(v, d)
    return { v, elems, bytes: elems * 2, ratio: elems / kvElemsPerTokenPerLayer('mha', { ...d, H_kv: d.H }) }
  }),
)

const variants: { v: AttentionKind; label: string }[] = [
  { v: 'mha', label: 'MHA' },
  { v: 'gqa', label: 'GQA' },
  { v: 'mqa', label: 'MQA' },
  { v: 'mla', label: 'MLA' },
]
const fmtElems = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : `${n}`)
</script>

<template>
  <div class="widget">
    <h4>Tensor Shape 流转：一层 Transformer</h4>
    <div class="row controls">
      <div class="toggle-group">
        <button v-for="o in variants" :key="o.v" :class="{ active: variant === o.v }" @click="variant = o.v">{{ o.label }}</button>
      </div>
      <div class="toggle-group">
        <button :class="{ active: phase === 'prefill' }" @click="phase = 'prefill'">prefill</button>
        <button :class="{ active: phase === 'decode' }" @click="phase = 'decode'">decode</button>
      </div>
      <label class="chk"><input type="checkbox" v-model="numeric" /> 代入数值</label>
    </div>

    <div v-if="numeric" class="row dims">
      <label v-for="k in (['B', 'S', 'S_kv', 'd', 'H', 'H_kv', 'd_ff'] as const)" :key="k" class="dim">
        <span>{{ k }}</span>
        <input type="number" v-model.number="dims[k]" :disabled="(k === 'H_kv' && variant !== 'gqa') || (k === 'S_kv' && phase === 'prefill')" />
      </label>
    </div>

    <ol class="steps">
      <li v-for="s in steps" :key="s.id" :class="[s.group, s.cache]">
        <div class="op">
          <span class="badge" :class="s.group">{{ s.group === 'attn' ? 'ATTN' : 'MLP' }}</span>
          <code>{{ s.op }}</code>
          <span v-if="s.cache" class="cache-tag" :class="s.cache">KV cache {{ s.cache === 'read' ? '读' : '写' }}</span>
        </div>
        <div class="shape">
          <span v-if="s.inputs.length" class="in">
            <code v-for="(i, k) in s.inputs" :key="k">{{ formatShape(i, show) }}</code>
          </span>
          <span class="arrow">→</span>
          <code class="out">{{ formatShape(s.output, show) }}</code>
          <span v-if="numeric" class="elems">{{ fmtElems(numel(s.output, effDims)) }} 元素</span>
        </div>
        <div v-if="s.note" class="note">{{ s.note }}</div>
      </li>
    </ol>

    <h4 class="mt">每层每 token 的 KV cache 元素数</h4>
    <table>
      <thead><tr><th>变体</th><th>元素/token/层</th><th>bf16 字节</th><th>相对 MHA</th></tr></thead>
      <tbody>
        <tr v-for="r in kvCompare" :key="r.v" :class="{ active: r.v === variant }">
          <td>{{ r.v.toUpperCase() }}</td>
          <td class="mono">{{ r.elems.toLocaleString() }}</td>
          <td class="mono">{{ (r.bytes / 1024).toFixed(2) }} KiB</td>
          <td class="mono">{{ (r.ratio * 100).toFixed(1) }}%</td>
        </tr>
      </tbody>
    </table>
    <p class="muted note">
      decode 每步都要把整个 KV cache 从 HBM 读一遍，所以 KV 每 token 的大小直接决定 decode 的访存量，也就决定了 TPOT 的下界。
      GQA 把 K/V 的 head 数从 H 降到 H_kv，MLA 只缓存一个 d_c 维 latent 加一个共享的 RoPE key。
    </p>
  </div>
</template>

<style scoped>
.controls { margin-bottom: 10px; }
.chk { display: flex; gap: 5px; align-items: center; font-size: 13px; cursor: pointer; }
.dims { gap: 8px 10px; margin-bottom: 12px; }
.dim { display: flex; align-items: center; gap: 4px; font-size: 12px; color: var(--wg-muted); }
.dim input { width: 72px; padding: 2px 6px; border: 1px solid var(--wg-border); border-radius: 4px; background: var(--vp-c-bg); color: var(--wg-text); font-family: var(--vp-font-family-mono); font-size: 12px; }
.dim input:disabled { opacity: 0.45; }
.steps { list-style: none; padding: 0; margin: 0; counter-reset: s; }
.steps li { position: relative; padding: 8px 10px 8px 34px; border-left: 3px solid var(--wg-border); margin: 0; background: var(--vp-c-bg); border-radius: 0 6px 6px 0; margin-bottom: 3px; }
.steps li.attn { border-left-color: #3b82f6; }
.steps li.mlp { border-left-color: #f97316; }
.steps li::before { counter-increment: s; content: counter(s); position: absolute; left: 10px; top: 9px; font-size: 11px; color: var(--wg-muted); font-family: var(--vp-font-family-mono); }
.steps li.read { background: color-mix(in srgb, #f59e0b 8%, var(--vp-c-bg)); }
.steps li.write { background: color-mix(in srgb, #22c55e 8%, var(--vp-c-bg)); }
.op { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.op code { font-size: 13px; background: none; padding: 0; color: var(--wg-text); }
.badge { font-size: 10px; font-weight: 700; padding: 1px 5px; border-radius: 3px; color: #fff; }
.badge.attn { background: #3b82f6; }
.badge.mlp { background: #f97316; }
.cache-tag { font-size: 11px; padding: 1px 6px; border-radius: 3px; }
.cache-tag.read { background: #f59e0b; color: #fff; }
.cache-tag.write { background: #22c55e; color: #fff; }
.shape { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 3px; font-size: 12px; }
.shape code { background: var(--vp-c-bg-soft); padding: 1px 6px; border-radius: 4px; font-size: 12px; }
.shape .out { background: var(--vp-c-brand-soft); }
.arrow { color: var(--wg-muted); }
.elems { color: var(--wg-muted); font-size: 11px; }
.note { font-size: 12px; color: var(--wg-muted); margin-top: 3px; line-height: 1.6; }
.mt { margin-top: 18px; }
tbody tr.active { background: var(--vp-c-brand-soft); font-weight: 600; }
</style>
