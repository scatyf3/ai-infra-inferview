<script setup lang="ts">
import { computed, ref } from 'vue'
import { models } from '@lib/gpus'
import { formatBytes } from '@lib/memory'
import { sharding, zeroMemory } from '@lib/parallel'
import type { ModelConfig } from '@lib/types'
import SelectField from './ui/SelectField.vue'
import StatCard from './ui/StatCard.vue'

const props = withDefaults(defineProps<{ tp?: number; pp?: number; dp?: number; ep?: number }>(), { tp: 4, pp: 2, dp: 1, ep: 1 })

const modelId = ref('llama3-70b')
const tp = ref(props.tp)
const pp = ref(props.pp)
const dp = ref(props.dp)
const ep = ref(props.ep)
const microBatches = ref(8)
const batch = ref(8)
const tokens = ref(2048)
const zeroStage = ref<0 | 1 | 2 | 3>(3)

const cfg = computed<ModelConfig>(() => models.find((m) => m.id === modelId.value)!)
const isMoe = computed(() => (cfg.value.numExperts ?? 0) > 0)
const res = computed(() =>
  sharding({
    cfg: cfg.value,
    par: { tp: tp.value, pp: pp.value, dp: dp.value, ep: isMoe.value ? ep.value : 1, microBatches: microBatches.value },
    batch: batch.value,
    tokens: tokens.value,
    weightDtype: 'bf16',
    kvDtype: 'bf16',
  }),
)
const zero = computed(() => zeroMemory(cfg.value.params ?? 70e9, dp.value, zeroStage.value))

const powers = [1, 2, 4, 8, 16]
const modelOpts = models.map((m) => ({ value: m.id, label: m.name }))

/** 网格：行 = PP stage，列 = TP rank（EP 时列复用为 expert 组） */
const grid = computed(() => {
  const rows = []
  for (let s = 0; s < pp.value; s++) {
    const cols = []
    for (let t = 0; t < tp.value; t++) cols.push({ stage: s, rank: t })
    rows.push(cols)
  }
  return rows
})

const commRows = computed(() =>
  res.value.comm.map((c) => ({
    ...c,
    perLayer: formatBytes(c.perGpuBytesPerLayer),
    perFwd: c.kind === 'TP' ? formatBytes(c.perGpuBytesPerLayer * res.value.layersPerGpu) : c.kind === 'EP' ? formatBytes(c.perGpuBytesPerLayer * res.value.layersPerGpu) : formatBytes(c.perGpuBytesPerLayer * (pp.value - 1)),
  })),
)

const sliceLabel = computed(() => {
  const parts = [`权重按 TP=${tp.value} 切列/行`, `按 PP=${pp.value} 切层（每卡 ${res.value.layersPerGpu} 层）`]
  if (isMoe.value && ep.value > 1) parts.push(`expert 按 EP=${ep.value} 切`)
  if (dp.value > 1) parts.push(`DP=${dp.value} 复制整套`)
  return parts.join('，')
})
</script>

<template>
  <div class="widget">
    <h4>并行切分与通信量</h4>

    <div class="row controls">
      <SelectField label="模型" v-model="modelId" :options="modelOpts" width="200px" />
      <div class="knob">
        <span>TP</span>
        <div class="toggle-group"><button v-for="p in powers" :key="p" :class="{ active: tp === p }" @click="tp = p">{{ p }}</button></div>
      </div>
      <div class="knob">
        <span>PP</span>
        <div class="toggle-group"><button v-for="p in [1, 2, 4, 8]" :key="p" :class="{ active: pp === p }" @click="pp = p">{{ p }}</button></div>
      </div>
      <div class="knob">
        <span>DP</span>
        <div class="toggle-group"><button v-for="p in [1, 2, 4, 8]" :key="p" :class="{ active: dp === p }" @click="dp = p">{{ p }}</button></div>
      </div>
      <div class="knob" v-if="isMoe">
        <span>EP</span>
        <div class="toggle-group"><button v-for="p in [1, 2, 4, 8, 16]" :key="p" :class="{ active: ep === p }" @click="ep = p">{{ p }}</button></div>
      </div>
    </div>

    <p class="muted slice">{{ sliceLabel }} · 共 {{ res.totalGpus }} 张卡</p>

    <div class="cards">
      <StatCard label="每卡权重" :value="formatBytes(res.perGpuWeightBytes)" :sub="`attn ${formatBytes(res.perGpuAttentionBytes)} · mlp ${formatBytes(res.perGpuMlpBytes)}`" />
      <StatCard label="每卡 KV" :value="formatBytes(res.perGpuKvBytes)" :sub="`batch ${batch} × ${tokens} token`" tone="warn" />
      <StatCard label="每卡层数" :value="String(res.layersPerGpu)" :sub="`总 ${cfg.layers} 层 ÷ PP=${pp}`" />
      <StatCard label="PP bubble" :value="(res.ppBubble * 100).toFixed(1) + '%'" :sub="`(pp-1)/m，m=${microBatches}`" :tone="res.ppBubble > 0.2 ? 'bad' : 'good'" />
    </div>

    <div class="layout">
      <div class="grid-wrap">
        <div class="axis-x">TP rank →</div>
        <div class="gridbox">
          <div v-for="(row, si) in grid" :key="si" class="grow">
            <div class="stage-tag">stage {{ si }}<br /><span>{{ res.layersPerGpu }} 层</span></div>
            <div v-for="c in row" :key="c.rank" class="cell">
              <div class="cell-head">GPU {{ si * tp + c.rank }}</div>
              <div class="seg attn" :style="{ height: '26px' }">attn 1/{{ tp }}</div>
              <div class="seg mlp" :style="{ height: '26px' }">{{ isMoe && ep > 1 ? `experts 1/${ep * tp}` : `mlp 1/${tp}` }}</div>
              <div class="seg kv">KV 1/{{ tp }}</div>
            </div>
            <div v-if="pp > 1 && si < pp - 1" class="p2p">↓ P2P 激活</div>
          </div>
        </div>
        <p class="muted small">TP 组内每层两次 all-reduce（横向）；PP stage 之间每个 micro-batch 一次 P2P（纵向）<span v-if="dp > 1">；DP {{ dp }} 个副本各持有一份上图（推理时互不通信）</span></p>
      </div>
    </div>

    <table>
      <thead><tr><th>并行</th><th>原语</th><th>消息大小</th><th>每层次数</th><th>每卡每层实际传输</th><th>一次前向合计</th></tr></thead>
      <tbody>
        <tr v-for="c in commRows" :key="c.kind">
          <td><strong>{{ c.kind }}</strong></td>
          <td class="mono">{{ c.primitive }}</td>
          <td class="mono">{{ c.messageBytes ? formatBytes(c.messageBytes) : '—' }}</td>
          <td class="mono">{{ c.timesPerLayer || '—' }}</td>
          <td class="mono">{{ c.perGpuBytesPerLayer ? c.perLayer : '—' }}</td>
          <td class="mono">{{ c.perGpuBytesPerLayer ? c.perFwd : '—' }}</td>
        </tr>
      </tbody>
    </table>
    <ul class="where">
      <li v-for="c in res.comm" :key="'w' + c.kind"><strong>{{ c.kind }}</strong>：{{ c.where }}</li>
    </ul>
    <p v-for="w in res.warnings" :key="w" class="warn">⚠ {{ w }}</p>

    <details class="adv">
      <summary>训练态：ZeRO / FSDP 的显存-通信 tradeoff</summary>
      <div class="row" style="margin: 8px 0">
        <div class="knob">
          <span>ZeRO stage</span>
          <div class="toggle-group"><button v-for="s in ([0, 1, 2, 3] as const)" :key="s" :class="{ active: zeroStage === s }" @click="zeroStage = s">{{ s }}</button></div>
        </div>
      </div>
      <p class="mono">每参数每卡 {{ zero.bytesPerParamPerGpu }} B（bf16 权重 2 + bf16 梯度 2 + fp32 master 4 + Adam m,v 8 = 16 B）→ 每卡 {{ formatBytes(zero.perGpuBytes) }}（DP={{ dp }}）</p>
      <p class="muted">通信：{{ zero.commPerStep }}</p>
    </details>
  </div>
</template>

<style scoped>
.controls { gap: 12px 18px; margin-bottom: 8px; }
.knob { display: flex; flex-direction: column; gap: 3px; }
.knob > span { font-size: 12px; color: var(--wg-muted); }
.slice { font-size: 12.5px; margin: 6px 0 12px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-bottom: 14px; }
.axis-x { font-size: 11px; color: var(--wg-muted); margin-bottom: 4px; }
.gridbox { display: flex; flex-direction: column; gap: 6px; }
.grow { display: flex; gap: 6px; align-items: stretch; flex-wrap: wrap; position: relative; }
.stage-tag { font-size: 11px; color: var(--wg-muted); width: 62px; display: flex; flex-direction: column; justify-content: center; line-height: 1.3; }
.cell { border: 1px solid var(--wg-border); border-radius: 6px; overflow: hidden; min-width: 84px; background: var(--vp-c-bg); }
.cell-head { font-size: 10px; text-align: center; padding: 2px; color: var(--wg-muted); border-bottom: 1px solid var(--wg-border); }
.seg { font-size: 10.5px; text-align: center; padding: 5px 2px; color: #fff; }
.seg.attn { background: #3b82f6; }
.seg.mlp { background: #f97316; }
.seg.kv { background: #8b5cf6; }
.p2p { width: 100%; font-size: 11px; color: var(--wg-muted); padding-left: 68px; }
.small { font-size: 11.5px; margin-top: 8px; }
.where { font-size: 12px; color: var(--wg-muted); line-height: 1.7; padding-left: 18px; margin: 6px 0; }
.adv summary { cursor: pointer; font-size: 13px; color: var(--wg-muted); }
</style>
