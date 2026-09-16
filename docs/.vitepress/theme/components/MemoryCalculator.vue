<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'
import { gpus, models } from '@lib/gpus'
import { formatBytes, formatNumber, headDim, memoryBreakdown } from '@lib/memory'
import { rooflineAnalysis } from '@lib/roofline'
import { GiB } from '@lib/types'
import type { DType, ModelConfig } from '@lib/types'
import NumberField from './ui/NumberField.vue'
import SelectField from './ui/SelectField.vue'
import StatCard from './ui/StatCard.vue'
import RooflineChart from './ui/RooflineChart.vue'

const modelId = ref('llama3-70b')
const gpuId = ref('h100-sxm')
const weightDtype = ref<DType>('bf16')
const kvDtype = ref<DType>('bf16')
const batch = ref(32)
const context = ref(8192)
const utilization = ref(0.9)
const mfu = ref(0.5)

const cfg = reactive<ModelConfig>({ ...models[1] })
watch(modelId, (id) => Object.assign(cfg, models.find((m) => m.id === id)!), { immediate: false })

const gpu = computed(() => gpus.find((g) => g.id === gpuId.value)!)
const wl = computed(() => ({ batch: batch.value, context: context.value, weightDtype: weightDtype.value, kvDtype: kvDtype.value }))
const mem = computed(() => memoryBreakdown(cfg, wl.value, { gpu: gpu.value, utilization: utilization.value }))
const roof = computed(() => rooflineAnalysis(cfg, wl.value, gpu.value, { mfu: mfu.value, tp: mem.value.minTP }))

const dtypeOpts = (['bf16', 'fp16', 'fp8', 'int8', 'int4', 'fp32'] as DType[]).map((v) => ({ value: v, label: v }))
const modelOpts = models.map((m) => ({ value: m.id, label: m.name }))
const gpuOpts = gpus.map((g) => ({ value: g.id, label: g.name }))

const bars = computed(() => {
  const m = mem.value
  return [
    { label: '权重', bytes: m.weights, color: '#3b82f6' },
    { label: 'KV cache', bytes: m.kv, color: '#f97316' },
    { label: '激活', bytes: m.activations, color: '#22c55e' },
    { label: '固定开销', bytes: m.overhead, color: '#94a3b8' },
  ].map((b) => ({ ...b, pct: (b.bytes / m.total) * 100 }))
})

const points = computed(() => [
  { label: 'prefill', ai: roof.value.prefill.ai, color: '#22c55e' },
  { label: 'decode', ai: roof.value.decode.ai, color: '#f59e0b' },
])
const ms = (s: number) => (s >= 1 ? `${s.toFixed(2)} s` : `${(s * 1000).toFixed(1)} ms`)
const sideText = (s: string) => (s === 'compute' ? 'compute-bound' : 'memory-bound')
</script>

<template>
  <div class="widget">
    <h4>显存账 & Roofline 计算器</h4>

    <div class="grid">
      <SelectField label="模型" v-model="modelId" :options="modelOpts" width="100%" />
      <SelectField label="GPU" v-model="gpuId" :options="gpuOpts" width="100%" />
      <SelectField label="权重 dtype" v-model="weightDtype" :options="dtypeOpts" width="100%" />
      <SelectField label="KV dtype" v-model="kvDtype" :options="dtypeOpts" width="100%" />
      <NumberField label="batch" v-model="batch" :min="1" width="100%" />
      <NumberField label="context (token)" v-model="context" :min="1" :step="1024" width="100%" />
      <NumberField label="显存利用率" v-model="utilization" :min="0.1" :max="1" :step="0.05" width="100%" hint="vLLM gpu_memory_utilization" />
      <NumberField label="MFU" v-model="mfu" :min="0.05" :max="1" :step="0.05" width="100%" hint="峰值算力的实际利用率，用于时间估算" />
    </div>

    <details class="adv">
      <summary>模型结构（可手改）</summary>
      <div class="grid">
        <NumberField label="layers" v-model="cfg.layers" :min="1" width="100%" />
        <NumberField label="hidden d" v-model="cfg.hidden" :min="1" :step="128" width="100%" />
        <NumberField label="heads H" v-model="cfg.heads" :min="1" width="100%" />
        <NumberField label="kv_heads" v-model="cfg.kvHeads" :min="1" width="100%" />
        <NumberField label="d_ff" v-model="cfg.ffn" :min="1" :step="256" width="100%" />
        <NumberField label="vocab" v-model="cfg.vocab" :min="1" :step="1000" width="100%" />
      </div>
      <p class="muted">attention = {{ cfg.attention }}，head_dim = {{ headDim(cfg) }}<span v-if="cfg.numExperts">，MoE: {{ cfg.numExperts }} experts / top-{{ cfg.topK }}</span></p>
    </details>

    <div class="cards">
      <StatCard label="参数量" :value="formatNumber(mem.params)" :sub="`权重 ${formatBytes(mem.weights)} @ ${weightDtype}`" />
      <StatCard label="KV / token" :value="formatBytes(mem.kvPerToken)" :sub="`${batch} × ${context} token → ${formatBytes(mem.kv)}`" tone="warn" />
      <StatCard label="总显存" :value="formatBytes(mem.total)" :sub="`单卡可用 ${formatBytes(mem.perGpuBudget)}`" />
      <StatCard label="需要几张卡" :value="`${mem.gpusNeeded} × ${gpu.name.split(' ')[0]}`" :sub="`最小 TP=${mem.minTP}，每卡 ${formatBytes(mem.perGpuAtMinTP)}`" :tone="mem.gpusNeeded > 8 ? 'bad' : 'good'" />
    </div>

    <div class="stack">
      <div class="stack-bar">
        <i v-for="b in bars" :key="b.label" :style="{ width: b.pct + '%', background: b.color }" :title="`${b.label} ${formatBytes(b.bytes)}`" />
      </div>
      <div class="stack-legend">
        <span v-for="b in bars" :key="b.label"><i :style="{ background: b.color }" />{{ b.label }} {{ formatBytes(b.bytes) }} ({{ b.pct.toFixed(0) }}%)</span>
      </div>
    </div>

    <p v-for="w in mem.warnings" :key="w" class="warn">⚠ {{ w }}</p>

    <h4 class="mt">Roofline（按 TP={{ mem.minTP }} 分摊后单卡视角）</h4>
    <div class="cards">
      <StatCard label="prefill AI" :value="roof.prefill.ai.toFixed(0) + ' FLOP/B'" :sub="sideText(roof.prefill.side) + ` · TTFT ≈ ${ms(roof.prefill.time)}`" :tone="roof.prefill.side === 'compute' ? 'good' : 'warn'" />
      <StatCard label="decode AI" :value="roof.decode.ai.toFixed(1) + ' FLOP/B'" :sub="sideText(roof.decode.side) + ` · TPOT ≈ ${ms(roof.decode.time)}`" :tone="roof.decode.side === 'compute' ? 'good' : 'warn'" />
      <StatCard label="ridge point" :value="roof.ridge.toFixed(0) + ' FLOP/B'" :sub="`peak ${(roof.peak / 1e12).toFixed(0)} TFLOP/s ÷ BW ${(roof.bw / 1e12).toFixed(2)} TB/s`" />
      <StatCard label="decode 吞吐" :value="(batch / roof.decode.time).toFixed(0) + ' tok/s'" :sub="`单副本，batch ${batch}`" />
    </div>
    <RooflineChart :peak="roof.peak" :bw="roof.bw" :points="points" :peak-label="`${(roof.peak / 1e12).toFixed(0)} TFLOP/s`" />
    <p class="muted note">
      prefill 的 AI ≈ 每序列 token 数（权重读一次、算 S 次），落在 ridge 右边 → compute-bound，优化方向是提高 tensor core 利用率（更大 tile、fp8、chunked prefill 填满 SM）。
      decode 的 AI ≈ batch（权重读一次只算一个 token），落在 ridge 左边 → memory-bound，优化方向是减少访存：weight-only 量化、GQA/MLA 缩小 KV、continuous batching 把 batch 堆上去。
    </p>
  </div>
</template>

<style scoped>
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin: 14px 0; }
.adv { margin: 12px 0; }
.adv summary { cursor: pointer; font-size: 13px; color: var(--wg-muted); margin-bottom: 8px; }
.stack-bar { display: flex; height: 22px; border-radius: 6px; overflow: hidden; background: var(--vp-c-bg); border: 1px solid var(--wg-border); }
.stack-bar i { display: block; height: 100%; }
.stack-legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--wg-muted); margin-top: 6px; }
.stack-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.mt { margin-top: 20px; }
.note { font-size: 12.5px; line-height: 1.7; margin-top: 10px; }
</style>
