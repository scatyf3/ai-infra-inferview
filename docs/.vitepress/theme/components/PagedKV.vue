<script setup lang="ts">
import { computed, onUnmounted, ref } from 'vue'
import { addSeq, createPool, finish, freeBlockCount, preempt, resume, step, type PagedState } from '@lib/paged'

const props = withDefaults(defineProps<{ blockSize?: number; numBlocks?: number }>(), { blockSize: 4, numBlocks: 24 })

const SHARED_PREFIX = Array.from({ length: 12 }, (_, i) => i) // 3 个满 block 的公共前缀
const seqPalette = ['#3b82f6', '#f97316', '#22c55e', '#ec4899', '#8b5cf6']

function initial(): PagedState {
  let s = createPool(props.numBlocks, props.blockSize)
  s = addSeq(s, 'A', [...SHARED_PREFIX, 901, 902])
  s = addSeq(s, 'B', [...SHARED_PREFIX, 801, 802, 803, 804, 805])
  return s
}
const state = ref<PagedState>(initial())
const autoTimer = ref<number | null>(null)

const seqColor = (id: string) => seqPalette[state.value.seqs.findIndex((q) => q.id === id) % seqPalette.length]
const blockOwners = computed(() => {
  const m: Record<number, string[]> = {}
  for (const q of state.value.seqs) for (const b of q.blockIds) (m[b] ??= []).push(q.id)
  return m
})
const nextId = () => String.fromCharCode(65 + state.value.seqs.length)
const running = computed(() => state.value.seqs.filter((q) => q.state === 'running'))
const free = computed(() => freeBlockCount(state.value))
const sharedCount = computed(() => state.value.blocks.filter((b) => b.refCount > 1).length)

function doStep() { state.value = step(state.value) }
function addNew(withPrefix: boolean) {
  const id = nextId()
  const tail = Array.from({ length: 5 }, (_, i) => 7000 + state.value.seqs.length * 100 + i)
  state.value = addSeq(state.value, id, withPrefix ? [...SHARED_PREFIX, ...tail] : [...tail, ...tail])
}
function doPreempt(mode: 'swap' | 'recompute') {
  const victim = running.value[running.value.length - 1]
  if (victim) state.value = preempt(state.value, victim.id, mode)
}
function doResume(id: string) { state.value = resume(state.value, id) }
function doFinish(id: string) { state.value = finish(state.value, id) }
function reset() { stopAuto(); state.value = initial() }
function stopAuto() {
  if (autoTimer.value !== null) { clearInterval(autoTimer.value); autoTimer.value = null }
}
function toggleAuto() {
  if (autoTimer.value !== null) return stopAuto()
  autoTimer.value = window.setInterval(() => {
    if (free.value === 0) { doPreempt('swap') } else { doStep() }
  }, 900)
}
onUnmounted(stopAuto)

const stateLabel: Record<string, string> = { running: '运行中', swapped: '已换出(CPU)', preempted: '已抢占(丢弃)', finished: '已结束' }
const cols = computed(() => Math.min(8, props.numBlocks))
</script>

<template>
  <div class="widget">
    <h4>PagedAttention：block 分配、prefix 共享与抢占</h4>

    <div class="row controls">
      <button class="btn" @click="doStep">decode 一步</button>
      <button class="btn" @click="addNew(true)">新请求（共享前缀）</button>
      <button class="btn" @click="addNew(false)">新请求（无共享）</button>
      <button class="btn" @click="doPreempt('swap')" :disabled="!running.length">抢占 · swap</button>
      <button class="btn" @click="doPreempt('recompute')" :disabled="!running.length">抢占 · recompute</button>
      <button class="btn" @click="toggleAuto">{{ autoTimer === null ? '自动播放' : '暂停' }}</button>
      <button class="btn" @click="reset">重置</button>
    </div>

    <div class="stats">
      <span>block_size = {{ state.blockSize }} token</span>
      <span>free: <b :class="{ warn: free === 0 }">{{ free }}</b> / {{ state.blocks.length }}</span>
      <span>共享 block: <b>{{ sharedCount }}</b></span>
      <span>running: <b>{{ running.length }}</b></span>
    </div>

    <div class="pool" :style="{ gridTemplateColumns: `repeat(${cols}, 1fr)` }">
      <div
        v-for="b in state.blocks"
        :key="b.id"
        class="block"
        :class="{ free: b.refCount === 0, shared: b.refCount > 1, touched: state.touched.includes(b.id) }"
        :style="b.refCount ? { borderColor: seqColor(blockOwners[b.id]?.[0] ?? ''), background: `color-mix(in srgb, ${seqColor(blockOwners[b.id]?.[0] ?? '')} ${b.refCount > 1 ? 26 : 14}%, var(--vp-c-bg))` } : {}"
      >
        <div class="bid">#{{ b.id }}</div>
        <div class="slots">
          <i v-for="k in state.blockSize" :key="k" :class="{ filled: k <= b.filled }" :style="k <= b.filled && b.refCount ? { background: seqColor(blockOwners[b.id]?.[0] ?? '') } : {}" />
        </div>
        <div class="owners">
          <span v-if="b.refCount === 0" class="muted">free</span>
          <template v-else>
            <span v-for="o in blockOwners[b.id]" :key="o" :style="{ color: seqColor(o) }">{{ o }}</span>
            <span v-if="b.refCount > 1" class="ref">×{{ b.refCount }}</span>
          </template>
        </div>
      </div>
    </div>

    <h4 class="mt">Block tables</h4>
    <table class="bt">
      <thead><tr><th>seq</th><th>状态</th><th>token</th><th>logical → physical block</th><th>操作</th></tr></thead>
      <tbody>
        <tr v-for="q in state.seqs" :key="q.id">
          <td><b :style="{ color: seqColor(q.id) }">{{ q.id }}</b></td>
          <td>{{ stateLabel[q.state] }}<span v-if="state.cpuBlocks[q.id]" class="muted"> ({{ state.cpuBlocks[q.id] }} block)</span></td>
          <td class="mono">{{ q.tokens.length }} <span class="muted">(prompt {{ q.promptLen }})</span></td>
          <td class="mono cells">
            <span v-for="(b, i) in q.blockIds" :key="i" class="bcell" :class="{ shared: state.blocks[b].refCount > 1 }" :style="{ borderColor: seqColor(q.id) }">{{ i }}→#{{ b }}</span>
            <span v-if="!q.blockIds.length" class="muted">（无 GPU block）</span>
            <span v-if="q.cachedBlocks" class="hit">prefix 命中 {{ q.cachedBlocks }}</span>
          </td>
          <td>
            <button v-if="q.state === 'swapped' || q.state === 'preempted'" class="btn tiny" @click="doResume(q.id)">恢复</button>
            <button v-else-if="q.state === 'running'" class="btn tiny" @click="doFinish(q.id)">结束</button>
          </td>
        </tr>
      </tbody>
    </table>

    <div class="log">
      <div v-for="(l, i) in state.log.slice(-6)" :key="i">{{ l }}</div>
    </div>

    <p class="muted note">
      物理 block 固定大小，逻辑上连续的 KV 可以散落在任意物理 block，所以没有外部碎片，浪费上限是每序列最后一个 block 的半块。
      相同前缀的满 block 通过 refCount 共享，这就是 prefix caching / RadixAttention 省掉的重复 prefill。
      显存耗尽时调度器必须抢占：swap 把 block 拷到 CPU 再拷回，付 PCIe 带宽；recompute 直接丢弃，恢复时重新 prefill，付算力。
      短序列通常 recompute 更划算，长序列 swap 更划算。
    </p>
  </div>
</template>

<style scoped>
.controls { gap: 8px; margin-bottom: 10px; }
.stats { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 12.5px; color: var(--wg-muted); margin-bottom: 10px; }
.stats b { color: var(--wg-text); font-family: var(--vp-font-family-mono); }
.stats b.warn { color: #ef4444; }
.pool { display: grid; gap: 6px; }
.block { border: 2px solid var(--wg-border); border-radius: 6px; padding: 4px; text-align: center; transition: transform 0.15s, box-shadow 0.15s; }
.block.free { opacity: 0.55; border-style: dashed; }
.block.shared { border-style: double; border-width: 4px; }
.block.touched { box-shadow: 0 0 0 2px var(--vp-c-brand-1); }
.bid { font-size: 10px; color: var(--wg-muted); font-family: var(--vp-font-family-mono); }
.slots { display: flex; gap: 2px; justify-content: center; margin: 3px 0; }
.slots i { width: 8px; height: 12px; border-radius: 2px; background: var(--vp-c-bg-soft); border: 1px solid var(--wg-border); }
.owners { font-size: 11px; font-weight: 600; display: flex; gap: 3px; justify-content: center; }
.ref { color: var(--wg-muted); font-weight: 400; }
.mt { margin-top: 18px; }
.bt { width: 100%; }
.cells { display: flex; flex-wrap: wrap; gap: 4px; }
.bcell { border: 1px solid; border-radius: 4px; padding: 0 5px; font-size: 11px; }
.bcell.shared { border-style: double; border-width: 3px; }
.hit { font-size: 11px; color: #22c55e; }
.btn.tiny { padding: 2px 8px; font-size: 11px; }
.log { margin-top: 10px; padding: 8px 10px; background: var(--vp-c-bg); border: 1px solid var(--wg-border); border-radius: 6px; font-family: var(--vp-font-family-mono); font-size: 11.5px; line-height: 1.7; color: var(--wg-muted); max-height: 130px; overflow-y: auto; }
.note { font-size: 12.5px; line-height: 1.75; margin-top: 12px; }
</style>
