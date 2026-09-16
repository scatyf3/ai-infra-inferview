import { describe, expect, it } from 'vitest'
import { findGpu, findModel } from '@lib/gpus'
import { activeParams, estimateParams, kvBytes, kvBytesPerToken, memoryBreakdown, weightBytes } from '@lib/memory'
import { GiB, KiB } from '@lib/types'

const llama70b = findModel('llama3-70b')
const h100 = findGpu('h100-sxm')

describe('memory accounting', () => {
  it('estimates Llama-3-70B params ≈ 70.5B', () => {
    const p = estimateParams(llama70b)
    expect(p / 1e9).toBeGreaterThan(70)
    expect(p / 1e9).toBeLessThan(71)
  })

  it('Llama-3-70B bf16 KV = 320 KiB / token', () => {
    expect(kvBytesPerToken(llama70b, 'bf16')).toBe(320 * KiB)
  })

  it('8k context × batch 32 → 80 GiB KV', () => {
    expect(kvBytes(llama70b, 'bf16', 8192 * 32) / GiB).toBeCloseTo(80, 5)
  })

  it('MLA (DeepSeek-V3) KV per token = layers × (512 + 64) × 2', () => {
    const ds = findModel('deepseek-v3')
    expect(kvBytesPerToken(ds, 'bf16')).toBe(61 * 576 * 2)
  })

  it('MoE active params < total params', () => {
    const ds = findModel('deepseek-v3')
    expect(activeParams(ds)).toBeLessThan(estimateParams(ds))
    expect(activeParams(ds) / 1e9).toBeGreaterThan(30)
    expect(activeParams(ds) / 1e9).toBeLessThan(45)
  })

  it('70B bf16 + 8k + b32 needs 4×H100 at TP=4', () => {
    const r = memoryBreakdown(llama70b, { batch: 32, context: 8192, weightDtype: 'bf16', kvDtype: 'bf16' }, { gpu: h100 })
    expect(r.weights / GiB).toBeCloseTo(131.4, 0)
    expect(r.kv / GiB).toBeCloseTo(80, 5)
    expect(r.gpusNeeded).toBe(4)
    expect(r.minTP).toBe(4)
    expect(r.perGpuAtMinTP).toBeLessThanOrEqual(r.perGpuBudget)
  })

  it('int4 weights are a quarter of bf16', () => {
    expect(weightBytes(llama70b, 'int4')).toBe(weightBytes(llama70b, 'bf16') / 4)
  })

  it('warns when kv_heads not divisible by TP', () => {
    const qwen = findModel('qwen2.5-7b') // kv_heads = 4
    const r = memoryBreakdown(qwen, { batch: 256, context: 32768, weightDtype: 'bf16', kvDtype: 'bf16' }, { gpu: findGpu('rtx-4090') })
    expect(r.minTP).toBeGreaterThan(4)
    expect(r.warnings.some((w) => w.includes('整除'))).toBe(true)
  })
})
