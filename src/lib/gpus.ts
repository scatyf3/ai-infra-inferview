import raw from '@data/gpus.json'
import modelsRaw from '@data/models.json'
import type { GpuSpec, ModelConfig } from './types'

export const gpus = raw as GpuSpec[]
export const models = modelsRaw as (ModelConfig & { id: string; name: string })[]

export function findGpu(id: string): GpuSpec {
  const g = gpus.find((x) => x.id === id)
  if (!g) throw new Error(`unknown gpu ${id}`)
  return g
}

export function findModel(id: string): ModelConfig {
  const m = models.find((x) => x.id === id)
  if (!m) throw new Error(`unknown model ${id}`)
  return m
}
