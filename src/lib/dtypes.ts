import type { DType } from './types'

const BYTES: Record<DType, number> = { fp32: 4, bf16: 2, fp16: 2, fp8: 1, int8: 1, int4: 0.5 }

export function bytesOf(dtype: DType): number {
  return BYTES[dtype]
}

export const DTYPES: DType[] = ['fp32', 'bf16', 'fp16', 'fp8', 'int8', 'int4']
