import raw from '@data/vllm-releases.json'

export type ReleaseTag =
  | 'memory'
  | 'scheduling'
  | 'kernel'
  | 'parallel'
  | 'architecture'
  | 'quantization'
  | 'models'
  | 'serving'
  | 'hardware'

/** UI 筛选用的粗分组：9 个细 tag 合并成 6 组 */
export type FilterGroup = 'memory' | 'scheduling' | 'kernel' | 'parallel' | 'architecture' | 'ecosystem'

export interface Era {
  id: string
  label: string
  summary: string
}

export interface Release {
  version: string
  date: string
  headline: string
  tags: ReleaseTag[]
  milestone: boolean
  era?: string
  observation?: string
  scenario?: string
  how?: string
  related?: string[]
  sources?: string[]
}

export const RELEASE_TAGS: readonly ReleaseTag[] = [
  'memory', 'scheduling', 'kernel', 'parallel', 'architecture', 'quantization', 'models', 'serving', 'hardware',
]

export const TAG_GROUPS: Record<FilterGroup, ReleaseTag[]> = {
  memory: ['memory'],
  scheduling: ['scheduling'],
  kernel: ['kernel', 'quantization'],
  parallel: ['parallel'],
  architecture: ['architecture', 'serving'],
  ecosystem: ['models', 'hardware'],
}

export const FILTER_GROUPS: readonly FilterGroup[] = ['memory', 'scheduling', 'kernel', 'parallel', 'architecture', 'ecosystem']

export const GROUP_LABEL: Record<FilterGroup, string> = {
  memory: '显存 / KV',
  scheduling: '调度',
  kernel: 'kernel / 量化',
  parallel: '并行',
  architecture: '架构 / 服务',
  ecosystem: '模型 / 硬件',
}

const data = raw as { eras: Era[]; releases: Release[] }

export const eras: Era[] = data.eras
export const releases: Release[] = sortByDate(data.releases)

/** 数值段比较：v0.10 > v0.9；缺省段按 0 */
export function compareVersion(a: string, b: string): number {
  const pa = a.replace(/^v/, '').split('.').map((x) => parseInt(x, 10) || 0)
  const pb = b.replace(/^v/, '').split('.').map((x) => parseInt(x, 10) || 0)
  const n = Math.max(pa.length, pb.length)
  for (let i = 0; i < n; i++) {
    const d = (pa[i] ?? 0) - (pb[i] ?? 0)
    if (d !== 0) return d
  }
  return 0
}

export function sortByDate(rs: Release[], dir: 'asc' | 'desc' = 'asc'): Release[] {
  const sign = dir === 'asc' ? 1 : -1
  return [...rs].sort((a, b) => {
    if (a.date !== b.date) return a.date < b.date ? -sign : sign
    return sign * compareVersion(a.version, b.version)
  })
}

export function filterByTag(rs: Release[], tags: ReleaseTag | ReleaseTag[]): Release[] {
  const want = Array.isArray(tags) ? tags : [tags]
  return rs.filter((r) => r.tags.some((t) => want.includes(t)))
}

export function filterByGroup(rs: Release[], group: FilterGroup | 'all'): Release[] {
  if (group === 'all') return rs
  return filterByTag(rs, TAG_GROUPS[group])
}

export function milestones(rs: Release[]): Release[] {
  return rs.filter((r) => r.milestone)
}

/**
 * 按 era 分组，保持 eras 的顺序，丢掉空组。
 * 没写 era 的行归到「日期不晚于它的最近一个里程碑」所属的 era；比第一个里程碑还早的归到第一个 era。
 */
export function groupByEra(rs: Release[], eraList: Era[]): { era: Era; releases: Release[] }[] {
  const sorted = sortByDate(rs)
  const buckets = new Map<string, Release[]>(eraList.map((e) => [e.id, []]))
  let currentEra = eraList[0]?.id
  for (const r of sorted) {
    if (r.era) currentEra = r.era
    const id = r.era ?? currentEra
    if (id === undefined) continue
    buckets.get(id)?.push(r)
  }
  return eraList
    .map((era) => ({ era, releases: buckets.get(era.id) ?? [] }))
    .filter((g) => g.releases.length > 0)
}

/** 把细 tag 映射到它所属的筛选组 */
export function groupOfTag(tag: ReleaseTag): FilterGroup {
  for (const g of FILTER_GROUPS) if (TAG_GROUPS[g].includes(tag)) return g
  return 'ecosystem'
}
