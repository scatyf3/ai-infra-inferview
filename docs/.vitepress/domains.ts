/** 领域元数据：目录名 → 展示信息。这是唯一需要手工维护的结构性元数据。 */
export interface Domain {
  dir: string
  label: string
  short: string
  color: string
  order: number
}

export const domains: Domain[] = [
  { dir: 'inference', label: '推理系统核心', short: '推理', color: '#3b82f6', order: 1 },
  { dir: 'parallel', label: '并行与通信', short: '并行', color: '#8b5cf6', order: 2 },
  { dir: 'gpu', label: 'GPU / 算子', short: 'GPU', color: '#22c55e', order: 3 },
  { dir: 'framework', label: '框架内功', short: '框架', color: '#f97316', order: 4 },
  { dir: 'posttrain', label: 'Post-train / Efficient', short: '训练', color: '#ec4899', order: 5 },
  { dir: 'basics', label: '基础不能挂', short: '基础', color: '#64748b', order: 6 },
  { dir: 'handson', label: '手撕高频', short: '手撕', color: '#eab308', order: 7 },
]

export const domainByDir = Object.fromEntries(domains.map((d) => [d.dir, d])) as Record<string, Domain>
