import { describe, expect, it } from 'vitest'
import { existsSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  RELEASE_TAGS,
  compareVersion,
  eras,
  filterByGroup,
  filterByTag,
  groupByEra,
  groupOfTag,
  milestones,
  releases,
  sortByDate,
  type Release,
} from '@lib/releases'

const here = dirname(fileURLToPath(import.meta.url))
const docsDir = resolve(here, '../../../docs')

const mk = (version: string, date: string, extra: Partial<Release> = {}): Release => ({
  version,
  date,
  headline: version,
  tags: ['memory'],
  milestone: false,
  ...extra,
})

describe('releases helpers', () => {
  it('compareVersion compares numeric segments, not lexically', () => {
    expect(compareVersion('v0.10', 'v0.9')).toBeGreaterThan(0)
    expect(compareVersion('v0.9', 'v0.10')).toBeLessThan(0)
    expect(compareVersion('v0.8.5', 'v0.8.5')).toBe(0)
    expect(compareVersion('v0.8.5', 'v0.8')).toBeGreaterThan(0)
  })

  it('sortByDate sorts ascending without mutating input', () => {
    const input = [mk('v0.2', '2023-09-28'), mk('v0.1', '2023-06-20'), mk('v0.10', '2025-07-24')]
    const copy = [...input]
    const out = sortByDate(input)
    expect(out.map((r) => r.version)).toEqual(['v0.1', 'v0.2', 'v0.10'])
    expect(input).toEqual(copy)
    expect(sortByDate(input, 'desc')[0].version).toBe('v0.10')
  })

  it('filterByTag / filterByGroup', () => {
    const rs = [
      mk('a', '2024-01-01', { tags: ['memory'] }),
      mk('b', '2024-01-02', { tags: ['quantization'] }),
      mk('c', '2024-01-03', { tags: ['kernel', 'models'] }),
    ]
    expect(filterByGroup(rs, 'all')).toHaveLength(3)
    expect(filterByTag(rs, 'memory').map((r) => r.version)).toEqual(['a'])
    expect(filterByGroup(rs, 'kernel').map((r) => r.version)).toEqual(['b', 'c'])
    expect(filterByGroup(rs, 'ecosystem').map((r) => r.version)).toEqual(['c'])
    expect(groupOfTag('quantization')).toBe('kernel')
    expect(groupOfTag('serving')).toBe('architecture')
  })

  it('groupByEra keeps era order, drops empty eras, attaches non-milestones to the latest preceding milestone era', () => {
    const eraList = [
      { id: 'e1', label: 'E1', summary: '' },
      { id: 'e2', label: 'E2', summary: '' },
      { id: 'e3', label: 'E3', summary: '' },
    ]
    const rs = [
      mk('v0.3', '2024-03-01'),
      mk('v0.2', '2024-02-01', { milestone: true, era: 'e2' }),
      mk('v0.1', '2024-01-01', { milestone: true, era: 'e1' }),
      mk('v0.0', '2023-12-01'),
    ]
    const g = groupByEra(rs, eraList)
    expect(g.map((x) => x.era.id)).toEqual(['e1', 'e2'])
    expect(g[0].releases.map((r) => r.version)).toEqual(['v0.0', 'v0.1'])
    expect(g[1].releases.map((r) => r.version)).toEqual(['v0.2', 'v0.3'])
  })
})

describe('vllm-releases.json integrity', () => {
  it('has one row per minor version, sorted, with unique versions and ISO dates', () => {
    expect(releases.length).toBeGreaterThanOrEqual(30)
    const versions = releases.map((r) => r.version)
    expect(new Set(versions).size).toBe(versions.length)
    for (const r of releases) expect(r.date).toMatch(/^\d{4}-\d{2}-\d{2}$/)
    for (let i = 1; i < releases.length; i++) expect(releases[i - 1].date <= releases[i].date).toBe(true)
  })

  it('uses only known tags and has at least one tag per row', () => {
    for (const r of releases) {
      expect(r.tags.length).toBeGreaterThan(0)
      for (const t of r.tags) expect(RELEASE_TAGS).toContain(t)
    }
  })

  it('milestones carry observation / scenario / how / related and a valid era', () => {
    const eraIds = new Set(eras.map((e) => e.id))
    const ms = milestones(releases)
    expect(ms.length).toBeGreaterThanOrEqual(10)
    for (const m of ms) {
      expect(m.era, m.version).toBeDefined()
      expect(eraIds.has(m.era!), `${m.version} era ${m.era}`).toBe(true)
      for (const k of ['observation', 'scenario', 'how'] as const) {
        expect(typeof m[k], `${m.version}.${k}`).toBe('string')
        expect(m[k]!.length, `${m.version}.${k}`).toBeGreaterThan(10)
      }
      expect(m.related?.length, `${m.version}.related`).toBeGreaterThan(0)
    }
  })

  it('every related path points at an existing docs page (component links bypass VitePress dead-link check)', () => {
    for (const r of releases) {
      for (const p of r.related ?? []) {
        expect(p.startsWith('/'), `${r.version}: ${p}`).toBe(true)
        const file = resolve(docsDir, p.slice(1) + '.md')
        expect(existsSync(file), `${r.version}: ${p} -> ${file}`).toBe(true)
      }
    }
  })

  it('every row has https sources', () => {
    for (const r of releases) {
      expect(r.sources?.length, r.version).toBeGreaterThan(0)
      for (const s of r.sources ?? []) expect(s.startsWith('https://'), `${r.version}: ${s}`).toBe(true)
    }
  })

  it('every era is used by at least one release', () => {
    const g = groupByEra(releases, eras)
    expect(g.map((x) => x.era.id)).toEqual(eras.map((e) => e.id))
  })
})
