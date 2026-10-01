import { describe, expect, it } from 'vitest'
import { locateQuote, makeQuote, renderMarks } from '@lib/annotate'

describe('renderMarks', () => {
  it('plain highlight', () => {
    expect(renderMarks('a ==重点== b')).toBe('a <mark>重点</mark> b')
  })

  it('highlight with note', () => {
    expect(renderMarks('==PagedAttention=={核心}')).toBe(
      '<mark class="anno" tabindex="0">PagedAttention<span class="anno-note" role="tooltip">核心</span></mark>',
    )
  })

  it('ignores comparison operators', () => {
    expect(renderMarks('if n == 1 and m == 2')).toBe('if n == 1 and m == 2')
  })

  it('escapes html before marking', () => {
    expect(renderMarks('==<b>==')).toBe('<mark>&lt;b&gt;</mark>')
  })
})

describe('text quote anchors', () => {
  const full = 'prefix caching saves prefill. chunked prefill improves ITL. prefill again.'

  it('round-trips a unique quote', () => {
    const start = full.indexOf('chunked')
    const q = makeQuote(full, start, start + 7)
    expect(locateQuote(full, q)).toEqual({ start, end: start + 7 })
  })

  it('disambiguates repeated text by context', () => {
    const start = full.indexOf('prefill again')
    const q = makeQuote(full, start, start + 7)
    expect(locateQuote(full, q)).toEqual({ start, end: start + 7 })
  })

  it('survives edits around the quote', () => {
    const start = full.indexOf('improves')
    const q = makeQuote(full, start, start + 8)
    const edited = 'NEW INTRO. ' + full
    expect(locateQuote(edited, q)?.start).toBe(edited.indexOf('improves'))
  })

  it('returns null when the text is gone', () => {
    expect(locateQuote('nothing here', makeQuote(full, 0, 6))).toBeNull()
  })
})
