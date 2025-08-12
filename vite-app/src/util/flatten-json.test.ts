import { describe, it, expect } from 'vitest'
import { flattenJson } from './flatten-json'
import { readFileSync } from 'node:fs'

describe('flattenJson against logs.json', () => {
  it('flattens each entry in logs.json.logs and matches snapshot', () => {
    const logsUrl = new URL('../../data/logs.json', import.meta.url)
    const raw = readFileSync(logsUrl, 'utf-8')
    const parsed = JSON.parse(raw) as { logs?: unknown[] }

    expect(Array.isArray(parsed.logs)).toBe(true)
    const flattened = (parsed.logs ?? []).map((entry) => flattenJson(entry))

    expect(flattened).toMatchSnapshot()
  })
})
