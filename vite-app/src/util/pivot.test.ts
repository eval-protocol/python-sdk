import { describe, it, expect } from 'vitest'
import { computePivot, type Aggregator } from './pivot'
import { readFileSync } from 'fs'
import flattenJson, { type FlatJson } from './flatten-json'

type Row = {
  region: string
  rep: string
  product: string
  amount?: number | string
}

const rows: Row[] = [
  { region: 'West', rep: 'A', product: 'Widget', amount: 120 },
  { region: 'West', rep: 'B', product: 'Gadget', amount: 90 },
  { region: 'East', rep: 'A', product: 'Widget', amount: 200 },
  { region: 'East', rep: 'B', product: 'Gadget', amount: '10' },
  { region: 'East', rep: 'B', product: 'Gadget', amount: 'not-a-number' },

]

describe('computePivot', () => {
  it('computes count when no valueField provided', () => {
    const res = computePivot<Row>({
      data: rows,
      rowFields: ['region'],
      columnFields: ['product'],
    })

    // Expect two row keys and two column keys
    expect(res.rowKeyTuples.map((t) => String(t))).toEqual([
      'East',
      'West',
    ])
    expect(res.colKeyTuples.map((t) => String(t))).toEqual([
      'Gadget',
      'Widget',
    ])

    // East/Gadget should count two (one invalid amount ignored in count mode)
    const rKeyEast = 'East'
    const cKeyGadget = 'Gadget'
    expect(res.cells[rKeyEast][cKeyGadget].value).toBe(2)
  })

  it('computes sum aggregator', () => {
    const res = computePivot<Row>({
      data: rows,
      rowFields: ['region'],
      columnFields: ['product'],
      valueField: 'amount',
      aggregator: 'sum',
    })

    const rKeyEast = 'East'
    const rKeyWest = 'West'
    const cKeyGadget = 'Gadget'
    const cKeyWidget = 'Widget'

    // East Gadget: 10 (string convertible) + invalid -> 10
    expect(res.cells[rKeyEast][cKeyGadget].value).toBe(10)
    // West Gadget: 90
    expect(res.cells[rKeyWest][cKeyGadget].value).toBe(90)
    // East Widget: 200
    expect(res.cells[rKeyEast][cKeyWidget].value).toBe(200)
    // West Widget: 120
    expect(res.cells[rKeyWest][cKeyWidget].value).toBe(120)
  })

  it('computes average aggregator', () => {
    const res = computePivot<Row>({
      data: rows,
      rowFields: ['region'],
      columnFields: ['product'],
      valueField: 'amount',
      aggregator: 'avg',
    })

    const rKeyEast = 'East'
    const rKeyWest = 'West'
    const cKeyGadget = 'Gadget'

    // East Gadget: values -> [10] => avg 10
    expect(res.cells[rKeyEast][cKeyGadget].value).toBe(10)
    // West Gadget: values -> [90] => avg 90
    expect(res.cells[rKeyWest][cKeyGadget].value).toBe(90)
  })

  it('computes minimum aggregator', () => {
    const res = computePivot<Row>({
      data: rows,
      rowFields: ['region'],
      columnFields: ['product'],
      valueField: 'amount',
      aggregator: 'min',
    })

    const rKeyEast = 'East'
    const rKeyWest = 'West'
    const cKeyGadget = 'Gadget'
    const cKeyWidget = 'Widget'

    // East Gadget: values -> [10] => min 10
    expect(res.cells[rKeyEast][cKeyGadget].value).toBe(10)
    // West Gadget: values -> [90] => min 90
    expect(res.cells[rKeyWest][cKeyGadget].value).toBe(90)
    // East Widget: values -> [200] => min 200
    expect(res.cells[rKeyEast][cKeyWidget].value).toBe(200)
    // West Widget: values -> [120] => min 120
    expect(res.cells[rKeyWest][cKeyWidget].value).toBe(120)
  })

  it('computes maximum aggregator', () => {
    const res = computePivot<Row>({
      data: rows,
      rowFields: ['region'],
      columnFields: ['product'],
      valueField: 'amount',
      aggregator: 'max',
    })

    const rKeyEast = 'East'
    const rKeyWest = 'West'
    const cKeyGadget = 'Gadget'
    const cKeyWidget = 'Widget'

    // East Gadget: values -> [10] => max 10
    expect(res.cells[rKeyEast][cKeyGadget].value).toBe(10)
    // West Gadget: values -> [90] => max 90
    expect(res.cells[rKeyWest][cKeyGadget].value).toBe(90)
    // East Widget: values -> [200] => max 200
    expect(res.cells[rKeyEast][cKeyWidget].value).toBe(200)
    // West Widget: values -> [120] => max 120
    expect(res.cells[rKeyWest][cKeyWidget].value).toBe(120)
  })

  it('handles empty cells for min/max aggregators', () => {
    // Add a row with no valid numeric values
    const rowsWithEmpty = [
      ...rows,
      { region: 'North', rep: 'C', product: 'Widget', amount: 'not-a-number' },
      { region: 'North', rep: 'D', product: 'Gadget', amount: 'also-not-a-number' },
    ]

    const res = computePivot<Row>({
      data: rowsWithEmpty,
      rowFields: ['region'],
      columnFields: ['product'],
      valueField: 'amount',
      aggregator: 'min',
    })

    const rKeyNorth = 'North'
    const cKeyWidget = 'Widget'
    const cKeyGadget = 'Gadget'

    // North region has no valid numeric values, should return 0 for min
    expect(res.cells[rKeyNorth][cKeyWidget].value).toBe(0)
    expect(res.cells[rKeyNorth][cKeyGadget].value).toBe(0)
  })

  it('applies filter before pivoting', () => {
    const res = computePivot<Row>({
      data: rows,
      rowFields: ['region'],
      columnFields: ['product'],
      valueField: 'amount',
      aggregator: 'sum',
      filter: (record) => record.region === 'East', // Only include East region
    })

    // Should only have East region rows
    expect(res.rowKeyTuples.map((t) => String(t))).toEqual(['East'])

    // Should still have all product columns
    expect(res.colKeyTuples.map((t) => String(t))).toEqual(['Gadget', 'Widget'])

    // East Gadget: 10 (string convertible) + invalid -> 10
    expect(res.cells['East']['Gadget'].value).toBe(10)
    // East Widget: 200
    expect(res.cells['East']['Widget'].value).toBe(200)

    // West region should not be present
    expect(res.cells['West']).toBeUndefined()

    // Grand total should only include East region data
    expect(res.grandTotal).toBe(210) // 10 + 200
  })

  it('supports custom aggregator', () => {
    const maxAgg: Aggregator<Row> = (values) =>
      values.length ? Math.max(...values) : 0

    const res = computePivot<Row>({
      data: rows,
      rowFields: ['region'],
      columnFields: ['product'],
      valueField: 'amount',
      aggregator: maxAgg,
    })

    const rKeyWest = 'West'
    const cKeyWidget = 'Widget'
    expect(res.cells[rKeyWest][cKeyWidget].value).toBe(120)
  })

  it('supports multiple column fields (composite columns)', () => {
    const res = computePivot<Row>({
      data: rows,
      rowFields: ['region'],
      columnFields: ['product', 'rep'],
      valueField: 'amount',
      aggregator: 'sum',
    })

    // Row and column key tuples
    expect(res.rowKeyTuples).toEqual([
      ['East'],
      ['West'],
    ])
    expect(res.colKeyTuples).toEqual([
      ['Gadget', 'B'],
      ['Widget', 'A'],
    ])

    const rEast = 'East'
    const rWest = 'West'
    const cGadgetB = 'Gadget||B'
    const cWidgetA = 'Widget||A'

    // Cell values (sum of numeric amounts)
    expect(res.cells[rEast][cGadgetB].value).toBe(10)
    expect(res.cells[rWest][cGadgetB].value).toBe(90)
    expect(res.cells[rEast][cWidgetA].value).toBe(200)
    expect(res.cells[rWest][cWidgetA].value).toBe(120)

    // Totals
    expect(res.rowTotals[rEast]).toBe(210)
    expect(res.rowTotals[rWest]).toBe(210)
    expect(res.colTotals[cGadgetB]).toBe(100)
    expect(res.colTotals[cWidgetA]).toBe(320)
    expect(res.grandTotal).toBe(420)
  })

  it('skips records with undefined row field values', () => {
    type LooseRow = {
      region?: string
      rep?: string
      product?: string
      amount?: number | string
    }

    const mixed: LooseRow[] = [
      { region: 'West', rep: 'A', product: 'Widget', amount: 120 },
      // Missing region should be excluded from cells entirely
      { rep: 'B', product: 'Gadget', amount: 90 },
    ]

    const res = computePivot<LooseRow>({
      data: mixed,
      rowFields: ['region'],
      columnFields: ['product'],
    })

    // Only 'West' row should be present; no 'undefined' row key
    expect(res.rowKeyTuples.map((t) => String(t))).toEqual(['West'])
    expect(Object.keys(res.cells)).toEqual(['West'])

    const rKeyWest = 'West'
    const cKeyWidget = 'Widget'

    // Count aggregator by default; only the valid record should be counted
    expect(res.cells[rKeyWest][cKeyWidget].value).toBe(1)

    // Grand total reflects only included records
    expect(res.grandTotal).toBe(1)
  })

  it("test_flaky_passes_sometimes", () => {
    // read logs.json from data/logs.json
    const logsUrl = new URL('../../data/logs.jsonl', import.meta.url)
    const raw = readFileSync(logsUrl, 'utf-8')
    const rows: FlatJson[] = []
    // iterate through each line and parse JSON
    raw.split('\n').forEach((line) => {
      if (line.trim() === '') return
      const parsed = JSON.parse(line)
      rows.push(flattenJson(parsed))
    })

    const res = computePivot({
      data: rows,
      rowFields: ['$.eval_metadata.name', '$.execution_metadata.experiment_id'],
      columnFields: ['$.input_metadata.completion_params.model'],
      valueField: '$.evaluation_result.score',
      aggregator: 'avg',
    })

    console.log(res)
  })
})
