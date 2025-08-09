import { describe, it, expect } from 'vitest'
import { computePivot, type Aggregator } from './pivot'

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
})
