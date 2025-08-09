/**
 * Aggregation strategy for computing each pivot cell value.
 *
 * - "count": number of records in the cell (ignores valueField)
 * - "sum": sum of numeric values extracted from valueField
 * - "avg": average of numeric values extracted from valueField
 * - custom: user function receiving the numeric values (from valueField if provided)
 *   and the raw records in the cell. Return the aggregated number to display.
 */
export type Aggregator<T> =
  | "count"
  | "sum"
  | "avg"
  | ((values: number[], records: T[]) => number);

export interface PivotComputationResult<T> {
  rowKeyTuples: unknown[][];
  colKeyTuples: unknown[][];
  cells: Record<string, Record<string, { value: number; records: T[] }>>;
  rowTotals: Record<string, number>;
  colTotals: Record<string, number>;
  grandTotal: number;
}

function toKey(parts: unknown[]): string {
  return parts.map((p) => String(p)).join("||");
}

function getTuple<T extends Record<string, unknown>>(
  record: T,
  fields: (keyof T)[]
): unknown[] {
  return fields.map((f) => record[f]);
}

function getNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function aggregate<T extends Record<string, unknown>>(
  values: number[],
  records: T[],
  agg: Aggregator<T>
): number {
  if (typeof agg === "function") return agg(values, records);
  if (agg === "sum") return values.reduce((a, b) => a + b, 0);
  if (agg === "avg")
    return values.length === 0
      ? 0
      : values.reduce((a, b) => a + b, 0) / values.length;
  // default: count
  return records.length;
}

export interface ComputePivotParams<T extends Record<string, unknown>> {
  data: T[];
  rowFields: (keyof T)[];
  columnFields: (keyof T)[];
  valueField?: keyof T;
  aggregator?: Aggregator<T>;
}

/**
 * Compute pivot table structures from input data and configuration.
 */
export function computePivot<T extends Record<string, unknown>>({
  data,
  rowFields,
  columnFields,
  valueField,
  aggregator = "count",
}: ComputePivotParams<T>): PivotComputationResult<T> {
  const rowKeyTuples: unknown[][] = [];
  const rowKeySet = new Set<string>();
  const colKeyTuples: unknown[][] = [];
  const colKeySet = new Set<string>();

  for (const rec of data) {
    const rTuple = getTuple(rec, rowFields);
    const rKey = toKey(rTuple);
    if (!rowKeySet.has(rKey)) {
      rowKeySet.add(rKey);
      rowKeyTuples.push(rTuple);
    }
    const cTuple = getTuple(rec, columnFields);
    const cKey = toKey(cTuple);
    if (!colKeySet.has(cKey)) {
      colKeySet.add(cKey);
      colKeyTuples.push(cTuple);
    }
  }

  // Deterministic ordering
  rowKeyTuples.sort((a, b) => toKey(a).localeCompare(toKey(b)));
  colKeyTuples.sort((a, b) => toKey(a).localeCompare(toKey(b)));

  type CellAgg = { value: number; records: T[] };
  const cells: Record<string, Record<string, CellAgg>> = {};
  const rowTotals: Record<string, number> = {};
  const colTotals: Record<string, number> = {};

  for (const rTuple of rowKeyTuples) {
    const rKey = toKey(rTuple);
    cells[rKey] = {};
    rowTotals[rKey] = 0;
  }
  for (const cTuple of colKeyTuples) {
    const cKey = toKey(cTuple);
    colTotals[cKey] = 0;
  }

  // Partition records per cell
  const cellRecords: Record<string, Record<string, T[]>> = {};
  for (const rec of data) {
    const rKey = toKey(getTuple(rec, rowFields));
    const cKey = toKey(getTuple(rec, columnFields));
    if (!cellRecords[rKey]) cellRecords[rKey] = {};
    if (!cellRecords[rKey][cKey]) cellRecords[rKey][cKey] = [];
    cellRecords[rKey][cKey].push(rec);
  }

  for (const rKey of Object.keys(cellRecords)) {
    for (const cKey of Object.keys(cellRecords[rKey])) {
      const records = cellRecords[rKey][cKey];
      const values: number[] = [];
      if (valueField != null) {
        for (const rec of records) {
          const v = getNumber(rec[valueField]);
          if (v != null) values.push(v);
        }
      }
      const value = aggregate(values, records, aggregator);
      cells[rKey][cKey] = { value, records };
      rowTotals[rKey] += value;
      colTotals[cKey] += value;
    }
  }

  const grandTotal = Object.values(rowTotals).reduce((a, b) => a + b, 0);

  return { rowKeyTuples, colKeyTuples, cells, rowTotals, colTotals, grandTotal };
}
