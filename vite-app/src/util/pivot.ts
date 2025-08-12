/**
 * Aggregation strategy for computing each pivot cell value.
 *
 * - "count": number of records in the cell (ignores valueField)
 * - "sum": sum of numeric values extracted from valueField
 * - "avg": average of numeric values extracted from valueField
 * - "min": minimum of numeric values extracted from valueField
 * - "max": maximum of numeric values extracted from valueField
 * - custom: user function receiving the numeric values (from valueField if provided)
 *   and the raw records in the cell. Return the aggregated number to display.
 */
export type Aggregator<T> =
  | "count"
  | "sum"
  | "avg"
  | "min"
  | "max"
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
  if (agg === "min") return values.length === 0 ? 0 : Math.min(...values);
  if (agg === "max") return values.length === 0 ? 0 : Math.max(...values);
  // default: count
  return records.length;
}

/**
 * Configuration parameters for `computePivot`.
 *
 * @template T - Shape of each input record. Must be indexable by the keys used in
 * `rowFields`, `columnFields`, and `valueField` (if provided).
 */
export interface ComputePivotParams<T extends Record<string, unknown>> {
  /**
   * Input records to pivot. Each record contributes to exactly one cell determined by
   * its `rowFields` and `columnFields` key tuple.
   */
  data: T[];

  /**
   * Ordered list of keys that form the row grouping key (tuple). Order matters; two
   * records with the same values in this order will be grouped into the same row.
   * Use an empty array to place all records into a single row.
   */
  rowFields: (keyof T)[];

  /**
   * Ordered list of keys that form the column grouping key (tuple). Order matters; two
   * records with the same values in this order will be grouped into the same column.
   * Use an empty array to place all records into a single column.
   */
  columnFields: (keyof T)[];

  /**
   * Optional key whose values are aggregated to compute each cell's numeric value.
   * Values are coerced using `Number(value)` and only finite numbers are included;
   * non-numeric/NaN/Infinity are ignored. If omitted, the default aggregation computes
   * counts of records per cell.
   */
  valueField?: keyof T;

  /**
   * Aggregation strategy applied per cell. Built-ins: `"count"` (default), `"sum"`,
   * and `"avg"`. You may also pass a custom function that receives the array of
   * numeric `values` (derived from `valueField`, if provided) and the raw `records`
   * for the cell, and returns the number to display.
   * @default "count"
   */
  aggregator?: Aggregator<T>;

  /**
   * Optional filter function to apply to records before pivoting.
   * Return true to include the record, false to exclude it.
   * This is useful for focusing analysis on specific subsets of data.
   */
  filter?: (record: T) => boolean;
}

/**
 * Compute pivot table structures from input data and configuration.
 *
 * Examples
 * 1) Count per region × product (default aggregator is "count")
 * ```ts
 * const res = computePivot({
 *   data: rows,
 *   rowFields: ['region'],
 *   columnFields: ['product'],
 * })
 * ```
 *
 * 2) Sum amounts per region × product
 * ```ts
 * const res = computePivot({
 *   data: rows,
 *   rowFields: ['region'],
 *   columnFields: ['product'],
 *   valueField: 'amount',
 *   aggregator: 'sum',
 * })
 * ```
 *
 * 3) Average amounts per region × product
 * ```ts
 * const res = computePivot({
 *   data: rows,
 *   rowFields: ['region'],
 *   columnFields: ['product'],
 *   valueField: 'amount',
 *   aggregator: 'avg',
 * })
 * ```
 *
 * 4) Minimum amounts per region × product
 * ```ts
 * const res = computePivot({
 *   data: rows,
 *   rowFields: ['region'],
 *   columnFields: ['product'],
 *   valueField: 'amount',
 *   aggregator: 'min',
 * })
 * ```
 *
 * 5) Maximum amounts per region × product
 * ```ts
 * const res = computePivot({
 *   data: rows,
 *   rowFields: ['region'],
 *   columnFields: ['product'],
 *   valueField: 'amount',
 *   aggregator: 'max',
 * })
 * ```
 *
 * 6) Multiple column fields (composite columns)
 * ```ts
 * const res = computePivot({
 *   data: rows,
 *   rowFields: ['region'],
 *   columnFields: ['product', 'quarter'],
 *   valueField: 'amount',
 *   aggregator: 'sum',
 * })
 * // Each column is the tuple [product, quarter]
 * ```
 *
 * 7) Custom aggregator (e.g., max)
 * ```ts
 * const res = computePivot({
 *   data: rows,
 *   rowFields: ['region'],
 *   columnFields: ['product'],
 *   valueField: 'amount',
 *   aggregator: (values) => values.length ? Math.max(...values) : 0,
 * })
 * ```
 *
 * 8) Single grand total (no rows/cols)
 * ```ts
 * const res = computePivot({
 *   data: rows,
 *   rowFields: [],
 *   columnFields: [],
 *   valueField: 'amount',
 *   aggregator: 'sum',
 * })
 * // res.grandTotal is the total sum
 * ```
 *
 * 9) Excel-style: multiple value fields alongside multiple column fields (recipe)
 * - Run computePivot once per metric (valueField + aggregator) and read values side-by-side.
 * ```ts
 * const metrics = [
 *   { key: 'Sum of Sales', valueField: 'sales' as const, aggregator: 'sum' as const },
 *   { key: 'Sum of Quantity', valueField: 'quantity' as const, aggregator: 'sum' as const },
 * ]
 *
 * const pivotsByMetric = Object.fromEntries(
 *   metrics.map((m) => [
 *     m.key,
 *     computePivot({
 *       data: rows,
 *       rowFields: ['year'],
 *       columnFields: ['region'],
 *       valueField: m.valueField,
 *       aggregator: m.aggregator,
 *     }),
 *   ]),
 * ) as Record<string, ReturnType<typeof computePivot<any>>>;
 *
 * // In the UI, iterate row/col keys from one pivot and render each metric column side-by-side:
 * // for (const rTuple of pivotsByMetric['Sum of Sales'].rowKeyTuples) {
 * //   const rKey = rTuple.join('||');
 * //   for (const cTuple of pivotsByMetric['Sum of Sales'].colKeyTuples) {
 * //     const cKey = cTuple.join('||');
 * //     const sales = pivotsByMetric['Sum of Sales'].cells[rKey]?.[cKey]?.value ?? 0;
 * //     const qty = pivotsByMetric['Sum of Quantity'].cells[rKey]?.[cKey]?.value ?? 0;
 * //     // Render: [Year, Region] -> Sales, Quantity
 * //   }
 * // }
 * ```
 */
export function computePivot<T extends Record<string, unknown>>({
  data,
  rowFields,
  columnFields,
  valueField,
  aggregator = "count",
  filter,
}: ComputePivotParams<T>): PivotComputationResult<T> {
  // Apply filter first if provided
  const filteredData = filter ? data.filter(filter) : data;

  // Filter out records that do not have defined values for all rowFields.
  // This avoids creating a row key of "undefined" and ensures such records
  // are not returned as part of the cells/row totals.
  const dataWithDefinedRows = filteredData.filter((rec) =>
    rowFields.every((f) => rec[f] !== undefined)
  );
  const rowKeyTuples: unknown[][] = [];
  const rowKeySet = new Set<string>();
  const colKeyTuples: unknown[][] = [];
  const colKeySet = new Set<string>();

  for (const rec of dataWithDefinedRows) {
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
  for (const rec of dataWithDefinedRows) {
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
    }
  }

  // Calculate column totals using the same aggregation method
  for (const cKey of Object.keys(colTotals)) {
    const columnRecords: T[] = [];
    const columnValues: number[] = [];

    for (const rKey of Object.keys(cellRecords)) {
      if (cellRecords[rKey][cKey]) {
        columnRecords.push(...cellRecords[rKey][cKey]);
        if (valueField != null) {
          for (const rec of cellRecords[rKey][cKey]) {
            const v = getNumber(rec[valueField]);
            if (v != null) columnValues.push(v);
          }
        }
      }
    }

    colTotals[cKey] = aggregate(columnValues, columnRecords, aggregator);
  }

  // Grand total should follow the same aggregation semantics over the entire dataset
  // rather than summing per-row/per-column aggregates (which can be incorrect for
  // non-additive aggregations like "avg").
  let grandTotal: number;
  {
    const allRecords = dataWithDefinedRows;
    const allValues: number[] = [];
    if (valueField != null) {
      for (const rec of allRecords) {
        const v = getNumber(rec[valueField]);
        if (v != null) allValues.push(v);
      }
    }
    grandTotal = aggregate(allValues, allRecords, aggregator);
  }

  return { rowKeyTuples, colKeyTuples, cells, rowTotals, colTotals, grandTotal };
}
