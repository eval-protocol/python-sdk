import React from "react";
import { computePivot } from "../util/pivot";

/**
 * Props for PivotTable.
 */
export interface PivotTableProps<T extends Record<string, unknown>> {
  /**
   * Source list of records to pivot.
   * Each record must expose the fields referenced by rowFields/columnFields/valueField.
   */
  data: T[];
  /**
   * Ordered list of record keys used to group rows.
   * Example: ["region", "rep"] or flattened JSONPath keys if using a flattener.
   */
  rowFields: (keyof T)[];
  /**
   * Ordered list of record keys used to group columns.
   * Example: ["product"] or flattened JSONPath keys if using a flattener.
   */
  columnFields: (keyof T)[];
  /**
   * Record key containing the numeric value to aggregate per cell.
   * If omitted, aggregator defaults to counting records ("count").
   */
  valueField?: keyof T;
  /**
   * Aggregation strategy. Built-ins: "count" | "sum" | "avg". Custom function allowed.
   * Default: "count". When using "sum"/"avg" or a custom function, numeric values are
   * extracted from valueField (if provided) and coerced via Number(). Non-finite values are ignored.
   */
  aggregator?: Parameters<typeof computePivot<T>>[0]["aggregator"];
  /**
   * Whether to render a right-most total column per row. Default: true.
   */
  showRowTotals?: boolean;
  /**
   * Whether to render a bottom total row per column (plus grand total if showRowTotals). Default: true.
   */
  showColumnTotals?: boolean;
  /**
   * Optional extra class names applied to the wrapping container.
   */
  className?: string;
  /**
   * Formatter applied to aggregated numeric values before rendering.
   * Default: toLocaleString with up to 3 fraction digits.
   */
  formatter?: (value: number) => React.ReactNode;
  /**
   * Value to render when a cell has no data for the given row/column intersection.
   * Default: "-".
   */
  emptyValue?: React.ReactNode;
}

function toKey(parts: unknown[]): string {
  return parts.map((p) => String(p)).join("||");
}

// removed local aggregation helpers; logic is in util/pivot.ts for testability

/**
 * Compact, generic pivot table component that renders a pivoted summary of arbitrary records.
 * Styling matches other components: white background, subtle borders, compact paddings.
 */
export function PivotTable<T extends Record<string, unknown>>({
  data,
  rowFields,
  columnFields,
  valueField,
  aggregator = "count",
  showRowTotals = true,
  showColumnTotals = true,
  className = "",
  formatter = (v) => v.toLocaleString(undefined, { maximumFractionDigits: 3 }),
  emptyValue = "-",
}: PivotTableProps<T>) {
  const {
    rowKeyTuples,
    colKeyTuples,
    cells,
    rowTotals,
    colTotals,
    grandTotal,
  } = computePivot<T>({
    data,
    rowFields,
    columnFields,
    valueField,
    aggregator,
  });

  return (
    <div
      className={`bg-white border border-gray-200 overflow-x-auto ${className}`}
    >
      <table className="w-full min-w-max">
        <thead className="bg-gray-50 border-b border-gray-200">
          <tr>
            {/* Row header labels */}
            {rowFields.map((f) => (
              <th
                key={String(f)}
                className="px-3 py-2 text-left text-xs font-semibold text-gray-700"
              >
                {String(f)}
              </th>
            ))}
            {/* Column headers (flattened) */}
            {colKeyTuples.map((tuple, idx) => (
              <th
                key={`col-${idx}`}
                className="px-3 py-2 text-right text-xs font-semibold text-gray-700 whitespace-nowrap"
              >
                {tuple.map((v) => String(v ?? "")).join(" / ")}
              </th>
            ))}
            {showRowTotals && (
              <th className="px-3 py-2 text-right text-xs font-semibold text-gray-700">
                Total
              </th>
            )}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {rowKeyTuples.map((rTuple, rIdx) => {
            const rKey = toKey(rTuple);
            return (
              <tr key={`row-${rIdx}`} className="text-xs">
                {/* Row header cells */}
                {rTuple.map((value, i) => (
                  <td
                    key={`rh-${i}`}
                    className="px-3 py-2 text-gray-900 whitespace-nowrap"
                  >
                    {String(value ?? "")}
                  </td>
                ))}
                {/* Data cells */}
                {colKeyTuples.map((cTuple, cIdx) => {
                  const cKey = toKey(cTuple);
                  const cell = cells[rKey]?.[cKey];
                  const content = cell ? formatter(cell.value) : emptyValue;
                  return (
                    <td
                      key={`c-${cIdx}`}
                      className="px-3 py-2 text-right text-gray-900 whitespace-nowrap"
                    >
                      {content}
                    </td>
                  );
                })}
                {/* Row total */}
                {showRowTotals && (
                  <td className="px-3 py-2 text-right text-gray-900 whitespace-nowrap font-medium">
                    {formatter(rowTotals[rKey] ?? 0)}
                  </td>
                )}
              </tr>
            );
          })}
          {showColumnTotals && (
            <tr className="bg-gray-50">
              {/* Total label spanning row header columns */}
              <td
                colSpan={Math.max(1, rowFields.length)}
                className="px-3 py-2 text-left text-xs font-semibold text-gray-700"
              >
                Total
              </td>
              {/* Column totals */}
              {colKeyTuples.map((cTuple, cIdx) => {
                const cKey = toKey(cTuple);
                return (
                  <td
                    key={`ct-${cIdx}`}
                    className="px-3 py-2 text-right text-gray-900 whitespace-nowrap font-medium"
                  >
                    {formatter(colTotals[cKey] ?? 0)}
                  </td>
                );
              })}
              {/* Grand total */}
              {showRowTotals && (
                <td className="px-3 py-2 text-right text-gray-900 whitespace-nowrap font-semibold">
                  {formatter(grandTotal)}
                </td>
              )}
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export default PivotTable;
