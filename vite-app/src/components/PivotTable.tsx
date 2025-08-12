import React from "react";
import { computePivot } from "../util/pivot";
import TableContainer, {
  TableHeader,
  TableCell,
  TableHead,
  TableBody,
  TableRow,
} from "./TableContainer";

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
  /**
   * Optional filter function to apply to records before pivoting.
   * Return true to include the record, false to exclude it.
   */
  filter?: (record: T) => boolean;
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
  filter,
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
    filter,
  });

  debugger;
  return (
    <TableContainer className={className}>
      <table className="w-full min-w-max">
        <TableHead>
          <tr>
            {/* Row header labels with enhanced styling */}
            {rowFields.map((f) => (
              <TableHeader key={String(f)} className="bg-blue-50">
                <div className="text-xs font-medium text-blue-700">
                  {String(f)}
                </div>
              </TableHeader>
            ))}
            {/* Column headers with enhanced styling */}
            {colKeyTuples.map((tuple, idx) => (
              <TableHeader
                key={`col-${idx}`}
                align="right"
                nowrap
                className={
                  columnFields.length > 0 ? "bg-green-50" : "bg-gray-50"
                }
              >
                <div
                  className={`text-xs font-medium ${
                    columnFields.length > 0 ? "text-green-700" : "text-gray-700"
                  }`}
                >
                  {tuple.map((v) => String(v ?? "")).join(" / ")}
                </div>
              </TableHeader>
            ))}
            {showRowTotals && (
              <TableHeader
                align="right"
                className="bg-gray-100 border-l-2 border-l-gray-300"
              >
                <div className="text-xs font-semibold text-gray-900">Total</div>
              </TableHeader>
            )}
          </tr>
        </TableHead>
        <TableBody>
          {rowKeyTuples.map((rTuple, rIdx) => {
            const rKey = toKey(rTuple);
            return (
              <TableRow
                key={`row-${rIdx}`}
                className="text-xs hover:bg-gray-50"
              >
                {/* Row header cells with enhanced styling */}
                {rTuple.map((value, i) => (
                  <TableCell key={`rh-${i}`} nowrap className="bg-blue-50">
                    <div className="font-medium text-blue-900">
                      {String(value ?? "")}
                    </div>
                  </TableCell>
                ))}
                {/* Data cells */}
                {colKeyTuples.map((cTuple, cIdx) => {
                  const cKey = toKey(cTuple);
                  const cell = cells[rKey]?.[cKey];
                  const content = cell ? formatter(cell.value) : emptyValue;
                  return (
                    <TableCell
                      key={`c-${cIdx}`}
                      align="right"
                      nowrap
                      className="bg-white"
                    >
                      <div className="font-mono text-sm">{content}</div>
                    </TableCell>
                  );
                })}
                {/* Row total with enhanced styling */}
                {showRowTotals && (
                  <TableCell
                    align="right"
                    nowrap
                    medium
                    className="bg-gray-100 border-l-2 border-l-gray-300"
                  >
                    <div className="text-xs font-semibold text-gray-900">
                      {formatter(rowTotals[rKey] ?? 0)}
                    </div>
                  </TableCell>
                )}
              </TableRow>
            );
          })}
          {showColumnTotals && (
            <TableRow gray>
              {/* Total label spanning row header columns */}
              <TableCell
                colSpan={Math.max(1, rowFields.length)}
                semibold
                className="bg-gray-100 border-t-2 border-gray-300"
              >
                <div className="text-xs font-semibold text-gray-900">Total</div>
              </TableCell>
              {/* Column totals with enhanced styling */}
              {colKeyTuples.map((cTuple, cIdx) => {
                const cKey = toKey(cTuple);
                return (
                  <TableCell
                    key={`ct-${cIdx}`}
                    align="right"
                    nowrap
                    medium
                    className="bg-gray-100 border-t-2 border-gray-300"
                  >
                    <div className="text-xs font-semibold text-gray-900">
                      {formatter(colTotals[cKey] ?? 0)}
                    </div>
                  </TableCell>
                );
              })}
              {/* Grand total with enhanced styling */}
              {showRowTotals && (
                <TableCell
                  align="right"
                  nowrap
                  semibold
                  className="bg-gray-200 border-t-0"
                >
                  <div className="text-sm font-bold text-gray-900">
                    {formatter(grandTotal)}
                  </div>
                </TableCell>
              )}
            </TableRow>
          )}
        </TableBody>
      </table>
    </TableContainer>
  );
}

export default PivotTable;
