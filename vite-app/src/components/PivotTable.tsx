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
    <TableContainer className={className}>
      <table className="w-full min-w-max">
        <TableHead>
          <tr>
            {/* Row header labels */}
            {rowFields.map((f) => (
              <TableHeader key={String(f)}>{String(f)}</TableHeader>
            ))}
            {/* Column headers (flattened) */}
            {colKeyTuples.map((tuple, idx) => (
              <TableHeader key={`col-${idx}`} align="right" nowrap>
                {tuple.map((v) => String(v ?? "")).join(" / ")}
              </TableHeader>
            ))}
            {showRowTotals && <TableHeader align="right">Total</TableHeader>}
          </tr>
        </TableHead>
        <TableBody>
          {rowKeyTuples.map((rTuple, rIdx) => {
            const rKey = toKey(rTuple);
            return (
              <TableRow key={`row-${rIdx}`} className="text-xs">
                {/* Row header cells */}
                {rTuple.map((value, i) => (
                  <TableCell key={`rh-${i}`} nowrap>
                    {String(value ?? "")}
                  </TableCell>
                ))}
                {/* Data cells */}
                {colKeyTuples.map((cTuple, cIdx) => {
                  const cKey = toKey(cTuple);
                  const cell = cells[rKey]?.[cKey];
                  const content = cell ? formatter(cell.value) : emptyValue;
                  return (
                    <TableCell key={`c-${cIdx}`} align="right" nowrap>
                      {content}
                    </TableCell>
                  );
                })}
                {/* Row total */}
                {showRowTotals && (
                  <TableCell align="right" nowrap medium>
                    {formatter(rowTotals[rKey] ?? 0)}
                  </TableCell>
                )}
              </TableRow>
            );
          })}
          {showColumnTotals && (
            <TableRow gray>
              {/* Total label spanning row header columns */}
              <TableCell colSpan={Math.max(1, rowFields.length)} semibold>
                Total
              </TableCell>
              {/* Column totals */}
              {colKeyTuples.map((cTuple, cIdx) => {
                const cKey = toKey(cTuple);
                return (
                  <TableCell key={`ct-${cIdx}`} align="right" nowrap medium>
                    {formatter(colTotals[cKey] ?? 0)}
                  </TableCell>
                );
              })}
              {/* Grand total */}
              {showRowTotals && (
                <TableCell align="right" nowrap semibold>
                  {formatter(grandTotal)}
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
