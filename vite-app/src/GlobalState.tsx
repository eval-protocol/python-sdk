import { makeAutoObservable } from "mobx";
import type { EvaluationRow } from "./types/eval-protocol";

export class GlobalState {
  isConnected: boolean = false;
  dataset: EvaluationRow[] = [];
  expandedRows: Set<string> = new Set();

  constructor() {
    makeAutoObservable(this);
  }

  setDataset(dataset: EvaluationRow[]) {
    // Preserve expansion state for existing rows
    const newExpandedRows = new Set<string>();
    dataset.forEach((row) => {
      if (this.expandedRows.has(row.input_metadata.row_id)) {
        newExpandedRows.add(row.input_metadata.row_id);
      }
    });
    this.expandedRows = newExpandedRows;
    this.dataset = dataset;
  }

  toggleRowExpansion(rowId: string) {
    if (this.expandedRows.has(rowId)) {
      this.expandedRows.delete(rowId);
    } else {
      this.expandedRows.add(rowId);
    }
  }

  isRowExpanded(rowId: string): boolean {
    return this.expandedRows.has(rowId);
  }

  // Method to expand/collapse all rows
  setAllRowsExpanded(expanded: boolean) {
    if (expanded) {
      this.dataset.forEach((row) => {
        this.expandedRows.add(row.input_metadata.row_id);
      });
    } else {
      this.expandedRows.clear();
    }
  }
}
