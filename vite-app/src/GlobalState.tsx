import { makeAutoObservable } from "mobx";
import type { EvaluationRow } from "./types/eval-protocol";

export class GlobalState {
  isConnected: boolean = false;
  dataset: Record<string, EvaluationRow> = {};
  expandedRows: Record<string, boolean> = {};

  constructor() {
    makeAutoObservable(this);
  }

  setDataset(dataset: EvaluationRow[]) {
    // Create new dataset object to avoid multiple re-renders
    dataset.forEach((row) => {
      this.dataset[row.input_metadata.row_id] = row;
    });
  }

  toggleRowExpansion(rowId: string) {
    if (this.expandedRows[rowId]) {
      this.expandedRows[rowId] = false;
    } else {
      this.expandedRows[rowId] = true;
    }
  }

  isRowExpanded(rowId: string): boolean {
    return this.expandedRows[rowId];
  }

  setAllRowsExpanded(expanded: boolean) {
    Object.keys(this.dataset).forEach((rowId) => {
      this.expandedRows[rowId] = expanded;
    });
  }

  // Computed values following MobX best practices
  get sortedDataset() {
    return Object.values(this.dataset).sort(
      (a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );
  }

  get totalCount() {
    return Object.keys(this.dataset).length;
  }
}
