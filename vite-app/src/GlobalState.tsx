import { makeAutoObservable } from "mobx";
import type { EvaluationRow } from "./types/eval-protocol";
import flattenJson from "./util/flatten-json";

export class GlobalState {
  isConnected: boolean = false;
  // rollout_id -> EvaluationRow
  dataset: Record<string, EvaluationRow> = {};
  // rollout_id -> expanded
  expandedRows: Record<string, boolean> = {};

  constructor() {
    makeAutoObservable(this);
  }

  upsertRows(dataset: EvaluationRow[]) {
    dataset.forEach((row) => {
      if (!row.execution_metadata?.rollout_id) {
        return;
      }
      this.dataset[row.execution_metadata.rollout_id] = row;
    });
  }

  toggleRowExpansion(rolloutId?: string) {
    if (!rolloutId) {
      return;
    }
    if (this.expandedRows[rolloutId]) {
      this.expandedRows[rolloutId] = false;
    } else {
      this.expandedRows[rolloutId] = true;
    }
  }

  isRowExpanded(rolloutId?: string): boolean {
    if (!rolloutId) {
      return false;
    }
    return this.expandedRows[rolloutId];
  }

  setAllRowsExpanded(expanded: boolean) {
    Object.keys(this.dataset).forEach((rolloutId) => {
      this.expandedRows[rolloutId] = expanded;
    });
  }

  // Computed values following MobX best practices
  get sortedDataset() {
    return Object.values(this.dataset).sort(
      (a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );
  }

  get flattenedDataset() {
    return this.sortedDataset.map((row) => flattenJson(row));
  }

  get flattenedDatasetKeys() {
    if (this.flattenedDataset.length === 0) return [];
    return Object.keys(this.flattenedDataset[0]);
  }

  get totalCount() {
    return Object.keys(this.dataset).length;
  }
}
