import { makeAutoObservable } from "mobx";
import type { EvaluationRow } from "./types/eval-protocol";
import type { PivotConfig } from "./types/filters";
import flattenJson from "./util/flatten-json";

// Default pivot configuration
const DEFAULT_PIVOT_CONFIG: PivotConfig = {
  selectedRowFields: ["$.eval_metadata.name"],
  selectedColumnFields: ["$.input_metadata.completion_params.model"],
  selectedValueField: "$.evaluation_result.score",
  selectedAggregator: "avg",
  filters: [],
};

export class GlobalState {
  isConnected: boolean = false;
  // rollout_id -> EvaluationRow
  dataset: Record<string, EvaluationRow> = {};
  // rollout_id -> expanded
  expandedRows: Record<string, boolean> = {};
  // Pivot configuration
  pivotConfig: PivotConfig;

  constructor() {
    // Load pivot config from localStorage or use defaults
    this.pivotConfig = this.loadPivotConfig();
    makeAutoObservable(this);
  }

  // Load pivot configuration from localStorage
  private loadPivotConfig(): PivotConfig {
    try {
      const stored = localStorage.getItem("pivotConfig");
      if (stored) {
        const parsed = JSON.parse(stored);
        // Merge with defaults to handle any missing properties
        return { ...DEFAULT_PIVOT_CONFIG, ...parsed };
      }
    } catch (error) {
      console.warn("Failed to load pivot config from localStorage:", error);
    }
    return { ...DEFAULT_PIVOT_CONFIG };
  }

  // Save pivot configuration to localStorage
  private savePivotConfig() {
    try {
      localStorage.setItem("pivotConfig", JSON.stringify(this.pivotConfig));
    } catch (error) {
      console.warn("Failed to save pivot config to localStorage:", error);
    }
  }

  // Update pivot configuration and save to localStorage
  updatePivotConfig(updates: Partial<PivotConfig>) {
    Object.assign(this.pivotConfig, updates);
    this.savePivotConfig();
  }

  // Reset pivot configuration to defaults
  resetPivotConfig() {
    this.pivotConfig = { ...DEFAULT_PIVOT_CONFIG };
    this.savePivotConfig();
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
    const keySet = new Set<string>();
    this.flattenedDataset.forEach((row) => {
      Object.keys(row).forEach((key) => keySet.add(key));
    });
    return Array.from(keySet);
  }

  get totalCount() {
    return Object.keys(this.dataset).length;
  }
}
