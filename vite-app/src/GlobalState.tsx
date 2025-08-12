import { makeAutoObservable, runInAction } from "mobx";
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

// Default pagination configuration
const DEFAULT_PAGINATION_CONFIG = {
  currentPage: 1,
  pageSize: 25,
};

export class GlobalState {
  isConnected: boolean = false;
  // rollout_id -> EvaluationRow
  dataset: Record<string, EvaluationRow> = {};
  // rollout_id -> expanded
  expandedRows: Record<string, boolean> = {};
  // Pivot configuration
  pivotConfig: PivotConfig;
  // Pagination configuration
  currentPage: number;
  pageSize: number;
  // Loading state
  isLoading: boolean = true;

  constructor() {
    // Load pivot config from localStorage or use defaults
    this.pivotConfig = this.loadPivotConfig();
    // Load pagination config from localStorage or use defaults
    const paginationConfig = this.loadPaginationConfig();
    this.currentPage = paginationConfig.currentPage;
    this.pageSize = paginationConfig.pageSize;
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

  // Load pagination configuration from localStorage
  private loadPaginationConfig() {
    try {
      const stored = localStorage.getItem("paginationConfig");
      if (stored) {
        const parsed = JSON.parse(stored);
        // Merge with defaults to handle any missing properties
        return { ...DEFAULT_PAGINATION_CONFIG, ...parsed };
      }
    } catch (error) {
      console.warn(
        "Failed to load pagination config from localStorage:",
        error
      );
    }
    return { ...DEFAULT_PAGINATION_CONFIG };
  }

  // Save pivot configuration to localStorage
  private savePivotConfig() {
    try {
      localStorage.setItem("pivotConfig", JSON.stringify(this.pivotConfig));
    } catch (error) {
      console.warn("Failed to save pivot config to localStorage:", error);
    }
  }

  // Save pagination configuration to localStorage
  private savePaginationConfig() {
    try {
      localStorage.setItem(
        "paginationConfig",
        JSON.stringify({
          currentPage: this.currentPage,
          pageSize: this.pageSize,
        })
      );
    } catch (error) {
      console.warn("Failed to save pagination config to localStorage:", error);
    }
  }

  // Update pivot configuration and save to localStorage
  updatePivotConfig(updates: Partial<PivotConfig>) {
    Object.assign(this.pivotConfig, updates);
    this.savePivotConfig();
  }

  // Update pagination configuration and save to localStorage
  updatePaginationConfig(
    updates: Partial<{ currentPage: number; pageSize: number }>
  ) {
    if (updates.currentPage !== undefined) {
      this.currentPage = updates.currentPage;
    }
    if (updates.pageSize !== undefined) {
      this.pageSize = updates.pageSize;
    }
    this.savePaginationConfig();
  }

  // Reset pivot configuration to defaults
  resetPivotConfig() {
    this.pivotConfig = {
      ...DEFAULT_PIVOT_CONFIG,
      filters: [], // Ensure filters is an empty array of FilterGroups
    };
    this.savePivotConfig();
  }

  // Reset pagination configuration to defaults
  resetPaginationConfig() {
    this.currentPage = DEFAULT_PAGINATION_CONFIG.currentPage;
    this.pageSize = DEFAULT_PAGINATION_CONFIG.pageSize;
    this.savePaginationConfig();
  }

  // Set current page
  setCurrentPage(page: number) {
    this.currentPage = page;
    this.savePaginationConfig();
  }

  // Set page size
  setPageSize(size: number) {
    this.pageSize = size;
    this.currentPage = 1; // Reset to first page when changing page size
    this.savePaginationConfig();
  }

  // Set loading state
  setLoading(loading: boolean) {
    this.isLoading = loading;
  }

  // Set connection state
  setConnected(connected: boolean) {
    this.isConnected = connected;
  }

  upsertRows(dataset: EvaluationRow[]) {
    runInAction(() => {
      this.isLoading = true;
    });

    dataset.forEach((row) => {
      if (!row.execution_metadata?.rollout_id) {
        return;
      }
      this.dataset[row.execution_metadata.rollout_id] = row;
    });

    runInAction(() => {
      // Reset to first page when dataset changes
      this.currentPage = 1;
      this.isLoading = false;
    });
    this.savePaginationConfig();
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

  get totalPages() {
    return Math.ceil(this.totalCount / this.pageSize);
  }

  get startRow() {
    return (this.currentPage - 1) * this.pageSize + 1;
  }

  get endRow() {
    return Math.min(this.currentPage * this.pageSize, this.totalCount);
  }
}
