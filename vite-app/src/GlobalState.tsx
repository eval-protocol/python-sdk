import { makeAutoObservable, runInAction } from "mobx";
import type { EvaluationRow } from "./types/eval-protocol";
import type { PivotConfig, FilterGroup } from "./types/filters";
import flattenJson from "./util/flatten-json";
import type { FlatJson } from "./util/flatten-json";
import { createFilterFunction } from "./util/filter-utils";

// Default pivot configuration
const DEFAULT_PIVOT_CONFIG: PivotConfig = {
  selectedRowFields: ["$.eval_metadata.name"],
  selectedColumnFields: ["$.input_metadata.completion_params.model"],
  selectedValueField: "$.evaluation_result.score",
  selectedAggregator: "avg",
  filters: [],
};

// Default table filter configuration
const DEFAULT_TABLE_FILTER_CONFIG: FilterGroup[] = [];

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
  // Table filter configuration
  tableFilterConfig: FilterGroup[];
  // Debounced, actually applied table filter configuration (for performance while typing)
  appliedTableFilterConfig: FilterGroup[];
  // Pagination configuration
  currentPage: number;
  pageSize: number;
  // Loading state
  isLoading: boolean = true;

  // Cached, denormalized data for performance
  // rollout_id -> flattened row
  private flattenedById: Record<string, FlatJson> = {};
  // rollout_id -> created_at timestamp (ms) for cheap sort
  private createdAtMsById: Record<string, number> = {};

  // Debounce timers for localStorage saves and filter application
  private savePivotConfigTimer: ReturnType<typeof setTimeout> | null = null;
  private saveTableFilterConfigTimer: ReturnType<typeof setTimeout> | null =
    null;
  private savePaginationConfigTimer: ReturnType<typeof setTimeout> | null =
    null;
  private applyTableFilterTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    // Load pivot config from localStorage or use defaults
    this.pivotConfig = this.loadPivotConfig();
    // Load table filter config from localStorage or use defaults
    this.tableFilterConfig = this.loadTableFilterConfig();
    // Initialize applied filter config with current value
    this.appliedTableFilterConfig = this.tableFilterConfig.slice();
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

  // Load table filter configuration from localStorage
  private loadTableFilterConfig(): FilterGroup[] {
    try {
      const stored = localStorage.getItem("tableFilterConfig");
      if (stored) {
        const parsed = JSON.parse(stored);
        return Array.isArray(parsed) ? parsed : DEFAULT_TABLE_FILTER_CONFIG;
      }
    } catch (error) {
      console.warn(
        "Failed to load table filter config from localStorage:",
        error
      );
    }
    return DEFAULT_TABLE_FILTER_CONFIG;
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
    if (this.savePivotConfigTimer) clearTimeout(this.savePivotConfigTimer);
    this.savePivotConfigTimer = setTimeout(() => {
      try {
        localStorage.setItem("pivotConfig", JSON.stringify(this.pivotConfig));
      } catch (error) {
        console.warn("Failed to save pivot config to localStorage:", error);
      }
    }, 200);
  }

  // Save table filter configuration to localStorage
  private saveTableFilterConfig() {
    if (this.saveTableFilterConfigTimer)
      clearTimeout(this.saveTableFilterConfigTimer);
    this.saveTableFilterConfigTimer = setTimeout(() => {
      try {
        localStorage.setItem(
          "tableFilterConfig",
          JSON.stringify(this.tableFilterConfig)
        );
      } catch (error) {
        console.warn(
          "Failed to save table filter config to localStorage:",
          error
        );
      }
    }, 200);
  }

  // Save pagination configuration to localStorage
  private savePaginationConfig() {
    if (this.savePaginationConfigTimer)
      clearTimeout(this.savePaginationConfigTimer);
    this.savePaginationConfigTimer = setTimeout(() => {
      try {
        localStorage.setItem(
          "paginationConfig",
          JSON.stringify({
            currentPage: this.currentPage,
            pageSize: this.pageSize,
          })
        );
      } catch (error) {
        console.warn(
          "Failed to save pagination config to localStorage:",
          error
        );
      }
    }, 200);
  }

  // Update pivot configuration and save to localStorage
  updatePivotConfig(updates: Partial<PivotConfig>) {
    Object.assign(this.pivotConfig, updates);
    this.savePivotConfig();
  }

  // Update table filter configuration and save to localStorage
  updateTableFilterConfig(filters: FilterGroup[]) {
    this.tableFilterConfig = filters;
    this.saveTableFilterConfig();

    // Debounce application of filters to avoid re-filtering on every keystroke
    if (this.applyTableFilterTimer) clearTimeout(this.applyTableFilterTimer);
    this.applyTableFilterTimer = setTimeout(() => {
      this.appliedTableFilterConfig = this.tableFilterConfig.slice();
    }, 150);
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

  // Reset table filter configuration to defaults
  resetTableFilterConfig() {
    this.tableFilterConfig = [...DEFAULT_TABLE_FILTER_CONFIG];
    this.appliedTableFilterConfig = [...DEFAULT_TABLE_FILTER_CONFIG];
    this.saveTableFilterConfig();
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
      const rolloutId = row.execution_metadata.rollout_id;
      this.dataset[rolloutId] = row;
      // Cache created_at in ms for cheap sorts
      const createdMs = new Date(row.created_at).getTime();
      this.createdAtMsById[rolloutId] = isNaN(createdMs) ? 0 : createdMs;
      // Cache flattened row for filtering/pivot keys
      this.flattenedById[rolloutId] = flattenJson(row);
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
  get sortedIds() {
    return Object.keys(this.dataset).sort(
      (a, b) => (this.createdAtMsById[b] ?? 0) - (this.createdAtMsById[a] ?? 0)
    );
  }

  get sortedDataset() {
    return this.sortedIds.map((id) => this.dataset[id]);
  }

  get flattenedDataset() {
    return this.sortedIds.map((id) => this.flattenedById[id]);
  }

  get filteredFlattenedDataset() {
    if (this.appliedTableFilterConfig.length === 0) {
      return this.flattenedDataset;
    }

    const filterFunction = createFilterFunction(this.appliedTableFilterConfig)!;
    return this.flattenedDataset.filter(filterFunction);
  }

  get filteredOriginalDataset() {
    if (this.appliedTableFilterConfig.length === 0) {
      return this.sortedDataset;
    }

    const filterFunction = createFilterFunction(this.appliedTableFilterConfig)!;
    return this.sortedIds
      .filter((id) => filterFunction(this.flattenedById[id]))
      .map((id) => this.dataset[id]);
  }

  get flattenedDatasetKeys() {
    const keySet = new Set<string>();
    // Iterate over cached flattened rows to build a unique key list
    this.sortedIds.forEach((id) => {
      const flat = this.flattenedById[id];
      if (flat) {
        Object.keys(flat).forEach((key) => keySet.add(key));
      }
    });
    return Array.from(keySet);
  }

  get totalCount() {
    return this.filteredFlattenedDataset.length;
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
