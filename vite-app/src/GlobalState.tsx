import { makeAutoObservable, runInAction } from "mobx";
import type { EvaluationRow } from "./types/eval-protocol";
import type {
  PivotConfig,
  FilterGroup,
  PaginationConfig,
  SortConfig,
  GlobalConfig,
  SortDirection,
} from "./types/configs";
import flattenJson from "./util/flatten-json";
import type { FlatJson } from "./util/flatten-json";
import { createFilterFunction } from "./util/filter-utils";
import {
  QueryParamsWatcher,
  queryParamsToPartialConfig,
} from "./util/query-params";

// Default pivot configuration
const DEFAULT_PIVOT_CONFIG: PivotConfig = {
  selectedRowFields: ["$.eval_metadata.name"],
  selectedColumnFields: ["$.input_metadata.completion_params.model"],
  selectedValueField: "$.evaluation_result.score",
  selectedAggregator: "avg",
};

// Default filter configuration
const DEFAULT_FILTER_CONFIG: FilterGroup[] = [];

// Default pagination configuration
const DEFAULT_PAGINATION_CONFIG: PaginationConfig = {
  currentPage: 1,
  pageSize: 25,
};

// Default sort configuration
const DEFAULT_SORT_CONFIG: SortConfig = {
  sortField: "created_at",
  sortDirection: "desc",
};

export const DEFAULT_GLOBAL_CONFIG: GlobalConfig = {
  pivotConfig: DEFAULT_PIVOT_CONFIG,
  filterConfig: DEFAULT_FILTER_CONFIG,
  paginationConfig: DEFAULT_PAGINATION_CONFIG,
  sortConfig: DEFAULT_SORT_CONFIG,
};

export class GlobalState {
  isConnected: boolean = false;
  // rollout_id -> EvaluationRow
  dataset: Record<string, EvaluationRow> = {};
  // rollout_id -> expanded
  expandedRows: Record<string, boolean> = {};
  // Unified global configuration
  globalConfig: GlobalConfig;
  // Debounced, actually applied filter configuration (for performance while typing)
  appliedFilterConfig: FilterGroup[];
  // Loading state
  isLoading: boolean = true;

  queryParamsWatcher: QueryParamsWatcher;

  // Cached, denormalized data for performance
  // rollout_id -> flattened row
  private flattenedById: Record<string, FlatJson> = {};
  // rollout_id -> created_at timestamp (ms) for cheap sort
  private createdAtMsById: Record<string, number> = {};

  // Debounce timers for localStorage saves and filter application
  private saveGlobalConfigTimer: ReturnType<typeof setTimeout> | null = null;
  private applyFilterTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    // Load global config from localStorage or use defaults
    this.globalConfig = this.loadGlobalConfig();
    // Initialize applied filter config with current value
    this.appliedFilterConfig = this.filterConfig.slice();
    makeAutoObservable(this);
    this.queryParamsWatcher = new QueryParamsWatcher(this);

    // Apply query params from URL if they exist
    this.applyQueryParamsFromUrl();
  }

  // Computed getters for individual configs
  get pivotConfig(): PivotConfig {
    return this.globalConfig.pivotConfig;
  }

  get filterConfig(): FilterGroup[] {
    return this.globalConfig.filterConfig;
  }

  get paginationConfig(): PaginationConfig {
    return this.globalConfig.paginationConfig;
  }

  get sortConfig(): SortConfig {
    return this.globalConfig.sortConfig;
  }

  // Computed getters for individual pagination properties
  get currentPage(): number {
    return this.paginationConfig.currentPage;
  }

  get pageSize(): number {
    return this.paginationConfig.pageSize;
  }

  // Computed getters for individual sort properties
  get sortField(): string {
    return this.sortConfig.sortField;
  }

  get sortDirection(): SortDirection {
    return this.sortConfig.sortDirection;
  }

  // Load global configuration from localStorage
  private loadGlobalConfig(): GlobalConfig {
    try {
      const stored = localStorage.getItem("globalConfig");
      if (stored) {
        const parsed = JSON.parse(stored);
        // Merge with defaults to handle any missing properties
        return {
          pivotConfig: { ...DEFAULT_PIVOT_CONFIG, ...parsed.pivotConfig },
          filterConfig: Array.isArray(parsed.filterConfig)
            ? parsed.filterConfig
            : DEFAULT_FILTER_CONFIG,
          paginationConfig: {
            ...DEFAULT_PAGINATION_CONFIG,
            ...parsed.paginationConfig,
          },
          sortConfig: { ...DEFAULT_SORT_CONFIG, ...parsed.sortConfig },
        };
      }
    } catch (error) {
      console.warn("Failed to load global config from localStorage:", error);
    }
    return { ...DEFAULT_GLOBAL_CONFIG };
  }

  // Save global configuration to localStorage
  private saveGlobalConfig() {
    if (this.saveGlobalConfigTimer) clearTimeout(this.saveGlobalConfigTimer);
    this.saveGlobalConfigTimer = setTimeout(() => {
      try {
        localStorage.setItem("globalConfig", JSON.stringify(this.globalConfig));
      } catch (error) {
        console.warn("Failed to save global config to localStorage:", error);
      }
    }, 200);
  }

  // Update pivot configuration and save to localStorage
  updatePivotConfig(updates: Partial<PivotConfig>) {
    Object.assign(this.globalConfig.pivotConfig, updates);
    this.saveGlobalConfig();
  }

  // Update filter configuration and save to localStorage
  updateFilterConfig(filters: FilterGroup[]) {
    this.globalConfig.filterConfig = filters;
    this.saveGlobalConfig();

    // Debounce application of filters to avoid re-filtering on every keystroke
    if (this.applyFilterTimer) clearTimeout(this.applyFilterTimer);
    this.applyFilterTimer = setTimeout(() => {
      this.appliedFilterConfig = this.filterConfig.slice();
    }, 150);
  }

  // Update pagination configuration and save to localStorage
  updatePaginationConfig(updates: Partial<PaginationConfig>) {
    Object.assign(this.globalConfig.paginationConfig, updates);
    this.saveGlobalConfig();
  }

  // Update sort configuration and save to localStorage
  updateSortConfig(updates: Partial<SortConfig>) {
    Object.assign(this.globalConfig.sortConfig, updates);
    // Reset to first page when sorting changes
    this.globalConfig.paginationConfig.currentPage = 1;
    this.saveGlobalConfig();
  }

  // Handle sort field click - toggle direction if same field, set to asc if new field
  handleSortFieldClick(field: string) {
    if (this.sortConfig.sortField === field) {
      // Toggle direction for same field
      this.globalConfig.sortConfig.sortDirection =
        this.sortConfig.sortDirection === "asc" ? "desc" : "asc";
    } else {
      // New field, set to ascending
      this.globalConfig.sortConfig.sortField = field;
      this.globalConfig.sortConfig.sortDirection = "asc";
    }
    this.saveGlobalConfig();
  }

  // Reset pivot configuration to defaults
  resetPivotConfig() {
    this.globalConfig.pivotConfig = { ...DEFAULT_PIVOT_CONFIG };
    this.saveGlobalConfig();
  }

  // Reset filter configuration to defaults
  resetFilterConfig() {
    this.globalConfig.filterConfig = [...DEFAULT_FILTER_CONFIG];
    this.appliedFilterConfig = [...DEFAULT_FILTER_CONFIG];
    this.saveGlobalConfig();
  }

  // Reset pagination configuration to defaults
  resetPaginationConfig() {
    this.globalConfig.paginationConfig = { ...DEFAULT_PAGINATION_CONFIG };
    this.saveGlobalConfig();
  }

  // Reset sort configuration to defaults
  resetSortConfig() {
    this.globalConfig.sortConfig = { ...DEFAULT_SORT_CONFIG };
    this.saveGlobalConfig();
  }

  // Set current page
  setCurrentPage(page: number) {
    this.globalConfig.paginationConfig.currentPage = page;
    this.saveGlobalConfig();
  }

  // Set page size
  setPageSize(size: number) {
    this.globalConfig.paginationConfig.pageSize = size;
    this.globalConfig.paginationConfig.currentPage = 1; // Reset to first page when changing page size
    this.saveGlobalConfig();
  }

  // Set loading state
  setLoading(loading: boolean) {
    this.isLoading = loading;
  }

  // Set connection state
  setConnected(connected: boolean) {
    this.isConnected = connected;
  }

  // Apply query params to global configuration
  applyQueryParams(queryParams: Record<string, string>) {
    debugger;
    const partialConfig = queryParamsToPartialConfig(queryParams);

    // Apply each section of the partial config
    if (partialConfig.pivotConfig) {
      this.updatePivotConfig(partialConfig.pivotConfig);
    }

    if (partialConfig.filterConfig) {
      this.updateFilterConfig(partialConfig.filterConfig);
    }

    if (partialConfig.paginationConfig) {
      this.updatePaginationConfig(partialConfig.paginationConfig);
    }

    if (partialConfig.sortConfig) {
      this.updateSortConfig(partialConfig.sortConfig);
    }
  }

  // Extract query params from URL and apply them to global configuration
  private applyQueryParamsFromUrl() {
    if (typeof window === "undefined") {
      return; // Skip on server-side rendering
    }

    const urlParams = new URLSearchParams(window.location.search);
    const queryParams: Record<string, string> = {};

    // Convert URLSearchParams to Record<string, string>
    for (const [key, value] of urlParams.entries()) {
      queryParams[key] = value;
    }

    // Only apply if there are query params
    if (Object.keys(queryParams).length > 0) {
      this.applyQueryParams(queryParams);
    }
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
      this.globalConfig.paginationConfig.currentPage = 1;
      this.isLoading = false;
    });
    this.saveGlobalConfig();
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
    const ids = Object.keys(this.dataset);

    if (this.sortConfig.sortField === "created_at") {
      // Special case for created_at - use cached timestamp
      return ids.sort((a, b) => {
        const aTime = this.createdAtMsById[a] ?? 0;
        const bTime = this.createdAtMsById[b] ?? 0;
        return this.sortConfig.sortDirection === "asc"
          ? aTime - bTime
          : bTime - aTime;
      });
    }

    // For other fields, sort by flattened data
    return ids.sort((a, b) => {
      const aFlat = this.flattenedById[a];
      const bFlat = this.flattenedById[b];

      if (!aFlat || !bFlat) return 0;

      const aValue = aFlat[this.sortConfig.sortField];
      const bValue = bFlat[this.sortConfig.sortField];

      // Handle undefined values
      if (aValue === undefined && bValue === undefined) return 0;
      if (aValue === undefined)
        return this.sortConfig.sortDirection === "asc" ? -1 : 1;
      if (bValue === undefined)
        return this.sortConfig.sortDirection === "asc" ? 1 : -1;

      // Handle different types
      if (typeof aValue === "string" && typeof bValue === "string") {
        const comparison = aValue.localeCompare(bValue);
        return this.sortConfig.sortDirection === "asc"
          ? comparison
          : -comparison;
      }

      if (typeof aValue === "number" && typeof bValue === "number") {
        return this.sortConfig.sortDirection === "asc"
          ? aValue - bValue
          : bValue - aValue;
      }

      // Fallback to string comparison
      const aStr = String(aValue);
      const bStr = String(bValue);
      const comparison = aStr.localeCompare(bStr);
      return this.sortConfig.sortDirection === "asc" ? comparison : -comparison;
    });
  }

  get sortedDataset() {
    return this.sortedIds.map((id) => this.dataset[id]);
  }

  get flattenedDataset() {
    return this.sortedIds.map((id) => this.flattenedById[id]);
  }

  get filteredFlattenedDataset() {
    if (this.appliedFilterConfig.length === 0) {
      return this.flattenedDataset;
    }

    const filterFunction = createFilterFunction(this.appliedFilterConfig)!;
    return this.flattenedDataset.filter(filterFunction);
  }

  get filteredOriginalDataset() {
    if (this.appliedFilterConfig.length === 0) {
      return this.sortedDataset;
    }

    const filterFunction = createFilterFunction(this.appliedFilterConfig)!;
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
    return Math.ceil(this.totalCount / this.paginationConfig.pageSize);
  }

  get startRow() {
    return (
      (this.paginationConfig.currentPage - 1) * this.paginationConfig.pageSize +
      1
    );
  }

  get endRow() {
    return Math.min(
      this.paginationConfig.currentPage * this.paginationConfig.pageSize,
      this.totalCount
    );
  }
}
