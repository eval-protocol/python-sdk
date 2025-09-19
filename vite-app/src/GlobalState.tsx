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

const DEFAULT_GLOBAL_CONFIG: GlobalConfig = {
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
  // Pivot configuration
  pivotConfig: PivotConfig;
  // Unified filter configuration for both pivot and table views
  filterConfig: FilterGroup[];
  // Pagination configuration
  paginationConfig: PaginationConfig;
  // Sort configuration
  sortConfig: SortConfig;
  // Debounced, actually applied filter configuration (for performance while typing)
  appliedFilterConfig: FilterGroup[];
  // Loading state
  isLoading: boolean = true;

  // Cached, denormalized data for performance
  // rollout_id -> flattened row
  private flattenedById: Record<string, FlatJson> = {};
  // rollout_id -> created_at timestamp (ms) for cheap sort
  private createdAtMsById: Record<string, number> = {};

  // Debounce timers for localStorage saves and filter application
  private savePivotConfigTimer: ReturnType<typeof setTimeout> | null = null;
  private saveFilterConfigTimer: ReturnType<typeof setTimeout> | null = null;
  private savePaginationConfigTimer: ReturnType<typeof setTimeout> | null =
    null;
  private applyFilterTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    // Load pivot config from localStorage or use defaults
    this.pivotConfig = this.loadPivotConfig();
    // Load filter config from localStorage or use defaults
    this.filterConfig = this.loadFilterConfig();
    // Initialize applied filter config with current value
    this.appliedFilterConfig = this.filterConfig.slice();
    // Load pagination config from localStorage or use defaults
    this.paginationConfig = this.loadPaginationConfig();
    // Load sort config from localStorage or use defaults
    this.sortConfig = this.loadSortConfig();
    makeAutoObservable(this);
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

  // Load filter configuration from localStorage
  private loadFilterConfig(): FilterGroup[] {
    try {
      const stored = localStorage.getItem("filterConfig");
      if (stored) {
        const parsed = JSON.parse(stored);
        return Array.isArray(parsed) ? parsed : DEFAULT_FILTER_CONFIG;
      }
    } catch (error) {
      console.warn("Failed to load filter config from localStorage:", error);
    }
    return DEFAULT_FILTER_CONFIG;
  }

  // Load pagination configuration from localStorage
  private loadPaginationConfig(): PaginationConfig {
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
    return DEFAULT_PAGINATION_CONFIG;
  }

  // Load sort configuration from localStorage
  private loadSortConfig(): SortConfig {
    try {
      const stored = localStorage.getItem("sortConfig");
      if (stored) {
        const parsed = JSON.parse(stored);
        // Merge with defaults to handle any missing properties
        return { ...DEFAULT_SORT_CONFIG, ...parsed };
      }
    } catch (error) {
      console.warn("Failed to load sort config from localStorage:", error);
    }
    return DEFAULT_SORT_CONFIG;
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

  // Save filter configuration to localStorage
  private saveFilterConfig() {
    if (this.saveFilterConfigTimer) clearTimeout(this.saveFilterConfigTimer);
    this.saveFilterConfigTimer = setTimeout(() => {
      try {
        localStorage.setItem("filterConfig", JSON.stringify(this.filterConfig));
      } catch (error) {
        console.warn("Failed to save filter config to localStorage:", error);
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
          JSON.stringify(this.paginationConfig)
        );
      } catch (error) {
        console.warn(
          "Failed to save pagination config to localStorage:",
          error
        );
      }
    }, 200);
  }

  // Save sort configuration to localStorage
  private saveSortConfig() {
    if (this.saveFilterConfigTimer) clearTimeout(this.saveFilterConfigTimer);
    this.saveFilterConfigTimer = setTimeout(() => {
      try {
        localStorage.setItem("sortConfig", JSON.stringify(this.sortConfig));
      } catch (error) {
        console.warn("Failed to save sort config to localStorage:", error);
      }
    }, 200);
  }

  // Update pivot configuration and save to localStorage
  updatePivotConfig(updates: Partial<PivotConfig>) {
    Object.assign(this.pivotConfig, updates);
    this.savePivotConfig();
  }

  // Update filter configuration and save to localStorage
  updateFilterConfig(filters: FilterGroup[]) {
    this.filterConfig = filters;
    this.saveFilterConfig();

    // Debounce application of filters to avoid re-filtering on every keystroke
    if (this.applyFilterTimer) clearTimeout(this.applyFilterTimer);
    this.applyFilterTimer = setTimeout(() => {
      this.appliedFilterConfig = this.filterConfig.slice();
    }, 150);
  }

  // Update pagination configuration and save to localStorage
  updatePaginationConfig(updates: Partial<PaginationConfig>) {
    Object.assign(this.paginationConfig, updates);
    this.savePaginationConfig();
  }

  // Update sort configuration and save to localStorage
  updateSortConfig(updates: Partial<SortConfig>) {
    Object.assign(this.sortConfig, updates);
    // Reset to first page when sorting changes
    this.paginationConfig.currentPage = 1;
    this.saveSortConfig();
  }

  // Handle sort field click - toggle direction if same field, set to asc if new field
  handleSortFieldClick(field: string) {
    if (this.sortConfig.sortField === field) {
      // Toggle direction for same field
      this.sortConfig.sortDirection =
        this.sortConfig.sortDirection === "asc" ? "desc" : "asc";
    } else {
      // New field, set to ascending
      this.sortConfig.sortField = field;
      this.sortConfig.sortDirection = "asc";
    }
    this.saveSortConfig();
  }

  // Reset pivot configuration to defaults
  resetPivotConfig() {
    this.pivotConfig = { ...DEFAULT_PIVOT_CONFIG };
    this.savePivotConfig();
  }

  // Reset filter configuration to defaults
  resetFilterConfig() {
    this.filterConfig = [...DEFAULT_FILTER_CONFIG];
    this.appliedFilterConfig = [...DEFAULT_FILTER_CONFIG];
    this.saveFilterConfig();
  }

  // Reset pagination configuration to defaults
  resetPaginationConfig() {
    this.paginationConfig = { ...DEFAULT_PAGINATION_CONFIG };
    this.savePaginationConfig();
  }

  // Reset sort configuration to defaults
  resetSortConfig() {
    this.sortConfig = { ...DEFAULT_SORT_CONFIG };
    this.saveSortConfig();
  }

  // Set current page
  setCurrentPage(page: number) {
    this.paginationConfig.currentPage = page;
    this.savePaginationConfig();
  }

  // Set page size
  setPageSize(size: number) {
    this.paginationConfig.pageSize = size;
    this.paginationConfig.currentPage = 1; // Reset to first page when changing page size
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
      this.paginationConfig.currentPage = 1;
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

  get globalConfig(): GlobalConfig {
    return {
      pivotConfig: this.pivotConfig,
      filterConfig: this.filterConfig,
      paginationConfig: this.paginationConfig,
      sortConfig: this.sortConfig,
    };
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
