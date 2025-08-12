import { makeAutoObservable, runInAction } from "mobx";
import type { EvaluationRow } from "./types/eval-protocol";
import type { PivotConfig, FilterGroup } from "./types/filters";
import flattenJson from "./util/flatten-json";

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
  // Pagination configuration
  currentPage: number;
  pageSize: number;
  // Loading state
  isLoading: boolean = true;

  constructor() {
    // Load pivot config from localStorage or use defaults
    this.pivotConfig = this.loadPivotConfig();
    // Load table filter config from localStorage or use defaults
    this.tableFilterConfig = this.loadTableFilterConfig();
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
    try {
      localStorage.setItem("pivotConfig", JSON.stringify(this.pivotConfig));
    } catch (error) {
      console.warn("Failed to save pivot config to localStorage:", error);
    }
  }

  // Save table filter configuration to localStorage
  private saveTableFilterConfig() {
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

  // Update table filter configuration and save to localStorage
  updateTableFilterConfig(filters: FilterGroup[]) {
    this.tableFilterConfig = filters;
    this.saveTableFilterConfig();
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

  get filteredFlattenedDataset() {
    if (this.tableFilterConfig.length === 0) {
      return this.flattenedDataset;
    }

    const filterFunction = this.createFilterFunction(this.tableFilterConfig);
    return this.flattenedDataset.filter(filterFunction);
  }

  get filteredOriginalDataset() {
    if (this.tableFilterConfig.length === 0) {
      return this.sortedDataset;
    }

    const filterFunction = this.createFilterFunction(this.tableFilterConfig);
    return this.sortedDataset.filter((row) => {
      const flattened = flattenJson(row);
      return filterFunction(flattened);
    });
  }

  get flattenedDatasetKeys() {
    const keySet = new Set<string>();
    this.flattenedDataset.forEach((row) => {
      Object.keys(row).forEach((key) => keySet.add(key));
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

  // Create filter function from filter group configuration
  private createFilterFunction(filterGroups: FilterGroup[]) {
    if (filterGroups.length === 0) return () => true;

    return (record: any) => {
      return filterGroups.every((group) => {
        if (group.filters.length === 0) return true;

        if (group.logic === "OR") {
          // For OR logic, at least one filter must pass
          return group.filters.some((filter) =>
            this.evaluateFilter(filter, record)
          );
        } else {
          // For AND logic, all filters must pass
          return group.filters.every((filter) =>
            this.evaluateFilter(filter, record)
          );
        }
      });
    };
  }

  // Helper function to evaluate a single filter
  private evaluateFilter(filter: any, record: any): boolean {
    if (!filter.field || !filter.value) return true; // Skip incomplete filters

    const fieldValue = record[filter.field];
    const filterValue = filter.value;
    const filterValue2 = filter.value2;

    // Handle date filtering
    if (filter.type === "date" || filter.type === "date-range") {
      const fieldDate = new Date(fieldValue);
      const valueDate = new Date(filterValue);

      if (isNaN(fieldDate.getTime()) || isNaN(valueDate.getTime())) {
        return true; // Skip invalid dates
      }

      switch (filter.operator) {
        case "==":
          return fieldDate.toDateString() === valueDate.toDateString();
        case "!=":
          return fieldDate.toDateString() !== valueDate.toDateString();
        case ">=":
          return fieldDate >= valueDate;
        case "<=":
          return fieldDate <= valueDate;
        case "between":
          if (filterValue2) {
            const valueDate2 = new Date(filterValue2);
            if (!isNaN(valueDate2.getTime())) {
              return fieldDate >= valueDate && fieldDate <= valueDate2;
            }
          }
          return true; // Skip incomplete between filter
        default:
          return true;
      }
    }

    // Handle text/numeric filtering
    switch (filter.operator) {
      case "==":
        return String(fieldValue) === filterValue;
      case "!=":
        return String(fieldValue) !== filterValue;
      case ">":
        return Number(fieldValue) > Number(filterValue);
      case "<":
        return Number(fieldValue) < Number(filterValue);
      case ">=":
        return Number(fieldValue) >= Number(filterValue);
      case "<=":
        return Number(fieldValue) <= Number(filterValue);
      case "contains":
        return String(fieldValue)
          .toLowerCase()
          .includes(filterValue.toLowerCase());
      case "!contains":
        return !String(fieldValue)
          .toLowerCase()
          .includes(filterValue.toLowerCase());
      default:
        return true;
    }
  }
}
