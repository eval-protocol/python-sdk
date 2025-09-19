export type Operator =
  | "=="
  | "!="
  | ">"
  | "<"
  | ">="
  | "<="
  | "contains"
  | "!contains"
  | "between";

export type FilterType = "text" | "date" | "date-range";

// Filter configuration interface
export interface FilterConfig {
  field: string;
  operator: Operator;
  value: string;
  value2?: string; // For filtering between dates
  type?: FilterType;
}

export type FilterOperator = {
  value: Operator;
  label: string;
};

export type FilterLogic = "AND" | "OR";

// Filter group interface for AND/OR logic
export interface FilterGroup {
  logic: FilterLogic;
  filters: FilterConfig[];
}

// Pivot configuration interface
export interface PivotConfig {
  selectedRowFields: string[];
  selectedColumnFields: string[];
  selectedValueField: string;
  selectedAggregator: string;
}

export interface PaginationConfig {
  currentPage: number;
  pageSize: number;
}

export type SortDirection = "asc" | "desc";

export interface SortConfig {
  sortField: string;
  sortDirection: SortDirection;
}

export interface GlobalConfig {
  pivotConfig: PivotConfig;
  filterConfig: FilterGroup[];
  paginationConfig: PaginationConfig;
  sortConfig: SortConfig;
}
