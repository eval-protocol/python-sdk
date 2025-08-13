// Filter configuration interface
export interface FilterConfig {
  field: string;
  operator: string;
  value: string;
  value2?: string; // For filtering between dates
  type?: "text" | "date" | "date-range";
}

export interface FilterOperator {
  value: string;
  label: string;
}

// Filter group interface for AND/OR logic
export interface FilterGroup {
  logic: "AND" | "OR";
  filters: FilterConfig[];
}

// Pivot configuration interface
export interface PivotConfig {
  selectedRowFields: string[];
  selectedColumnFields: string[];
  selectedValueField: string;
  selectedAggregator: string;
  filters: FilterGroup[];
}
