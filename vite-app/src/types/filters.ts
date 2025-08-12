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

// Pivot configuration interface
export interface PivotConfig {
  selectedRowFields: string[];
  selectedColumnFields: string[];
  selectedValueField: string;
  selectedAggregator: string;
  filters: FilterConfig[];
}
