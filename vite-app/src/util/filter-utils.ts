import type { FilterConfig, FilterGroup, FilterOperator } from "../types/filters";

// Filter utilities
export const isDateField = (field: string): boolean => {
  return (
    field.toLowerCase().includes("date") ||
    field.toLowerCase().includes("time") ||
    field.toLowerCase().includes("created") ||
    field.toLowerCase().includes("updated")
  );
};

export const getFieldType = (field: string): "text" | "date" | "date-range" => {
  return isDateField(field) ? "date" : "text";
};

export const getOperatorsForField = (field: string, type?: string): FilterOperator[] => {
  if (type === "date" || type === "date-range" || isDateField(field)) {
    return [
      { value: ">=", label: "on or after" },
      { value: "<=", label: "on or before" },
      { value: "==", label: "on" },
      { value: "!=", label: "not on" },
      { value: "between", label: "between" },
    ];
  }

  return [
    { value: "contains", label: "contains" },
    { value: "!contains", label: "not contains" },
    { value: "==", label: "equals" },
    { value: "!=", label: "not equals" },
    { value: ">", label: "greater than" },
    { value: "<", label: "less than" },
    { value: ">=", label: "greater than or equal" },
    { value: "<=", label: "less than or equal" },
  ];
};

// Create filter function from filter group configuration
export const createFilterFunction = (filterGroups: FilterGroup[]) => {
  if (filterGroups.length === 0) return undefined;

  return (record: any) => {
    return filterGroups.every((group) => {
      if (group.filters.length === 0) return true;

      if (group.logic === "OR") {
        // For OR logic, at least one filter must pass
        return group.filters.some((filter) => evaluateFilter(filter, record));
      } else {
        // For AND logic, all filters must pass
        return group.filters.every((filter) => evaluateFilter(filter, record));
      }
    });
  };
};

// Helper function to evaluate a single filter
const evaluateFilter = (filter: FilterConfig, record: any): boolean => {
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
};
