import React from "react";
import type { FilterConfig, FilterGroup } from "../types/filters";
import SearchableSelect from "./SearchableSelect";
import FilterInput from "./FilterInput";
import { getFieldType, getOperatorsForField } from "../util/filter-utils";

interface FilterSelectorProps {
  filters: FilterGroup[];
  onFiltersChange: (filters: FilterGroup[]) => void;
  availableKeys: string[];
  title?: string;
}

const FilterSelector = ({
  filters,
  onFiltersChange,
  availableKeys,
  title = "Filters",
}: FilterSelectorProps) => {
  const addFilterGroup = () => {
    onFiltersChange([...filters, { logic: "AND", filters: [] }]);
  };

  const removeFilterGroup = (index: number) => {
    onFiltersChange(filters.filter((_, i) => i !== index));
  };

  const updateFilterGroupLogic = (index: number, logic: "AND" | "OR") => {
    const newFilters = [...filters];
    newFilters[index] = { ...newFilters[index], logic };
    onFiltersChange(newFilters);
  };

  const addFilterToGroup = (groupIndex: number) => {
    const newFilters = [...filters];
    newFilters[groupIndex].filters.push({
      field: "",
      operator: "contains",
      value: "",
      type: "text",
    });
    onFiltersChange(newFilters);
  };

  const removeFilterFromGroup = (groupIndex: number, filterIndex: number) => {
    const newFilters = [...filters];
    newFilters[groupIndex].filters.splice(filterIndex, 1);
    onFiltersChange(newFilters);
  };

  const updateFilterInGroup = (
    groupIndex: number,
    filterIndex: number,
    updates: Partial<FilterConfig>
  ) => {
    const newFilters = [...filters];
    newFilters[groupIndex].filters[filterIndex] = {
      ...newFilters[groupIndex].filters[filterIndex],
      ...updates,
    };
    onFiltersChange(newFilters);
  };

  return (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700">{title}</h3>
        <button
          onClick={addFilterGroup}
          className="text-xs text-blue-600 hover:text-blue-800 px-2 py-1"
        >
          + Add Filter Group
        </button>
      </div>
      <div className="space-y-4">
        {filters.map((group, groupIndex) => (
          <div
            key={groupIndex}
            className="border border-gray-200 rounded-lg p-4 bg-gray-50"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-medium text-gray-700 bg-white px-2 py-1 rounded border border-gray-200">
                  Group {groupIndex + 1}
                </span>
                <SearchableSelect
                  value={group.logic}
                  onChange={(value) =>
                    updateFilterGroupLogic(groupIndex, value as "AND" | "OR")
                  }
                  options={[
                    { value: "AND", label: "AND (all filters must match)" },
                    { value: "OR", label: "OR (any filter can match)" },
                  ]}
                  size="sm"
                  className="min-w-48"
                />
              </div>
              <button
                onClick={() => removeFilterGroup(groupIndex)}
                className="text-xs text-red-600 hover:text-red-800 px-2 py-1 hover:bg-red-50 rounded"
              >
                Remove Group
              </button>
            </div>

            <div className="space-y-3 ml-4 pl-4 border-l-2 border-gray-200">
              {group.filters.map((filter, filterIndex) => {
                const fieldType = filter.type || getFieldType(filter.field);
                const operators = getOperatorsForField(fieldType);

                return (
                  <div
                    key={filterIndex}
                    className="flex items-center space-x-2 p-2 bg-white rounded border border-gray-200"
                  >
                    <SearchableSelect
                      value={filter.field}
                      onChange={(value) => {
                        const newType = getFieldType(value);
                        updateFilterInGroup(groupIndex, filterIndex, {
                          field: value,
                          type: newType,
                          operator: operators[0]?.value || "contains",
                        });
                      }}
                      options={availableKeys.map((key) => ({
                        value: key,
                        label: key,
                      }))}
                      placeholder="Select field..."
                      size="sm"
                      className="min-w-32"
                    />
                    <SearchableSelect
                      value={filter.operator}
                      onChange={(value) =>
                        updateFilterInGroup(groupIndex, filterIndex, {
                          operator: value,
                        })
                      }
                      options={operators}
                      size="sm"
                      className="min-w-32"
                    />
                    <FilterInput
                      filter={filter}
                      onUpdate={(updates) =>
                        updateFilterInGroup(groupIndex, filterIndex, updates)
                      }
                    />
                    <button
                      onClick={() =>
                        removeFilterFromGroup(groupIndex, filterIndex)
                      }
                      className="text-xs text-red-600 hover:text-red-800 px-2 py-1 hover:bg-red-50 rounded"
                    >
                      Remove
                    </button>
                  </div>
                );
              })}

              <button
                onClick={() => addFilterToGroup(groupIndex)}
                className="text-xs text-blue-600 hover:text-blue-800 px-2 py-1 hover:bg-blue-50 rounded border border-dashed border-blue-300 w-full"
              >
                + Add Filter to Group
              </button>
            </div>
          </div>
        ))}

        {filters.length === 0 && (
          <div className="text-xs text-gray-500 text-center py-4 px-3 bg-gray-50 border border-dashed border-gray-300 rounded-lg">
            No filters configured. Click "Add Filter Group" to start filtering.
          </div>
        )}

        <button
          onClick={addFilterGroup}
          className="text-xs text-blue-600 hover:text-blue-800 px-3 py-2 hover:bg-blue-50 rounded border border-dashed border-blue-300 w-full transition-colors"
        >
          + Add Filter Group
        </button>
      </div>
    </div>
  );
};

export default FilterSelector;
