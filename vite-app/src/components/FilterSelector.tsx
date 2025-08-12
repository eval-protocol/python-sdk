import type { FilterConfig, FilterGroup } from "../types/filters";
import SearchableSelect from "./SearchableSelect";
import FilterInput from "./FilterInput";
import Button from "./Button";
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
    <div className="mb-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-semibold text-gray-700">{title}</h3>
        {filters.length > 0 && (
          <Button onClick={addFilterGroup} size="sm" variant="secondary">
            + Add Filter Group
          </Button>
        )}
      </div>

      <div className="space-y-2">
        {filters.map((group, groupIndex) => (
          <div
            key={groupIndex}
            className={`pt-2 ${
              groupIndex > 0 ? "border-t border-gray-200 mt-2" : ""
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-gray-600">
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
                  className="min-w-40"
                />
              </div>
              <Button
                onClick={() => removeFilterGroup(groupIndex)}
                size="sm"
                variant="secondary"
              >
                Remove Group
              </Button>
            </div>

            <div className="space-y-2">
              {group.filters.map((filter, filterIndex) => {
                const fieldType = filter.type || getFieldType(filter.field);
                const operators = getOperatorsForField(fieldType);

                return (
                  <div key={filterIndex} className="flex items-center gap-2">
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
                      className="min-w-40"
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
                      className="min-w-36"
                    />
                    <FilterInput
                      filter={filter}
                      onUpdate={(updates) =>
                        updateFilterInGroup(groupIndex, filterIndex, updates)
                      }
                    />
                    <Button
                      onClick={() =>
                        removeFilterFromGroup(groupIndex, filterIndex)
                      }
                      size="sm"
                      variant="secondary"
                    >
                      Remove
                    </Button>
                  </div>
                );
              })}

              <div>
                <Button
                  onClick={() => addFilterToGroup(groupIndex)}
                  size="sm"
                  variant="secondary"
                >
                  + Add Filter to Group
                </Button>
              </div>
            </div>
          </div>
        ))}

        {filters.length === 0 && (
          <div className="flex justify-center py-4">
            <Button onClick={addFilterGroup} size="sm" variant="primary">
              + Add Filter Group
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default FilterSelector;
