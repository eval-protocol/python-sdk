import { observer } from "mobx-react";
import PivotTable from "./PivotTable";
import Select from "./Select";
import Button from "./Button";
import FilterInput from "./FilterInput";
import { state } from "../App";
import { type FilterConfig } from "../types/filters";
import {
  getFieldType,
  getOperatorsForField,
  createFilterFunction,
} from "../util/filter-utils";

interface FieldSelectorProps {
  title: string;
  fields: string[];
  onFieldChange: (index: number, value: string) => void;
  onAddField: () => void;
  onRemoveField: (index: number) => void;
  availableKeys: string[];
}

const FieldSelector = ({
  title,
  fields,
  onFieldChange,
  onAddField,
  onRemoveField,
  availableKeys,
  variant = "default",
}: FieldSelectorProps & { variant?: "row" | "column" | "default" }) => {
  const variantStyles = {
    row: "border-l-4 border-l-blue-500 pl-3",
    column: "border-l-4 border-l-green-500 pl-3",
    default: "",
  };

  return (
    <div className={`mb-4 ${variantStyles[variant]}`}>
      <div className="text-xs font-medium text-gray-700 mb-2">{title}:</div>
      <div className="space-y-2">
        {fields.map((field, index) => (
          <div key={index} className="flex items-center space-x-2">
            <Select
              value={field}
              onChange={(e) => onFieldChange(index, e.target.value)}
              size="sm"
              className="min-w-48"
            >
              <option value="">Select a field...</option>
              {availableKeys?.map((key) => (
                <option key={key} value={key}>
                  {key}
                </option>
              ))}
            </Select>
            {fields.length > 0 && (
              <button
                onClick={() => onRemoveField(index)}
                className="text-xs text-red-600 hover:text-red-800 px-2 py-1"
              >
                Remove
              </button>
            )}
          </div>
        ))}
        {fields.length < 3 && (
          <button
            onClick={onAddField}
            className="text-xs text-blue-600 hover:text-blue-800 px-2 py-1"
          >
            + Add {title.slice(0, -1)}
          </button>
        )}
      </div>
    </div>
  );
};

const SingleFieldSelector = ({
  title,
  field,
  onFieldChange,
  availableKeys,
}: {
  title: string;
  field: string;
  onFieldChange: (value: string) => void;
  availableKeys: string[];
}) => (
  <div className="mb-4">
    <div className="text-xs font-medium text-gray-700 mb-2">{title}:</div>
    <Select
      value={field}
      onChange={(e) => onFieldChange(e.target.value)}
      size="sm"
      className="min-w-48"
    >
      <option value="">Select a field...</option>
      {availableKeys?.map((key) => (
        <option key={key} value={key}>
          {key}
        </option>
      ))}
    </Select>
  </div>
);

const AggregatorSelector = ({
  aggregator,
  onAggregatorChange,
}: {
  aggregator: string;
  onAggregatorChange: (value: string) => void;
}) => (
  <div className="mb-4">
    <div className="text-xs font-medium text-gray-700 mb-2">
      Aggregation Method:
    </div>
    <Select
      value={aggregator}
      onChange={(e) => onAggregatorChange(e.target.value)}
      size="sm"
      className="min-w-48"
    >
      <option value="count">Count</option>
      <option value="sum">Sum</option>
      <option value="avg">Average</option>
      <option value="min">Minimum</option>
      <option value="max">Maximum</option>
    </Select>
  </div>
);

const FilterSelector = ({
  filters,
  onFiltersChange,
  availableKeys,
}: {
  filters: FilterConfig[];
  onFiltersChange: (filters: FilterConfig[]) => void;
  availableKeys: string[];
}) => {
  const addFilter = () => {
    onFiltersChange([
      ...filters,
      { field: "", operator: "contains", value: "", type: "text" },
    ]);
  };

  const removeFilter = (index: number) => {
    onFiltersChange(filters.filter((_, i) => i !== index));
  };

  const updateFilter = (index: number, updates: Partial<FilterConfig>) => {
    const newFilters = [...filters];
    newFilters[index] = { ...newFilters[index], ...updates };
    onFiltersChange(newFilters);
  };

  return (
    <div className="mb-4">
      <div className="text-xs font-medium text-gray-700 mb-2">Filters:</div>
      <div className="space-y-2">
        {filters.map((filter, index) => {
          const fieldType = filter.type || getFieldType(filter.field);
          const operators = getOperatorsForField(filter.field, fieldType);

          return (
            <div key={index} className="flex items-center space-x-2">
              <Select
                value={filter.field}
                onChange={(e) => {
                  const newField = e.target.value;
                  const newType = getFieldType(newField);
                  updateFilter(index, { field: newField, type: newType });
                }}
                size="sm"
                className="min-w-48"
              >
                <option value="">Select a field...</option>
                {availableKeys?.map((key) => (
                  <option key={key} value={key}>
                    {key}
                  </option>
                ))}
              </Select>
              <Select
                value={filter.operator}
                onChange={(e) =>
                  updateFilter(index, { operator: e.target.value })
                }
                size="sm"
                className="min-w-32"
              >
                {operators.map((op) => (
                  <option key={op.value} value={op.value}>
                    {op.label}
                  </option>
                ))}
              </Select>
              <FilterInput
                filter={filter}
                index={index}
                onUpdate={(updates) => updateFilter(index, updates)}
              />
              <button
                onClick={() => removeFilter(index)}
                className="text-xs text-red-600 hover:text-red-800 px-2 py-1"
              >
                Remove
              </button>
            </div>
          );
        })}
        <button
          onClick={addFilter}
          className="text-xs text-blue-600 hover:text-blue-800 px-2 py-1"
        >
          + Add Filter
        </button>
      </div>
    </div>
  );
};

const PivotTab = observer(() => {
  const { pivotConfig } = state;

  const updateRowFields = (index: number, value: string) => {
    const newRowFields = [...pivotConfig.selectedRowFields];
    newRowFields[index] = value;
    state.updatePivotConfig({ selectedRowFields: newRowFields });
  };

  const updateColumnFields = (index: number, value: string) => {
    const newColumnFields = [...pivotConfig.selectedColumnFields];
    newColumnFields[index] = value;
    state.updatePivotConfig({ selectedColumnFields: newColumnFields });
  };

  const updateValueField = (value: string) => {
    state.updatePivotConfig({ selectedValueField: value });
  };

  const updateAggregator = (value: string) => {
    state.updatePivotConfig({ selectedAggregator: value });
  };

  const updateFilters = (filters: FilterConfig[]) => {
    state.updatePivotConfig({ filters });
  };

  const createFieldHandler = (
    updater: (index: number, value: string) => void
  ) => {
    return (index: number, value: string) => {
      updater(index, value);
    };
  };

  const createAddHandler = (
    fields: string[],
    updater: (fields: string[]) => void
  ) => {
    return () => {
      if (fields.length < 3) {
        updater([...fields, ""]);
      }
    };
  };

  const createRemoveHandler = (
    fields: string[],
    updater: (fields: string[]) => void
  ) => {
    return (index: number) => {
      updater(fields.filter((_, i) => i !== index));
    };
  };

  const availableKeys = state.flattenedDatasetKeys;

  return (
    <div>
      <div className="text-xs text-gray-600 mb-2 max-w-2xl">
        Answer questions about your dataset by creating pivot tables that
        summarize and analyze your data. Select fields for rows, columns, and
        values to explore patterns, compare metrics across different dimensions,
        and gain insights from your evaluation results. Use filters to focus on
        specific subsets of your data.
      </div>

      {/* Controls Section with Reset Button */}
      <div className="mb-4 flex justify-between items-center">
        <Button
          onClick={() => state.resetPivotConfig()}
          variant="secondary"
          size="sm"
        >
          Reset to Defaults
        </Button>
      </div>

      <FieldSelector
        title="Row Fields"
        fields={pivotConfig.selectedRowFields}
        onFieldChange={createFieldHandler(updateRowFields)}
        onAddField={createAddHandler(pivotConfig.selectedRowFields, (fields) =>
          state.updatePivotConfig({ selectedRowFields: fields })
        )}
        onRemoveField={createRemoveHandler(
          pivotConfig.selectedRowFields,
          (fields) => state.updatePivotConfig({ selectedRowFields: fields })
        )}
        availableKeys={availableKeys}
        variant="row"
      />

      <FieldSelector
        title="Column Fields"
        fields={pivotConfig.selectedColumnFields}
        onFieldChange={createFieldHandler(updateColumnFields)}
        onAddField={createAddHandler(
          pivotConfig.selectedColumnFields,
          (fields) => state.updatePivotConfig({ selectedColumnFields: fields })
        )}
        onRemoveField={createRemoveHandler(
          pivotConfig.selectedColumnFields,
          (fields) => state.updatePivotConfig({ selectedColumnFields: fields })
        )}
        availableKeys={availableKeys}
        variant="column"
      />

      <SingleFieldSelector
        title="Value Field"
        field={pivotConfig.selectedValueField}
        onFieldChange={updateValueField}
        availableKeys={availableKeys}
      />

      <AggregatorSelector
        aggregator={pivotConfig.selectedAggregator}
        onAggregatorChange={updateAggregator}
      />

      <FilterSelector
        filters={pivotConfig.filters}
        onFiltersChange={updateFilters}
        availableKeys={availableKeys}
      />

      <PivotTable
        data={state.flattenedDataset}
        rowFields={
          pivotConfig.selectedRowFields.filter(
            (field) => field !== ""
          ) as (keyof (typeof state.flattenedDataset)[number])[]
        }
        columnFields={
          pivotConfig.selectedColumnFields.filter(
            (field) => field !== ""
          ) as (keyof (typeof state.flattenedDataset)[number])[]
        }
        valueField={
          pivotConfig.selectedValueField as keyof (typeof state.flattenedDataset)[number]
        }
        aggregator={pivotConfig.selectedAggregator as any}
        showRowTotals
        showColumnTotals
        filter={createFilterFunction(pivotConfig.filters)}
      />
    </div>
  );
});

export default PivotTab;
