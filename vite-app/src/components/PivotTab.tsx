import { observer } from "mobx-react";
import PivotTable from "./PivotTable";
import Select from "./Select";
import { state } from "../App";
import { useState } from "react";

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
            + Add {title.slice(0, -1)} Field
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

const PivotTab = observer(() => {
  const [selectedRowFields, setSelectedRowFields] = useState<string[]>([
    "$.eval_metadata.name",
  ]);
  const [selectedColumnFields, setSelectedColumnFields] = useState<string[]>([
    "$.input_metadata.completion_params.model",
  ]);
  const [selectedValueField, setSelectedValueField] = useState<string>(
    "$.evaluation_result.score"
  );
  const [selectedAggregator, setSelectedAggregator] = useState<string>("avg");

  const createFieldHandler = (
    setter: React.Dispatch<React.SetStateAction<string[]>>
  ) => {
    return (index: number, value: string) => {
      setter((prev) => {
        const newFields = [...prev];
        newFields[index] = value;
        return newFields;
      });
    };
  };

  const createAddHandler = (
    setter: React.Dispatch<React.SetStateAction<string[]>>
  ) => {
    return () => {
      setter((prev) => (prev.length < 3 ? [...prev, ""] : prev));
    };
  };

  const createRemoveHandler = (
    setter: React.Dispatch<React.SetStateAction<string[]>>
  ) => {
    return (index: number) => {
      setter((prev) => prev.filter((_, i) => i !== index));
    };
  };

  const availableKeys = state.flattenedDatasetKeys[0] || [];

  return (
    <div>
      <div className="text-xs text-gray-600 mb-2 max-w-2xl">
        Configure your pivot table by selecting fields for rows, columns, and
        values. Use the dropdowns below to choose from available flattened
        JSONPath keys. You can add/remove fields and change the value field to
        pivot on different metrics.
      </div>

      <FieldSelector
        title="Row Fields"
        fields={selectedRowFields}
        onFieldChange={createFieldHandler(setSelectedRowFields)}
        onAddField={createAddHandler(setSelectedRowFields)}
        onRemoveField={createRemoveHandler(setSelectedRowFields)}
        availableKeys={availableKeys}
        variant="row"
      />

      <FieldSelector
        title="Column Fields"
        fields={selectedColumnFields}
        onFieldChange={createFieldHandler(setSelectedColumnFields)}
        onAddField={createAddHandler(setSelectedColumnFields)}
        onRemoveField={createRemoveHandler(setSelectedColumnFields)}
        availableKeys={availableKeys}
        variant="column"
      />

      <SingleFieldSelector
        title="Value Field"
        field={selectedValueField}
        onFieldChange={setSelectedValueField}
        availableKeys={availableKeys}
      />

      <AggregatorSelector
        aggregator={selectedAggregator}
        onAggregatorChange={setSelectedAggregator}
      />

      <PivotTable
        data={state.flattenedDataset}
        rowFields={
          selectedRowFields.filter(
            (field) => field !== ""
          ) as (keyof (typeof state.flattenedDataset)[number])[]
        }
        columnFields={
          selectedColumnFields.filter(
            (field) => field !== ""
          ) as (keyof (typeof state.flattenedDataset)[number])[]
        }
        valueField={
          selectedValueField as keyof (typeof state.flattenedDataset)[number]
        }
        aggregator={selectedAggregator as any}
        showRowTotals
        showColumnTotals
      />
    </div>
  );
});

export default PivotTab;
