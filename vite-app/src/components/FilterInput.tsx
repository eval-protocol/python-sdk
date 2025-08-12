import React from "react";
import type { FilterConfig } from "../types/filters";
import { commonStyles } from "../styles/common";

interface FilterInputProps {
  filter: FilterConfig;
  onUpdate: (updates: Partial<FilterConfig>) => void;
}

const FilterInput = ({ filter, onUpdate }: FilterInputProps) => {
  const fieldType = filter.type || "text";

  if (fieldType === "date") {
    return (
      <div className="flex space-x-2">
        <input
          type="date"
          value={filter.value}
          onChange={(e) => onUpdate({ value: e.target.value })}
          className={`${commonStyles.input.base} ${commonStyles.input.size.sm} ${commonStyles.width.sm}`}
          style={{ boxShadow: commonStyles.input.shadow }}
        />
        {filter.operator === "between" && (
          <input
            type="date"
            value={filter.value2 || ""}
            onChange={(e) => onUpdate({ value2: e.target.value })}
            className={`${commonStyles.input.base} ${commonStyles.input.size.sm} ${commonStyles.width.sm}`}
            placeholder="End date"
            style={{ boxShadow: commonStyles.input.shadow }}
          />
        )}
      </div>
    );
  }

  return (
    <input
      type="text"
      value={filter.value}
      onChange={(e) => onUpdate({ value: e.target.value })}
      placeholder="Value"
      className={`${commonStyles.input.base} ${commonStyles.input.size.sm} ${commonStyles.width.sm}`}
      style={{ boxShadow: commonStyles.input.shadow }}
    />
  );
};

export default FilterInput;
