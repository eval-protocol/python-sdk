import React, { useEffect, useRef, useState } from "react";
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

  // Debounced text input to reduce re-renders while typing
  const [localValue, setLocalValue] = useState<string>(filter.value || "");
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep local state in sync when external value changes (e.g., clearing filters)
  useEffect(() => {
    setLocalValue(filter.value || "");
  }, [filter.value]);

  const commit = (value: string) => {
    if (debounceTimer.current) clearTimeout(debounceTimer.current);
    onUpdate({ value });
  };

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const next = e.target.value;
    setLocalValue(next);
    if (debounceTimer.current) clearTimeout(debounceTimer.current);
    debounceTimer.current = setTimeout(() => {
      onUpdate({ value: next });
    }, 250);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      commit(localValue);
    }
  };

  const onBlur = () => commit(localValue);

  return (
    <input
      type="text"
      value={localValue}
      onChange={onChange}
      onKeyDown={onKeyDown}
      onBlur={onBlur}
      placeholder="Value"
      className={`${commonStyles.input.base} ${commonStyles.input.size.sm} ${commonStyles.width.sm}`}
      style={{ boxShadow: commonStyles.input.shadow }}
    />
  );
};

export default FilterInput;
