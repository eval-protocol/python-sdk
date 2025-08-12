import React, { useState, useRef, useEffect } from "react";
import { commonStyles } from "../styles/common";

interface SearchableSelectProps {
  options: { value: string; label: string }[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  size?: "sm" | "md";
  className?: string;
  disabled?: boolean;
}

const SearchableSelect = React.forwardRef<
  HTMLDivElement,
  SearchableSelectProps
>(
  (
    {
      options,
      value,
      onChange,
      placeholder = "Select...",
      size = "sm",
      className = "",
      disabled = false,
    },
    ref
  ) => {
    const [isOpen, setIsOpen] = useState(false);
    const [searchTerm, setSearchTerm] = useState("");
    const [filteredOptions, setFilteredOptions] = useState(options);
    const containerRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
      const filtered = options.filter(
        (option) =>
          option.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
          option.value.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredOptions(filtered);
    }, [searchTerm, options]);

    useEffect(() => {
      const handleClickOutside = (event: MouseEvent) => {
        if (
          containerRef.current &&
          !containerRef.current.contains(event.target as Node)
        ) {
          setIsOpen(false);
          setSearchTerm("");
        }
      };

      document.addEventListener("mousedown", handleClickOutside);
      return () =>
        document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    const handleSelect = (optionValue: string) => {
      onChange(optionValue);
      setIsOpen(false);
      setSearchTerm("");
    };

    const handleToggle = () => {
      if (!disabled) {
        setIsOpen(!isOpen);
        if (!isOpen) {
          setTimeout(() => inputRef.current?.focus(), 0);
        }
      }
    };

    const selectedOption = options.find((option) => option.value === value);

    return (
      <div ref={containerRef} className={`relative ${className}`}>
        <div
          ref={ref}
          onClick={handleToggle}
          className={`
						${commonStyles.input.base}
						${commonStyles.input.size[size]}
						cursor-pointer flex items-center justify-between
						${disabled ? "opacity-50 cursor-not-allowed" : ""}
					`}
          style={{ boxShadow: commonStyles.input.shadow }}
        >
          <span className={value ? "text-gray-900" : "text-gray-400"}>
            {selectedOption ? selectedOption.label : placeholder}
          </span>
          <svg
            className={`w-4 h-4 text-gray-400 transition-transform ${
              isOpen ? "rotate-180" : ""
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>

        {isOpen && (
          <div className="absolute z-50 w-max min-w-full mt-1 bg-white border border-gray-200 rounded-md max-h-60 overflow-auto">
            <div className="p-2 border-b border-gray-200">
              <input
                ref={inputRef}
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search..."
                className={`${commonStyles.input.base} ${commonStyles.input.size.sm} w-full`}
                style={{ boxShadow: commonStyles.input.shadow }}
                onKeyDown={(e) => {
                  if (e.key === "Escape") {
                    setIsOpen(false);
                    setSearchTerm("");
                  }
                }}
              />
            </div>
            <div className="max-h-48 overflow-auto">
              {filteredOptions.length > 0 ? (
                filteredOptions.map((option) => (
                  <div
                    key={option.value}
                    onClick={() => handleSelect(option.value)}
                    className="px-3 py-2 text-sm cursor-pointer hover:bg-gray-50 text-gray-700 border-b border-gray-100 last:border-b-0"
                  >
                    {option.label}
                  </div>
                ))
              ) : (
                <div className="px-3 py-2 text-sm text-gray-500">
                  No options found
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    );
  }
);

SearchableSelect.displayName = "SearchableSelect";

export default SearchableSelect;
