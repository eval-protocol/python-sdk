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
    const [dropdownPosition, setDropdownPosition] = useState<"left" | "right">(
      "left"
    );
    const [highlightedIndex, setHighlightedIndex] = useState(-1);
    const containerRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
      const filtered = options.filter(
        (option) =>
          option.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
          option.value.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredOptions(filtered);
      setHighlightedIndex(-1); // Reset highlighted index when options change
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
      setHighlightedIndex(-1);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
      if (!isOpen) return;

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setHighlightedIndex((prev) =>
            prev < filteredOptions.length - 1 ? prev + 1 : 0
          );
          break;
        case "ArrowUp":
          e.preventDefault();
          setHighlightedIndex((prev) =>
            prev > 0 ? prev - 1 : filteredOptions.length - 1
          );
          break;
        case "Enter":
          e.preventDefault();
          if (highlightedIndex >= 0 && filteredOptions[highlightedIndex]) {
            handleSelect(filteredOptions[highlightedIndex].value);
          }
          break;
        case "Escape":
          setIsOpen(false);
          setSearchTerm("");
          setHighlightedIndex(-1);
          break;
      }
    };

    const calculateDropdownPosition = () => {
      if (!containerRef.current) return "left";

      const rect = containerRef.current.getBoundingClientRect();
      const windowWidth = window.innerWidth;
      const estimatedDropdownWidth = 300; // Approximate width for dropdown content

      // If dropdown would overflow right edge, position it to the left
      if (rect.left + estimatedDropdownWidth > windowWidth) {
        return "right";
      }
      return "left";
    };

    const handleToggle = () => {
      if (!disabled) {
        if (!isOpen) {
          setDropdownPosition(calculateDropdownPosition());
        }
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
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              handleToggle();
            }
          }}
          tabIndex={0}
          role="combobox"
          aria-expanded={isOpen}
          aria-haspopup="listbox"
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
          <div
            className={`absolute z-50 w-max min-w-full mt-1 ${
              commonStyles.input.base
            } rounded-md max-h-60 overflow-auto ${
              dropdownPosition === "right" ? "right-0" : "left-0"
            }`}
            style={{
              maxWidth: `min(calc(100vw - 2rem), 500px)`,
              right: dropdownPosition === "right" ? "0" : undefined,
              left: dropdownPosition === "left" ? "0" : undefined,
              boxShadow: commonStyles.input.shadow,
            }}
          >
            <div
              className={`p-2 border-b border-gray-200 rounded-t-md`}
              style={{ boxShadow: commonStyles.input.shadow }}
            >
              <input
                ref={inputRef}
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Search..."
                className={`${commonStyles.input.base} ${commonStyles.input.size.sm} w-full min-w-48`}
                style={{ boxShadow: commonStyles.input.shadow }}
                role="searchbox"
                aria-label="Search options"
              />
            </div>
            <div
              className="max-h-48 overflow-auto"
              role="listbox"
              aria-label="Options"
            >
              {filteredOptions.length > 0 ? (
                filteredOptions.map((option, index) => (
                  <div
                    key={option.value}
                    onClick={() => handleSelect(option.value)}
                    onMouseEnter={() => setHighlightedIndex(index)}
                    className={`px-3 py-2 text-xs font-medium cursor-pointer hover:bg-gray-100 text-gray-700 border-b border-gray-100 last:border-b-0 ${
                      highlightedIndex === index ? "bg-gray-100" : ""
                    }`}
                    role="option"
                    aria-selected={highlightedIndex === index}
                    tabIndex={-1}
                  >
                    {option.label}
                  </div>
                ))
              ) : (
                <div className="px-3 py-2 text-xs font-medium text-gray-500">
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
