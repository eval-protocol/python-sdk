import React, {
  useState,
  useRef,
  useEffect,
  useMemo,
  useLayoutEffect,
} from "react";
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
    // Memoize filtering to avoid extra state updates and re-renders
    const filteredOptions = useMemo(() => {
      const lowered = searchTerm.toLowerCase();
      if (!lowered) return options;
      return options.filter(
        (option) =>
          option.label.toLowerCase().includes(lowered) ||
          option.value.toLowerCase().includes(lowered)
      );
    }, [searchTerm, options]);
    const [dropdownPosition, setDropdownPosition] = useState<"left" | "right">(
      "left"
    );
    const [dropdownWidth, setDropdownWidth] = useState<number | undefined>(
      undefined
    );
    const [highlightedIndex, setHighlightedIndex] = useState(-1);
    const containerRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // Reset highlighted index when the search or options change
    useEffect(() => {
      setHighlightedIndex(-1);
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

    // Compute side and width so the dropdown can be wider than the trigger but still fit viewport
    const computeDropdownLayout = (): {
      side: "left" | "right";
      width: number;
    } => {
      if (!containerRef.current) return { side: "left", width: 240 };

      const rect = containerRef.current.getBoundingClientRect();
      const windowWidth = window.innerWidth;
      const VIEWPORT_MARGIN = 16; // px
      const EXTRA_WIDTH = 240; // desired extra room beyond trigger
      const MAX_WIDTH = 600; // hard cap

      const desired = Math.min(rect.width + EXTRA_WIDTH, MAX_WIDTH);

      const spaceRight = windowWidth - rect.left - VIEWPORT_MARGIN; // space if anchored left-0
      const spaceLeft = rect.right - VIEWPORT_MARGIN; // space if anchored right-0

      const widthRight = Math.max(rect.width, Math.min(desired, spaceRight));
      const widthLeft = Math.max(rect.width, Math.min(desired, spaceLeft));

      // Prefer the side that can accommodate closer to desired width
      if (widthRight >= widthLeft && widthRight >= rect.width) {
        return { side: "left", width: widthRight };
      }
      if (widthLeft > widthRight && widthLeft >= rect.width) {
        return { side: "right", width: widthLeft };
      }

      // Fallback: clamp to viewport with preference to right anchoring if less overflow
      const clamped = Math.max(
        rect.width,
        Math.min(desired, windowWidth - VIEWPORT_MARGIN * 2)
      );
      const preferRight = rect.left < windowWidth - rect.right;
      return { side: preferRight ? "left" : "right", width: clamped };
    };

    const handleToggle = () => {
      if (!disabled) {
        if (!isOpen) {
          const layout = computeDropdownLayout();
          setDropdownPosition(layout.side);
          setDropdownWidth(layout.width);
        }
        setIsOpen(!isOpen);
        if (!isOpen) {
          setTimeout(() => inputRef.current?.focus(), 0);
        }
      }
    };

    const selectedOption = useMemo(
      () => options.find((option) => option.value === value),
      [options, value]
    );

    // --- Simple list virtualization for large option sets ---
    const listRef = useRef<HTMLDivElement>(null);
    const [scrollTop, setScrollTop] = useState(0);
    const [containerHeight, setContainerHeight] = useState(192); // Tailwind max-h-48 (~192px)

    const ITEM_HEIGHT = 32; // Approximate item height in px
    const OVERSCAN = 5;

    const totalItems = filteredOptions.length;
    const visibleCount = Math.max(1, Math.ceil(containerHeight / ITEM_HEIGHT));
    const startIndex = Math.max(
      0,
      Math.floor(scrollTop / ITEM_HEIGHT) - OVERSCAN
    );
    const endIndex = Math.min(
      totalItems,
      startIndex + visibleCount + OVERSCAN * 2
    );
    const topPaddingHeight = startIndex * ITEM_HEIGHT;
    const bottomPaddingHeight = Math.max(
      0,
      (totalItems - endIndex) * ITEM_HEIGHT
    );

    // Measure container height when open and on resize
    useLayoutEffect(() => {
      if (!isOpen) return;
      const el = listRef.current;
      if (!el) return;

      const measure = () => {
        setContainerHeight(el.clientHeight || 192);
      };
      measure();

      const ro = new ResizeObserver(measure);
      ro.observe(el);
      return () => ro.disconnect();
    }, [isOpen]);

    // Reset scroll when opening or when search changes
    useEffect(() => {
      if (isOpen && listRef.current) {
        listRef.current.scrollTop = 0;
        setScrollTop(0);
      }
    }, [isOpen, searchTerm]);

    const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
      setScrollTop((e.target as HTMLDivElement).scrollTop);
    };

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
          <span
            className={value ? "text-gray-900" : "text-gray-400"}
            title={selectedOption ? selectedOption.label : placeholder}
          >
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
            className={`absolute z-50 mt-1 ${
              commonStyles.input.base
            } max-h-60 overflow-x-hidden ${
              dropdownPosition === "right" ? "right-0" : "left-0"
            }`}
            style={{
              width: dropdownWidth,
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
                className={`${commonStyles.input.base} ${commonStyles.input.size.sm} w-full`}
                style={{ boxShadow: commonStyles.input.shadow }}
                role="searchbox"
                aria-label="Search options"
              />
            </div>
            <div
              ref={listRef}
              className="max-h-48 overflow-y-auto overflow-x-hidden"
              role="listbox"
              aria-label="Options"
              onScroll={handleScroll}
            >
              {totalItems > 0 ? (
                <>
                  {topPaddingHeight > 0 && (
                    <div style={{ height: topPaddingHeight }} />
                  )}
                  {filteredOptions
                    .slice(startIndex, endIndex)
                    .map((option, i) => {
                      const absoluteIndex = startIndex + i;
                      return (
                        <div
                          key={option.value}
                          onClick={() => handleSelect(option.value)}
                          onMouseEnter={() =>
                            setHighlightedIndex(absoluteIndex)
                          }
                          className={`px-3 py-2 text-xs font-medium cursor-pointer hover:bg-gray-100 text-gray-700 border-b border-gray-100 last:border-b-0 ${
                            highlightedIndex === absoluteIndex
                              ? "bg-gray-100"
                              : ""
                          }`}
                          role="option"
                          aria-selected={highlightedIndex === absoluteIndex}
                          aria-label={option.label}
                          tabIndex={-1}
                          style={{ height: ITEM_HEIGHT }}
                          title={option.label}
                        >
                          <div className="overflow-x-auto whitespace-nowrap">
                            {option.label}
                          </div>
                        </div>
                      );
                    })}
                  {bottomPaddingHeight > 0 && (
                    <div style={{ height: bottomPaddingHeight }} />
                  )}
                </>
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
