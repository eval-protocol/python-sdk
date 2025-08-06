import { useState } from "react";

export const MetadataSection = ({
  title,
  data,
  defaultExpanded = false,
}: {
  title: string;
  data: any;
  defaultExpanded?: boolean;
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  if (!data || Object.keys(data).length === 0) return null;

  return (
    <div className="mb-2">
      <div
        className="flex items-center justify-between cursor-pointer hover:bg-gray-50 p-1 rounded"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h4 className="font-semibold text-xs text-gray-700">{title}</h4>
        <svg
          className={`h-3 w-3 text-gray-500 transition-transform duration-200 ${
            isExpanded ? "rotate-180" : ""
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
      {isExpanded && (
        <div className="border border-gray-200 p-2 text-xs bg-white mt-1">
          <pre className="whitespace-pre-wrap overflow-x-auto">
            {JSON.stringify(data, null, 1)}
          </pre>
        </div>
      )}
    </div>
  );
};
