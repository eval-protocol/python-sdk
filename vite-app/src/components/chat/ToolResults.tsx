import type { ToolResult } from "../../services/AgentService";

interface ToolResultsProps {
  toolResults: ToolResult[];
}

export const ToolResults = ({ toolResults }: ToolResultsProps) => {
  if (!toolResults || toolResults.length === 0) return null;

  return (
    <div className="space-y-3">
      {toolResults.map((result) => (
        <ToolResultItem key={result.id} result={result} />
      ))}
    </div>
  );
};

interface ToolResultItemProps {
  result: ToolResult;
}

const ToolResultItem = ({ result }: ToolResultItemProps) => {
  if (!result.success) {
    return (
      <div className="bg-red-50 border border-red-200 rounded p-3">
        <div className="text-sm text-red-800">
          <strong>Tool Error:</strong> {result.error}
        </div>
      </div>
    );
  }

  if (!result.data) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded p-3">
        <div className="text-sm text-gray-600">No data returned</div>
      </div>
    );
  }

  const renderContent = () => {
    switch (result.visualizationType) {
      case "chart":
        return <ChartVisualization data={result.data} />;
      case "text":
        return <TextVisualization data={result.data} />;
      case "insight":
        return <InsightVisualization data={result.data} />;
      case "table":
      default:
        return <TableVisualization data={result.data} />;
    }
  };

  return (
    <div className="bg-green-50 border border-green-200 rounded p-3">
      <div className="text-xs text-green-700 mb-2 font-medium">
        Analysis Result
      </div>
      {renderContent()}
    </div>
  );
};

// Table visualization for structured data
const TableVisualization = ({ data }: { data: any[] }) => {
  if (!Array.isArray(data) || data.length === 0) {
    return <div className="text-sm text-gray-600">No data to display</div>;
  }

  const columns = Object.keys(data[0] || {});

  return (
    <div className="overflow-x-auto max-w-full">
      <table className="w-full text-xs table-fixed">
        <thead>
          <tr className="border-b border-gray-300">
            {columns.map((column) => (
              <th
                key={column}
                className="text-left py-2 px-2 font-medium text-gray-700 truncate"
                style={{ width: `${100 / columns.length}%` }}
              >
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 10).map((row, index) => (
            <tr key={index} className="border-b border-gray-200">
              {columns.map((column) => (
                <td key={column} className="py-2 px-2 text-gray-600 truncate">
                  {String(row[column] || "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {data.length > 10 && (
        <div className="text-xs text-gray-500 mt-2 text-center">
          Showing 10 of {data.length} results
        </div>
      )}
    </div>
  );
};

// Chart visualization (placeholder - you can integrate with a charting library)
const ChartVisualization = ({ data }: { data: any[] }) => {
  // For now, just show a table with a note about charting
  return (
    <div>
      <div className="text-xs text-gray-600 mb-2">
        📊 Chart visualization (integrate with charting library)
      </div>
      <TableVisualization data={data} />
    </div>
  );
};

// Text visualization for insights and summaries
const TextVisualization = ({ data }: { data: any }) => {
  if (Array.isArray(data)) {
    return (
      <div className="space-y-2">
        {data.map((item, index) => (
          <div key={index} className="text-sm text-gray-700">
            {String(item)}
          </div>
        ))}
      </div>
    );
  }

  return <div className="text-sm text-gray-700">{String(data)}</div>;
};

// Insight visualization for high-level analysis
const InsightVisualization = ({ data }: { data: any }) => {
  return (
    <div className="space-y-2">
      <div className="text-sm font-medium text-gray-800">💡 Key Insights</div>
      <div className="text-sm text-gray-700">
        {Array.isArray(data) ? data.join(" • ") : String(data)}
      </div>
    </div>
  );
};
