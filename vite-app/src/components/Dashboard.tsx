import { observer } from "mobx-react";
import { state } from "../App";
import Button from "./Button";
import { Row } from "./Row";

interface DashboardProps {
  onRefresh: () => void;
}

const EmptyState = ({ onRefresh }: { onRefresh: () => void }) => {
  const handleRefresh = () => {
    onRefresh();
  };

  return (
    <div className="bg-white border border-gray-200 p-8 text-center">
      <div className="max-w-sm mx-auto">
        <div className="text-gray-400 mb-4">
          <svg
            className="mx-auto h-12 w-12"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
        </div>
        <h3 className="text-sm font-medium text-gray-900 mb-2">
          No evaluation data available
        </h3>
        <p className="text-xs text-gray-500 mb-4">
          No evaluation rows have been loaded yet. Click refresh to reconnect
          and load data.
        </p>
        <Button onClick={handleRefresh} size="md">
          Refresh
        </Button>
      </div>
    </div>
  );
};

const Dashboard = observer(({ onRefresh }: DashboardProps) => {
  return (
    <div className="text-sm">
      {/* Summary Stats */}
      <div className="mb-4 bg-white border border-gray-200 p-3">
        <h2 className="text-sm font-semibold text-gray-900 mb-2">
          Dataset Summary
        </h2>
        <table className="w-full text-xs">
          <tbody>
            <tr>
              <td className="pr-4">
                <span className="font-semibold text-gray-700">Total Rows:</span>{" "}
                {state.dataset.length}
              </td>
              <td className="pr-4">
                <span className="font-semibold text-gray-700">Avg Score:</span>{" "}
                {state.dataset.length > 0
                  ? (
                      state.dataset.reduce(
                        (sum, row) => sum + (row.evaluation_result?.score || 0),
                        0
                      ) / state.dataset.length
                    ).toFixed(3)
                  : "N/A"}
              </td>
              <td>
                <span className="font-semibold text-gray-700">
                  Total Messages:
                </span>{" "}
                {state.dataset.reduce(
                  (sum, row) => sum + row.messages.length,
                  0
                )}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Show empty state or main table */}
      {state.dataset.length === 0 ? (
        <EmptyState onRefresh={onRefresh} />
      ) : (
        <div className="bg-white border border-gray-200">
          <table className="w-full">
            {/* Table Header */}
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700 w-8">
                  {/* Expand/Collapse column */}
                </th>
                <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">
                  Row ID
                </th>
                <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">
                  Model
                </th>
                <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">
                  Score
                </th>
                <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">
                  Messages
                </th>
              </tr>
            </thead>

            {/* Table Body */}
            <tbody className="divide-y divide-gray-200">
              {state.dataset.map((row, index) => (
                <Row key={index} row={row} index={index} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
});

export default Dashboard;
