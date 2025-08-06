import { observer } from "mobx-react";
import { state } from "../App";
import Button from "./Button";
import { EvaluationTable } from "./EvaluationTable";

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
  const expandAll = () => state.setAllRowsExpanded(true);
  const collapseAll = () => state.setAllRowsExpanded(false);

  return (
    <div className="text-sm">
      {/* Summary Stats */}
      <div className="mb-4 bg-white border border-gray-200 p-3">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-sm font-semibold text-gray-900">
            Dataset Summary
          </h2>
          {state.totalCount > 0 && (
            <div className="flex gap-2">
              <Button onClick={expandAll} size="sm" variant="secondary">
                Expand All
              </Button>
              <Button onClick={collapseAll} size="sm" variant="secondary">
                Collapse All
              </Button>
            </div>
          )}
        </div>
        <div className="text-xs space-y-1">
          <div>
            <span className="font-semibold text-gray-700">Total Rows:</span>{" "}
            {state.totalCount}
          </div>
        </div>
      </div>

      {/* Show empty state or main table */}
      {state.totalCount === 0 ? (
        <EmptyState onRefresh={onRefresh} />
      ) : (
        <EvaluationTable />
      )}
    </div>
  );
});

export default Dashboard;
