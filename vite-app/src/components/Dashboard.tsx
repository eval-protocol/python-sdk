import { observer } from "mobx-react";
import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { state } from "../App";
import Button from "./Button";
import { EvaluationTable } from "./EvaluationTable";
import PivotTab from "./PivotTab";
import TabButton from "./TabButton";

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

const LoadingState = () => {
  return (
    <div className="bg-white border border-gray-200 p-8 text-center">
      <div className="max-w-sm mx-auto">
        <div className="text-gray-400 mb-4">
          <div className="animate-spin rounded-full h-6 w-6 border-2 border-gray-300 border-t-gray-600 mx-auto"></div>
        </div>
        <h3 className="text-sm font-medium text-gray-900 mb-2">
          Loading evaluation data...
        </h3>
        <p className="text-xs text-gray-500">
          Connecting to the server and loading data
        </p>
      </div>
    </div>
  );
};

const Dashboard = observer(({ onRefresh }: DashboardProps) => {
  const expandAll = () => state.setAllRowsExpanded(true);
  const collapseAll = () => state.setAllRowsExpanded(false);

  const location = useLocation();
  const navigate = useNavigate();

  const deriveTabFromPath = (path: string): "table" | "pivot" =>
    path.endsWith("/pivot") ? "pivot" : "table";

  const [activeTab, setActiveTab] = useState<"table" | "pivot">(
    deriveTabFromPath(location.pathname)
  );

  useEffect(() => {
    setActiveTab(deriveTabFromPath(location.pathname));
  }, [location.pathname]);

  return (
    <div className="text-sm">
      {/* Summary */}
      <div className="mb-4 bg-white border border-gray-200 p-3">
        <h2 className="text-sm font-semibold text-gray-900 mb-2">
          Dataset Summary
        </h2>
        <div className="text-xs">
          <span className="font-semibold text-gray-700">Total Rows:</span>{" "}
          {state.totalCount}
        </div>
      </div>

      {/* Content Area */}
      {state.isLoading ? (
        <LoadingState />
      ) : state.totalCount === 0 ? (
        <EmptyState onRefresh={onRefresh} />
      ) : (
        <div className="bg-white border border-gray-200">
          {/* Tabs + contextual actions */}
          <div className="px-3 pt-2 border-b  border-gray-200">
            <div className="flex justify-between h-8">
              <div id="tabs" className="flex gap-1">
                <TabButton
                  label="Table"
                  isActive={activeTab === "table"}
                  onClick={() => {
                    setActiveTab("table");
                    navigate("/table");
                  }}
                  title="View table"
                />
                <TabButton
                  label="Pivot"
                  isActive={activeTab === "pivot"}
                  onClick={() => {
                    setActiveTab("pivot");
                    navigate("/pivot");
                  }}
                  title="View pivot"
                />
              </div>
              {activeTab === "table" && (
                <div className="flex gap-2 pb-2">
                  <Button onClick={expandAll} size="sm" variant="secondary">
                    Expand All
                  </Button>
                  <Button onClick={collapseAll} size="sm" variant="secondary">
                    Collapse All
                  </Button>
                </div>
              )}
            </div>
          </div>

          {/* Tab content */}
          <div className="p-3">
            {activeTab === "table" ? <EvaluationTable /> : <PivotTab />}
          </div>
        </div>
      )}
    </div>
  );
});

export default Dashboard;
