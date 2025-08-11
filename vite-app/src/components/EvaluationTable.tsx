import { observer } from "mobx-react";
import { useState, useEffect } from "react";
import { state } from "../App";
import { EvaluationRow } from "./EvaluationRow";
import Button from "./Button";

const TableBody = observer(
  ({ currentPage, pageSize }: { currentPage: number; pageSize: number }) => {
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    const paginatedData = state.sortedDataset.slice(startIndex, endIndex);

    return (
      <tbody className="divide-y divide-gray-200">
        {paginatedData.map((row, index) => (
          <EvaluationRow
            key={row.rollout_id}
            row={row}
            index={startIndex + index}
          />
        ))}
      </tbody>
    );
  }
);

// Dedicated component for rendering the list - following MobX best practices
export const EvaluationTable = observer(() => {
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);

  const totalRows = state.sortedDataset.length;
  const totalPages = Math.ceil(totalRows / pageSize);
  const startRow = (currentPage - 1) * pageSize + 1;
  const endRow = Math.min(currentPage * pageSize, totalRows);

  const handlePageChange = (page: number) => {
    setCurrentPage(Math.max(1, Math.min(page, totalPages)));
  };

  const handlePageSizeChange = (newPageSize: number) => {
    setPageSize(newPageSize);
    setCurrentPage(1); // Reset to first page when changing page size
  };

  // Reset to first page when dataset changes
  useEffect(() => {
    setCurrentPage(1);
  }, [totalRows]);

  return (
    <div className="bg-white border border-gray-200 overflow-x-auto">
      {/* Pagination Controls */}
      <div className="px-3 py-2 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="text-xs text-gray-600">
            Showing {startRow}-{endRow} of {totalRows} rows
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-600">Page size:</label>
            <select
              value={pageSize}
              onChange={(e) => handlePageSizeChange(Number(e.target.value))}
              className="text-xs border border-gray-300 rounded px-2 py-1 bg-white"
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
            </select>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            onClick={() => handlePageChange(1)}
            disabled={currentPage === 1}
            size="sm"
            variant="secondary"
          >
            First
          </Button>
          <Button
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage === 1}
            size="sm"
            variant="secondary"
          >
            Previous
          </Button>
          <span className="text-xs text-gray-600 px-2">
            Page {currentPage} of {totalPages}
          </span>
          <Button
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            size="sm"
            variant="secondary"
          >
            Next
          </Button>
          <Button
            onClick={() => handlePageChange(totalPages)}
            disabled={currentPage === totalPages}
            size="sm"
            variant="secondary"
          >
            Last
          </Button>
        </div>
      </div>

      <table className="w-full min-w-max">
        {/* Table Header */}
        <thead className="bg-gray-50 border-b border-gray-200">
          <tr>
            <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700 w-8">
              {/* Expand/Collapse column */}
            </th>
            <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">
              Name
            </th>
            <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">
              Status
            </th>
            <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">
              Rollout ID
            </th>
            <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">
              Model
            </th>
            <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">
              Score
            </th>
            <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">
              Created
            </th>
          </tr>
        </thead>

        {/* Table Body */}
        <TableBody currentPage={currentPage} pageSize={pageSize} />
      </table>
    </div>
  );
});
