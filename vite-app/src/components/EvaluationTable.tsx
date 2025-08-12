import { observer } from "mobx-react";
import { useState, useEffect } from "react";
import { state } from "../App";
import { EvaluationRow } from "./EvaluationRow";
import Button from "./Button";
import Select from "./Select";
import {
  TableHeader,
  TableHead,
  TableBody as TableBodyBase,
} from "./TableContainer";

const TableBody = observer(
  ({ currentPage, pageSize }: { currentPage: number; pageSize: number }) => {
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    const paginatedData = state.sortedDataset.slice(startIndex, endIndex);

    return (
      <TableBodyBase>
        {paginatedData.map((row, index) => (
          <EvaluationRow
            key={row.execution_metadata?.rollout_id}
            row={row}
            index={startIndex + index}
          />
        ))}
      </TableBodyBase>
    );
  }
);

// Dedicated component for rendering the list - following MobX best practices
export const EvaluationTable = observer(() => {
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

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
    <div className="bg-white border border-gray-200">
      {/* Pagination Controls - Fixed outside scrollable area */}
      <div className="px-3 py-2 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="text-xs text-gray-600">
            Showing {startRow}-{endRow} of {totalRows} rows
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-600">Page size:</label>
            <Select
              value={pageSize}
              onChange={(e) => handlePageSizeChange(Number(e.target.value))}
              size="sm"
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
            </Select>
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

      {/* Table Container - Only this area scrolls */}
      <div className="overflow-x-auto">
        <table className="w-full min-w-max">
          {/* Table Header */}
          <TableHead>
            <tr>
              <TableHeader className="w-8">&nbsp;</TableHeader>
              <TableHeader>Name</TableHeader>
              <TableHeader>Status</TableHeader>
              <TableHeader>Rollout ID</TableHeader>
              <TableHeader>Model</TableHeader>
              <TableHeader>Score</TableHeader>
              <TableHeader>Created</TableHeader>
            </tr>
          </TableHead>

          {/* Table Body */}
          <TableBody currentPage={currentPage} pageSize={pageSize} />
        </table>
      </div>
    </div>
  );
});
