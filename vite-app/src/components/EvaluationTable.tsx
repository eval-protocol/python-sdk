import { observer } from "mobx-react";
import { state } from "../App";
import { EvaluationRow } from "./EvaluationRow";
import Button from "./Button";
import Select from "./Select";
import FilterSelector from "./FilterSelector";
import {
  TableHeader,
  TableHead,
  TableBody as TableBodyBase,
  SortableTableHeader,
} from "./TableContainer";

const TableBody = observer(
  ({ currentPage, pageSize }: { currentPage: number; pageSize: number }) => {
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    // Use filtered original data for pagination
    const filteredData = state.filteredOriginalDataset;
    const paginatedData = filteredData.slice(startIndex, endIndex);

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
  const totalRows = state.filteredOriginalDataset.length;
  const totalPages = Math.ceil(totalRows / state.pageSize);
  const startRow = (state.currentPage - 1) * state.pageSize + 1;
  const endRow = Math.min(state.currentPage * state.pageSize, totalRows);

  const handlePageChange = (page: number) => {
    state.setCurrentPage(Math.max(1, Math.min(page, totalPages)));
  };

  const handlePageSizeChange = (newPageSize: number) => {
    state.setPageSize(newPageSize);
  };

  const handleFiltersChange = (filters: any[]) => {
    state.updateFilterConfig(filters);
  };

  const handleSort = (field: string) => {
    state.handleSortFieldClick(field);
  };

  return (
    <div className="bg-white border border-gray-200">
      {/* Filter Controls */}
      <div className="px-3 py-3 border-b border-gray-200 bg-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h3 className="text-sm font-medium text-gray-700">Table Filters</h3>
            <div className="text-xs text-gray-600">
              {state.filterConfig.length > 0 ? (
                <>
                  Showing {totalRows} of {state.sortedDataset.length} rows
                  {totalRows !== state.sortedDataset.length && (
                    <span className="text-blue-600 ml-1">(filtered)</span>
                  )}
                </>
              ) : (
                `Showing all ${state.sortedDataset.length} rows`
              )}
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg">
          <FilterSelector
            filters={state.filterConfig}
            onFiltersChange={handleFiltersChange}
            availableKeys={state.flattenedDatasetKeys}
            title=""
          />
        </div>
      </div>

      {/* Pagination Controls - Fixed outside scrollable area */}
      <div className="px-3 py-2 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="text-xs text-gray-600">
            Showing {startRow}-{endRow} of {totalRows} rows
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-600">Page size:</label>
            <Select
              value={state.pageSize}
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
            disabled={state.currentPage === 1}
            size="sm"
            variant="secondary"
          >
            First
          </Button>
          <Button
            onClick={() => handlePageChange(state.currentPage - 1)}
            disabled={state.currentPage === 1}
            size="sm"
            variant="secondary"
          >
            Previous
          </Button>
          <span className="text-xs text-gray-600 px-2">
            Page {state.currentPage} of {totalPages}
          </span>
          <Button
            onClick={() => handlePageChange(state.currentPage + 1)}
            disabled={state.currentPage === totalPages}
            size="sm"
            variant="secondary"
          >
            Next
          </Button>
          <Button
            onClick={() => handlePageChange(totalPages)}
            disabled={state.currentPage === totalPages}
            size="sm"
            variant="secondary"
          >
            Last
          </Button>
        </div>
      </div>

      {/* Table Container - Only this area scrolls */}
      {totalRows === 0 ? (
        <div className="px-3 py-6 text-center text-xs text-gray-600">
          <div className="mb-2">No rows match your current filters.</div>
          <Button
            onClick={() => handleFiltersChange([])}
            size="sm"
            variant="secondary"
          >
            Clear filters
          </Button>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-max">
            {/* Table Header */}
            <TableHead>
              <tr>
                <TableHeader className="w-8">&nbsp;</TableHeader>
                <SortableTableHeader
                  sortField="created_at"
                  currentSortField={state.sortField}
                  currentSortDirection={state.sortDirection}
                  onSort={handleSort}
                >
                  Created
                </SortableTableHeader>
                <SortableTableHeader
                  sortField="$.eval_metadata.name"
                  currentSortField={state.sortField}
                  currentSortDirection={state.sortDirection}
                  onSort={handleSort}
                >
                  Name
                </SortableTableHeader>
                <SortableTableHeader
                  sortField="$.eval_metadata.status.code"
                  currentSortField={state.sortField}
                  currentSortDirection={state.sortDirection}
                  onSort={handleSort}
                >
                  Eval Status
                </SortableTableHeader>
                <SortableTableHeader
                  sortField="$.rollout_status.code"
                  currentSortField={state.sortField}
                  currentSortDirection={state.sortDirection}
                  onSort={handleSort}
                >
                  Rollout Status
                </SortableTableHeader>
                <SortableTableHeader
                  sortField="$.execution_metadata.invocation_id"
                  currentSortField={state.sortField}
                  currentSortDirection={state.sortDirection}
                  onSort={handleSort}
                >
                  Invocation ID
                </SortableTableHeader>
                <SortableTableHeader
                  sortField="$.execution_metadata.experiment_id"
                  currentSortField={state.sortField}
                  currentSortDirection={state.sortDirection}
                  onSort={handleSort}
                >
                  Experiment ID
                </SortableTableHeader>
                <SortableTableHeader
                  sortField="$.execution_metadata.run_id"
                  currentSortField={state.sortField}
                  currentSortDirection={state.sortDirection}
                  onSort={handleSort}
                >
                  Run ID
                </SortableTableHeader>
                <SortableTableHeader
                  sortField="$.input_metadata.row_id"
                  currentSortField={state.sortField}
                  currentSortDirection={state.sortDirection}
                  onSort={handleSort}
                >
                  Row ID
                </SortableTableHeader>
                <SortableTableHeader
                  sortField="$.execution_metadata.rollout_id"
                  currentSortField={state.sortField}
                  currentSortDirection={state.sortDirection}
                  onSort={handleSort}
                >
                  Rollout ID
                </SortableTableHeader>
                <SortableTableHeader
                  sortField="$.input_metadata.completion_params.model"
                  currentSortField={state.sortField}
                  currentSortDirection={state.sortDirection}
                  onSort={handleSort}
                >
                  Model
                </SortableTableHeader>
                <SortableTableHeader
                  sortField="$.evaluation_result.score"
                  currentSortField={state.sortField}
                  currentSortDirection={state.sortDirection}
                  onSort={handleSort}
                >
                  Score
                </SortableTableHeader>
              </tr>
            </TableHead>

            {/* Table Body */}
            <TableBody
              currentPage={state.currentPage}
              pageSize={state.pageSize}
            />
          </table>
        </div>
      )}
    </div>
  );
});
