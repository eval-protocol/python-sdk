import { observer } from "mobx-react";
import { state } from "../App";
import { EvaluationRow } from "./EvaluationRow";

const TableBody = observer(() => {
  return (
    <tbody className="divide-y divide-gray-200">
      {state.sortedDataset.map((row, index) => (
        <EvaluationRow key={row.rollout_id} row={row} index={index} />
      ))}
    </tbody>
  );
});

// Dedicated component for rendering the list - following MobX best practices
export const EvaluationTable = observer(() => {
  return (
    <div className="bg-white border border-gray-200 overflow-x-auto">
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
        <TableBody />
      </table>
    </div>
  );
});
