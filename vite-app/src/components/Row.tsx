import { observer } from "mobx-react";
import type { EvaluationRow } from "../types/eval-protocol";
import { ChatInterface } from "./ChatInterface";
import { MetadataSection } from "./MetadataSection";
import StatusIndicator from "./StatusIndicator";
import { state } from "../App";

export const Row = observer(
  ({ row }: { row: EvaluationRow; index: number }) => {
    const rowId = row.input_metadata.row_id;
    const isExpanded = state.isRowExpanded(rowId);

    const toggleExpanded = () => state.toggleRowExpansion(rowId);

    return (
      <>
        {/* Main Table Row */}
        <tr
          className="hover:bg-gray-50 cursor-pointer text-sm border-b border-gray-200"
          onClick={toggleExpanded}
        >
          {/* Expand/Collapse Icon */}
          <td className="px-3 py-3 w-8">
            <div className="w-4 h-4 flex items-center justify-center">
              <svg
                className={`h-4 w-4 text-gray-500 transition-transform duration-200 ${
                  isExpanded ? "rotate-90" : ""
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5l7 7-7 7"
                />
              </svg>
            </div>
          </td>

          {/* Name */}
          <td className="px-3 py-3 text-xs">
            <span className="text-gray-900 truncate block">
              {row.eval_metadata?.name || "N/A"}
            </span>
          </td>

          {/* Status */}
          <td className="px-3 py-3 text-xs">
            <div className="whitespace-nowrap">
              <StatusIndicator
                showSpinner={row.eval_metadata?.status === "running"}
                status={row.eval_metadata?.status || "N/A"}
              />
            </div>
          </td>

          {/* Row ID */}
          <td className="px-3 py-3 text-xs">
            <span className="font-mono text-gray-900 whitespace-nowrap">
              {row.input_metadata.row_id}
            </span>
          </td>

          {/* Model */}
          <td className="px-3 py-3 text-xs">
            <span className="text-gray-900 truncate block">
              {row.input_metadata.completion_params?.model || "N/A"}
            </span>
          </td>

          {/* Score */}
          <td className="px-3 py-3 text-xs">
            <span
              className={`font-mono whitespace-nowrap ${
                row.evaluation_result?.score
                  ? row.evaluation_result.score >= 0.8
                    ? "text-green-700"
                    : row.evaluation_result.score >= 0.6
                    ? "text-yellow-700"
                    : "text-red-700"
                  : "text-gray-500"
              }`}
            >
              {row.evaluation_result?.score?.toFixed(3) || "N/A"}
            </span>
          </td>

          {/* Created */}
          <td className="px-3 py-3 text-xs">
            <span className="text-gray-600 whitespace-nowrap">
              {row.created_at instanceof Date
                ? row.created_at.toLocaleDateString() +
                  " " +
                  row.created_at.toLocaleTimeString()
                : new Date(row.created_at).toLocaleDateString() +
                  " " +
                  new Date(row.created_at).toLocaleTimeString()}
            </span>
          </td>
        </tr>

        {/* Expanded Content Row */}
        {isExpanded && (
          <tr>
            <td colSpan={8} className="p-0">
              <div className="p-4 bg-gray-50 border-t border-gray-200">
                <div className="flex gap-3 w-fit">
                  {/* Left Column - Chat Interface */}
                  <div className="min-w-0">
                    <ChatInterface messages={row.messages} />
                  </div>

                  {/* Right Column - Metadata */}
                  <div className="w-[500px] flex-shrink-0 space-y-3">
                    {/* Eval Metadata */}
                    <MetadataSection
                      title="Eval Metadata"
                      data={row.eval_metadata}
                      defaultExpanded={true}
                    />

                    {/* Evaluation Result */}
                    <MetadataSection
                      title="Evaluation Result"
                      data={row.evaluation_result}
                      defaultExpanded={true}
                    />

                    {/* Ground Truth */}
                    <MetadataSection
                      title="Ground Truth"
                      data={row.ground_truth}
                    />

                    {/* Usage Stats - Compact */}
                    <MetadataSection title="Usage Stats" data={row.usage} />

                    {/* Input Metadata - Less Important */}
                    <MetadataSection
                      title="Input Metadata"
                      data={row.input_metadata}
                    />

                    {/* Tools - Least Important */}
                    <MetadataSection title="Tools" data={row.tools} />
                  </div>
                </div>
              </div>
            </td>
          </tr>
        )}
      </>
    );
  }
);
