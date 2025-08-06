import { observer } from "mobx-react";
import { useState } from "react";
import type { EvaluationRow } from "../types/eval-protocol";
import { ChatInterface } from "./ChatInterface";
import { MetadataSection } from "./MetadataSection";

export const Row = observer(
  ({ row, index }: { row: EvaluationRow; index: number }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    const toggleExpanded = () => setIsExpanded(!isExpanded);

    return (
      <>
        {/* Main Table Row */}
        <tr
          className="hover:bg-gray-50 cursor-pointer text-sm border-b border-gray-200"
          onClick={toggleExpanded}
        >
          {/* Expand/Collapse Icon */}
          <td className="px-3 py-3 w-8">
            {isExpanded ? (
              <svg
                className="h-4 w-4 text-gray-500"
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
            ) : (
              <svg
                className="h-4 w-4 text-gray-500"
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
            )}
          </td>

          {/* Row ID */}
          <td className="px-3 py-3 text-xs">
            <span className="font-mono text-gray-900">
              {row.input_metadata.row_id}
            </span>
          </td>

          {/* Model */}
          <td className="px-3 py-3 text-xs">
            <span className="text-gray-900">
              {row.input_metadata.completion_params?.model || "N/A"}
            </span>
          </td>

          {/* Score */}
          <td className="px-3 py-3 text-xs">
            <span
              className={`font-mono ${
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

          {/* Messages */}
          <td className="px-3 py-3 text-xs">
            <span className="text-gray-900">{row.messages.length}</span>
          </td>
        </tr>

        {/* Expanded Content Row */}
        {isExpanded && (
          <tr>
            <td colSpan={5} className="p-0">
              <div className="p-4 bg-gray-50 border-t border-gray-200">
                <div className="flex gap-6">
                  {/* Left Column - Chat Interface */}
                  <ChatInterface messages={row.messages} />

                  {/* Right Column - Metadata */}
                  <div className="flex-1 space-y-3 min-w-0">
                    <h4 className="font-semibold text-sm text-gray-700 mb-2 pb-1">
                      Metadata
                    </h4>

                    {/* Evaluation Result */}
                    <MetadataSection
                      title="Evaluation Result"
                      data={row.evaluation_result}
                    />

                    {/* Ground Truth */}
                    <MetadataSection
                      title="Ground Truth"
                      data={row.ground_truth}
                    />

                    {/* Usage Stats - Compact */}
                    {row.usage && (
                      <div className="mb-2">
                        <h5 className="font-semibold text-xs text-gray-700 mb-1">
                          Token Usage
                        </h5>
                        <div className="bg-gray-50 border border-gray-200 p-2 text-xs">
                          <div className="grid grid-cols-3 gap-2">
                            <div>
                              <span className="font-semibold text-gray-600">
                                Prompt:
                              </span>{" "}
                              <span className="font-mono text-gray-900">
                                {row.usage.prompt_tokens}
                              </span>
                            </div>
                            <div>
                              <span className="font-semibold text-gray-600">
                                Completion:
                              </span>{" "}
                              <span className="font-mono text-gray-900">
                                {row.usage.completion_tokens}
                              </span>
                            </div>
                            <div>
                              <span className="font-semibold text-gray-600">
                                Total:
                              </span>{" "}
                              <span className="font-mono text-gray-900">
                                {row.usage.total_tokens}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Input Metadata - Less Important */}
                    <MetadataSection
                      title="Input Metadata"
                      data={row.input_metadata}
                    />

                    {/* Tools - Least Important */}
                    {row.tools && row.tools.length > 0 && (
                      <div className="mb-2">
                        <h5 className="font-semibold text-xs text-gray-700 mb-1">
                          Available Tools
                        </h5>
                        <div className="bg-gray-50 border border-gray-200 p-2 text-xs">
                          <pre className="whitespace-pre-wrap overflow-x-auto text-gray-900">
                            {JSON.stringify(row.tools, null, 1)}
                          </pre>
                        </div>
                      </div>
                    )}
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
