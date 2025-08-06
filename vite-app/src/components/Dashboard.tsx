import { observer } from "mobx-react";
import { useState } from "react";
import { state } from "../App";
import type { EvaluationRow, Message } from "../types/eval-protocol";

const MessageBubble = ({ message }: { message: Message }) => {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const hasToolCalls = message.tool_calls && message.tool_calls.length > 0;
  const hasFunctionCall = message.function_call;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-2`}>
      <div
        className={`max-w-xs lg:max-w-md px-3 py-2 border text-sm ${
          isUser
            ? "bg-blue-50 border-blue-200 text-blue-900"
            : isSystem
            ? "bg-gray-50 border-gray-200 text-gray-800"
            : hasToolCalls || hasFunctionCall
            ? "bg-orange-50 border-orange-200 text-orange-900"
            : "bg-white border-gray-200 text-gray-900"
        }`}
      >
        <div className="font-semibold text-xs mb-1 capitalize">
          {message.role}
          {(hasToolCalls || hasFunctionCall) && (
            <span className="ml-1 text-orange-600">• Tool Call</span>
          )}
        </div>
        <div className="whitespace-pre-wrap">
          {typeof message.content === "string"
            ? message.content
            : Array.isArray(message.content)
            ? message.content
                .map((part, i) =>
                  part.type === "text" ? part.text : JSON.stringify(part)
                )
                .join("")
            : JSON.stringify(message.content)}
        </div>
        {hasToolCalls && message.tool_calls && (
          <div className="mt-3 pt-2 border-t border-orange-200">
            <div className="font-semibold text-xs text-orange-700 mb-1">
              Tool Calls:
            </div>
            {message.tool_calls.map((call, i) => (
              <div
                key={i}
                className="mb-2 p-2 bg-orange-100 border border-orange-200 rounded text-xs"
              >
                <div className="font-semibold text-orange-800 mb-1">
                  {call.function.name}
                </div>
                <div className="text-orange-700 font-mono text-xs">
                  {call.function.arguments}
                </div>
              </div>
            ))}
          </div>
        )}
        {hasFunctionCall && message.function_call && (
          <div className="mt-3 pt-2 border-t border-orange-200">
            <div className="font-semibold text-xs text-orange-700 mb-1">
              Function Call:
            </div>
            <div className="p-2 bg-orange-100 border border-orange-200 rounded text-xs">
              <div className="font-semibold text-orange-800 mb-1">
                {message.function_call.name}
              </div>
              <div className="text-orange-700 font-mono text-xs">
                {message.function_call.arguments}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const MetadataSection = ({ title, data }: { title: string; data: any }) => {
  if (!data || Object.keys(data).length === 0) return null;

  return (
    <div className="mb-2">
      <h4 className="font-semibold text-xs text-gray-700 mb-1">{title}</h4>
      <div className="bg-gray-50 border border-gray-200 p-2 text-xs">
        <pre className="whitespace-pre-wrap overflow-x-auto">
          {JSON.stringify(data, null, 1)}
        </pre>
      </div>
    </div>
  );
};

const Row = observer(
  ({ row, index }: { row: EvaluationRow; index: number }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    const toggleExpanded = () => setIsExpanded(!isExpanded);

    return (
      <div className="border-b border-gray-200">
        {/* Collapsed Row - Shows Metadata */}
        <div
          className="flex items-center p-3 hover:bg-gray-50 cursor-pointer text-sm"
          onClick={toggleExpanded}
        >
          <div className="mr-2">
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
          </div>
          <div className="flex-1 grid grid-cols-4 gap-4 text-xs">
            <div className="font-mono">
              <span className="font-semibold text-gray-600">ID:</span>{" "}
              <span className="text-gray-900">{row.input_metadata.row_id}</span>
            </div>
            <div>
              <span className="font-semibold text-gray-600">Model:</span>{" "}
              <span className="text-gray-900">
                {row.input_metadata.completion_params?.model || "N/A"}
              </span>
            </div>
            <div>
              <span className="font-semibold text-gray-600">Score:</span>{" "}
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
            </div>
            <div>
              <span className="font-semibold text-gray-600">Messages:</span>{" "}
              <span className="text-gray-900">{row.messages.length}</span>
            </div>
          </div>
        </div>

        {/* Expanded Content */}
        {isExpanded && (
          <div className="p-4 bg-gray-50 border-t border-gray-200">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left Column - Chat Interface */}
              <div>
                <h4 className="font-semibold text-sm text-gray-700 mb-2 border-b border-gray-200 pb-1">
                  Conversation ({row.messages.length} messages)
                </h4>
                <div className="bg-white border border-gray-200 p-4 max-h-96 overflow-y-auto">
                  {row.messages.map((message, msgIndex) => (
                    <MessageBubble key={msgIndex} message={message} />
                  ))}
                </div>
              </div>

              {/* Right Column - Metadata */}
              <div className="space-y-3">
                <h4 className="font-semibold text-sm text-gray-700 mb-2 border-b border-gray-200 pb-1">
                  Metadata
                </h4>

                {/* Evaluation Result - Most Important */}
                {row.evaluation_result && (
                  <div className="mb-4">
                    <h5 className="font-semibold text-sm text-gray-700 mb-2">
                      Evaluation Result
                    </h5>
                    <div className="bg-gray-50 border border-gray-200 p-3 text-xs">
                      <div className="grid grid-cols-2 gap-3 mb-3">
                        <div>
                          <span className="font-semibold text-gray-600">
                            Score:
                          </span>{" "}
                          <span
                            className={`font-mono text-sm ${
                              row.evaluation_result.score >= 0.8
                                ? "text-green-700"
                                : row.evaluation_result.score >= 0.6
                                ? "text-yellow-700"
                                : "text-red-700"
                            }`}
                          >
                            {row.evaluation_result.score}
                          </span>
                        </div>
                        <div>
                          <span className="font-semibold text-gray-600">
                            Valid:
                          </span>{" "}
                          <span
                            className={
                              row.evaluation_result.is_score_valid
                                ? "text-green-700"
                                : "text-red-700"
                            }
                          >
                            {row.evaluation_result.is_score_valid
                              ? "Yes"
                              : "No"}
                          </span>
                        </div>
                      </div>
                      {row.evaluation_result.reason && (
                        <div className="mb-3 p-3 bg-white border border-gray-200">
                          <span className="font-semibold text-gray-600">
                            Reason:
                          </span>{" "}
                          <span className="text-gray-900">
                            {row.evaluation_result.reason}
                          </span>
                        </div>
                      )}
                      {row.evaluation_result.metrics &&
                        Object.keys(row.evaluation_result.metrics).length >
                          0 && (
                          <div>
                            <span className="font-semibold text-gray-600">
                              Metrics:
                            </span>
                            <pre className="mt-2 p-3 bg-white border border-gray-200 whitespace-pre-wrap overflow-x-auto text-xs">
                              {JSON.stringify(
                                row.evaluation_result.metrics,
                                null,
                                1
                              )}
                            </pre>
                          </div>
                        )}
                    </div>
                  </div>
                )}

                {/* Ground Truth - Secondary Importance */}
                {row.ground_truth && (
                  <div className="mb-3">
                    <h5 className="font-semibold text-xs text-gray-700 mb-1">
                      Ground Truth
                    </h5>
                    <div className="bg-gray-50 border border-gray-200 p-2 text-xs">
                      <div className="whitespace-pre-wrap text-gray-900">
                        {row.ground_truth}
                      </div>
                    </div>
                  </div>
                )}

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
        )}
      </div>
    );
  }
);

const Dashboard = observer(() => {
  return (
    <div className="text-sm">
      {/* Summary Stats */}
      <div className="mb-4 bg-white border border-gray-200 p-3">
        <h2 className="text-sm font-semibold text-gray-900 mb-2">
          Dataset Summary
        </h2>
        <div className="grid grid-cols-4 gap-4 text-xs">
          <div>
            <span className="font-semibold text-gray-700">Total Rows:</span>{" "}
            {state.dataset.length}
          </div>
          <div>
            <span className="font-semibold text-gray-700">Avg Score:</span>{" "}
            {state.dataset.length > 0
              ? (
                  state.dataset.reduce(
                    (sum, row) => sum + (row.evaluation_result?.score || 0),
                    0
                  ) / state.dataset.length
                ).toFixed(3)
              : "N/A"}
          </div>
          <div>
            <span className="font-semibold text-gray-700">Connected:</span>{" "}
            {
              state.dataset.filter(
                (row) => row.evaluation_result?.is_score_valid
              ).length
            }
          </div>
          <div>
            <span className="font-semibold text-gray-700">Total Messages:</span>{" "}
            {state.dataset.reduce((sum, row) => sum + row.messages.length, 0)}
          </div>
        </div>
      </div>

      {/* Main Table */}
      <div className="bg-white border border-gray-200">
        {/* Table Header */}
        <div className="bg-gray-50 px-3 py-2 border-b border-gray-200">
          <div className="grid grid-cols-4 gap-4 text-xs font-semibold text-gray-700">
            <div>Row ID</div>
            <div>Model</div>
            <div>Score</div>
            <div>Messages</div>
          </div>
        </div>

        {/* Table Rows */}
        <div className="divide-y divide-gray-200">
          {state.dataset.map((row, index) => (
            <Row key={index} row={row} index={index} />
          ))}
        </div>
      </div>
    </div>
  );
});

export default Dashboard;
