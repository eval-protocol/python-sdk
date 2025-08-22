import { observer } from "mobx-react";
import type {
  EvaluationRow as EvaluationRowType,
  Status,
} from "../types/eval-protocol";
import { ChatInterface } from "./ChatInterface";
import { MetadataSection } from "./MetadataSection";
import StatusIndicator from "./StatusIndicator";
import { state } from "../App";
import { TableCell, TableRowInteractive } from "./TableContainer";
import { useState } from "react";
import type { FilterGroup, FilterConfig } from "../types/filters";
import { Tooltip } from "./Tooltip";

// Add filter button component
const AddFilterButton = observer(
  ({
    fieldPath,
    value,
    label,
  }: {
    fieldPath: string;
    value: string;
    label: string;
  }) => {
    const [added, setAdded] = useState(false);

    const handleClick = (e: React.MouseEvent) => {
      e.stopPropagation(); // Prevent row expansion

      // Create a new filter for this field/value
      const newFilter: FilterConfig = {
        field: fieldPath,
        operator: "==",
        value: value,
        type: "text",
      };

      // Add the filter to the existing filter configuration
      const currentFilters = state.filterConfig;
      let newFilters: FilterGroup[];

      if (currentFilters.length === 0) {
        // If no filters exist, create a new filter group
        newFilters = [
          {
            logic: "AND",
            filters: [newFilter],
          },
        ];
      } else {
        // Add to the first filter group (assuming AND logic)
        newFilters = [...currentFilters];
        newFilters[0] = {
          ...newFilters[0],
          filters: [...newFilters[0].filters, newFilter],
        };
      }

      state.updateFilterConfig(newFilters);
      setAdded(true);

      // Reset to "Add Filter" state after 2 seconds
      setTimeout(() => setAdded(false), 2000);
    };

    return (
      <Tooltip
        content={added ? "Filter Added!" : `Add ${label} Filter`}
        position="top"
        className="text-gray-400 hover:text-gray-600 transition-colors"
      >
        <div className="flex items-center gap-1">
          <button
            className="cursor-pointer"
            onClick={handleClick}
            title="Add filter for this value"
          >
            {/* Icon */}
            {added ? (
              <svg
                className="w-3 h-3 text-green-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
            ) : (
              <svg
                className="w-3 h-3"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.207A1 1 0 013 6.5V4z"
                />
              </svg>
            )}
          </button>
        </div>
      </Tooltip>
    );
  }
);

// Small, focused components following "dereference values late" principle
const ExpandIcon = observer(({ rolloutId }: { rolloutId?: string }) => {
  if (!rolloutId) {
    throw new Error("Rollout ID is required");
  }
  const isExpanded = state.isRowExpanded(rolloutId);
  return (
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
  );
});

const RowName = observer(({ name }: { name: string | undefined }) => (
  <span className="text-gray-900 truncate block">{name || "N/A"}</span>
));

const RowStatus = observer(
  ({
    status,
    showSpinner,
  }: {
    status: Status | undefined;
    showSpinner: boolean;
  }) => (
    <div className="whitespace-nowrap">
      <StatusIndicator
        showSpinner={showSpinner}
        status={status || { code: 2, message: "N/A", details: [] }}
      />
    </div>
  )
);

const RolloutId = observer(
  ({ rolloutId: rolloutId }: { rolloutId?: string }) => {
    if (!rolloutId) {
      return null;
    }
    return (
      <span className="font-mono text-gray-900 whitespace-nowrap">
        {rolloutId}
      </span>
    );
  }
);

const InvocationId = observer(({ invocationId }: { invocationId?: string }) => {
  if (!invocationId) {
    return null;
  }
  return (
    <span className="font-mono text-gray-900 whitespace-nowrap flex items-center gap-1">
      {invocationId}
      <AddFilterButton
        fieldPath="$.execution_metadata.invocation_id"
        value={invocationId}
        label="Invocation"
      />
    </span>
  );
});

const RowModel = observer(({ model }: { model: string | undefined }) => (
  <span className="text-gray-900 truncate block">{model || "N/A"}</span>
));

const RowScore = observer(({ score }: { score: number | undefined }) => {
  const scoreClass = score
    ? score >= 0.8
      ? "text-green-700"
      : score >= 0.6
      ? "text-yellow-700"
      : "text-red-700"
    : "text-gray-500";

  return (
    <span className={`font-mono whitespace-nowrap ${scoreClass}`}>
      {score?.toFixed(3) || "N/A"}
    </span>
  );
});

const RowCreated = observer(({ created_at }: { created_at: Date | string }) => {
  const date = created_at instanceof Date ? created_at : new Date(created_at);

  return (
    <span className="text-gray-600 whitespace-nowrap">
      {date.toLocaleDateString() + " " + date.toLocaleTimeString()}
    </span>
  );
});

// Granular metadata components following "dereference late" principle
const EvalMetadataSection = observer(
  ({ data }: { data: EvaluationRowType["eval_metadata"] }) => (
    <MetadataSection title="Eval Metadata" data={data} defaultExpanded={true} />
  )
);

const RolloutStatusSection = observer(
  ({ data }: { data: EvaluationRowType["rollout_status"] }) => (
    <MetadataSection
      title="Rollout Status"
      data={data}
      defaultExpanded={true}
    />
  )
);

const EvaluationResultSection = observer(
  ({ data }: { data: EvaluationRowType["evaluation_result"] }) => (
    <MetadataSection
      title="Evaluation Result"
      data={data}
      defaultExpanded={true}
    />
  )
);

const GroundTruthSection = observer(
  ({ data }: { data: EvaluationRowType["ground_truth"] }) => (
    <MetadataSection title="Ground Truth" data={data} />
  )
);

const UsageStatsSection = observer(
  ({ data }: { data: EvaluationRowType["usage"] }) => (
    <MetadataSection title="Usage Stats" data={data} />
  )
);

const InputMetadataSection = observer(
  ({ data }: { data: EvaluationRowType["input_metadata"] }) => (
    <MetadataSection title="Input Metadata" data={data} />
  )
);

const IdSection = observer(({ data }: { data: EvaluationRowType }) => (
  <MetadataSection
    title="IDs"
    data={{
      rollout_id: data.execution_metadata?.rollout_id,
      experiment_id: data.execution_metadata?.experiment_id,
      invocation_id: data.execution_metadata?.invocation_id,
      run_id: data.execution_metadata?.run_id,
    }}
  />
));

const ToolsSection = observer(
  ({ data }: { data: EvaluationRowType["tools"] }) => (
    <MetadataSection title="Tools" data={data} />
  )
);

const ChatInterfaceSection = observer(
  ({ messages }: { messages: EvaluationRowType["messages"] }) => (
    <ChatInterface messages={messages} />
  )
);

const ExpandedContent = observer(
  ({
    row,
    messages,
    eval_metadata,
    evaluation_result,
    ground_truth,
    usage,
    input_metadata,
    tools,
    rollout_status,
  }: {
    row: EvaluationRowType;
    messages: EvaluationRowType["messages"];
    eval_metadata: EvaluationRowType["eval_metadata"];
    evaluation_result: EvaluationRowType["evaluation_result"];
    ground_truth: EvaluationRowType["ground_truth"];
    usage: EvaluationRowType["usage"];
    input_metadata: EvaluationRowType["input_metadata"];
    tools: EvaluationRowType["tools"];
    rollout_status: EvaluationRowType["rollout_status"];
  }) => (
    <div className="p-4 bg-gray-50">
      <div className="flex gap-3 w-fit">
        {/* Left Column - Chat Interface */}
        <div className="min-w-0">
          <ChatInterfaceSection messages={messages} />
        </div>

        {/* Right Column - Metadata */}
        <div className="w-[500px] flex-shrink-0 space-y-3">
          <EvalMetadataSection data={eval_metadata} />
          <EvaluationResultSection data={evaluation_result} />
          <RolloutStatusSection data={rollout_status} />
          <IdSection data={row} />
          <GroundTruthSection data={ground_truth} />
          <UsageStatsSection data={usage} />
          <InputMetadataSection data={input_metadata} />
          <ToolsSection data={tools} />
        </div>
      </div>
    </div>
  )
);

export const EvaluationRow = observer(
  ({ row }: { row: EvaluationRowType; index: number }) => {
    const rolloutId = row.execution_metadata?.rollout_id;
    const isExpanded = state.isRowExpanded(rolloutId);

    const toggleExpanded = () => state.toggleRowExpansion(rolloutId);

    return (
      <>
        {/* Main Table Row */}
        <TableRowInteractive onClick={toggleExpanded}>
          {/* Expand/Collapse Icon */}
          <TableCell className="w-8 py-3">
            <ExpandIcon rolloutId={rolloutId} />
          </TableCell>

          {/* Name */}
          <TableCell className="py-3 text-xs">
            <RowName name={row.eval_metadata?.name} />
          </TableCell>

          {/* Eval Status */}
          <TableCell className="py-3 text-xs">
            <RowStatus
              status={row.eval_metadata?.status}
              showSpinner={row.eval_metadata?.status?.code === 101}
            />
          </TableCell>

          {/* Rollout Status */}
          <TableCell className="py-3 text-xs">
            <RowStatus
              status={row.rollout_status}
              showSpinner={row.rollout_status?.code === 101}
            />
          </TableCell>

          {/* Invocation ID */}
          <TableCell className="py-3 text-xs">
            <InvocationId
              invocationId={row.execution_metadata?.invocation_id}
            />
          </TableCell>

          {/* Rollout ID */}
          <TableCell className="py-3 text-xs">
            <RolloutId rolloutId={row.execution_metadata?.rollout_id} />
          </TableCell>

          {/* Model */}
          <TableCell className="py-3 text-xs">
            <RowModel model={row.input_metadata.completion_params.model} />
          </TableCell>

          {/* Score */}
          <TableCell className="py-3 text-xs">
            <RowScore score={row.evaluation_result?.score} />
          </TableCell>

          {/* Created */}
          <TableCell className="py-3 text-xs">
            <RowCreated created_at={row.created_at} />
          </TableCell>
        </TableRowInteractive>

        {/* Expanded Content Row */}
        {isExpanded && (
          <tr>
            <td colSpan={9} className="p-0">
              <ExpandedContent
                row={row}
                messages={row.messages}
                eval_metadata={row.eval_metadata}
                evaluation_result={row.evaluation_result}
                ground_truth={row.ground_truth}
                usage={row.usage}
                input_metadata={row.input_metadata}
                tools={row.tools}
                rollout_status={row.rollout_status}
              />
            </td>
          </tr>
        )}
      </>
    );
  }
);
