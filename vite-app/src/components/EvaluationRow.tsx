import { observer } from "mobx-react";
import type { EvaluationRow as EvaluationRowType } from "../types/eval-protocol";
import { ChatInterface } from "./ChatInterface";
import { MetadataSection } from "./MetadataSection";
import StatusIndicator from "./StatusIndicator";
import { state } from "../App";
import { TableCell, TableRowInteractive } from "./TableContainer";

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
    status: string | undefined;
    showSpinner: boolean;
  }) => (
    <div className="whitespace-nowrap">
      <StatusIndicator showSpinner={showSpinner} status={status || "N/A"} />
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
      rollout_id: data.rollout_id,
      cohort_id: data.cohort_id,
      invocation_id: data.invocation_id,
      run_id: data.run_id,
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
  }: {
    row: EvaluationRowType;
    messages: EvaluationRowType["messages"];
    eval_metadata: EvaluationRowType["eval_metadata"];
    evaluation_result: EvaluationRowType["evaluation_result"];
    ground_truth: EvaluationRowType["ground_truth"];
    usage: EvaluationRowType["usage"];
    input_metadata: EvaluationRowType["input_metadata"];
    tools: EvaluationRowType["tools"];
  }) => (
    <div className="p-4 bg-gray-50 border-t border-gray-200">
      <div className="flex gap-3 w-fit">
        {/* Left Column - Chat Interface */}
        <div className="min-w-0">
          <ChatInterfaceSection messages={messages} />
        </div>

        {/* Right Column - Metadata */}
        <div className="w-[500px] flex-shrink-0 space-y-3">
          <EvalMetadataSection data={eval_metadata} />
          <EvaluationResultSection data={evaluation_result} />
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
    const rolloutId = row.rollout_id;
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

          {/* Status */}
          <TableCell className="py-3 text-xs">
            <RowStatus
              status={row.eval_metadata?.status}
              showSpinner={row.eval_metadata?.status === "running"}
            />
          </TableCell>

          {/* Rollout ID */}
          <TableCell className="py-3 text-xs">
            <RolloutId rolloutId={row.rollout_id} />
          </TableCell>

          {/* Model */}
          <TableCell className="py-3 text-xs">
            <RowModel model={row.input_metadata.completion_params?.model} />
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
            <td colSpan={8} className="p-0">
              <ExpandedContent
                row={row}
                messages={row.messages}
                eval_metadata={row.eval_metadata}
                evaluation_result={row.evaluation_result}
                ground_truth={row.ground_truth}
                usage={row.usage}
                input_metadata={row.input_metadata}
                tools={row.tools}
              />
            </td>
          </tr>
        )}
      </>
    );
  }
);
