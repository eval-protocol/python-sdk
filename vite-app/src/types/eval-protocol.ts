import { z } from 'zod';


// Base schemas
export const ChatCompletionContentPartTextParamSchema = z.object({
  text: z.string().describe('The text content.'),
  type: z.literal('text').default('text').describe('The type of the content part.')
});


export const FunctionCallSchema = z.object({
  name: z.string(),
  arguments: z.string()
});

export const ChatCompletionMessageToolCallSchema = z.object({
  id: z.string(),
  type: z.literal('function'),
  function: FunctionCallSchema
});

export const MessageSchema = z.object({
  role: z.string().describe('assistant, user, system, tool'),
  content: z.union([z.string(), z.array(ChatCompletionContentPartTextParamSchema)]).optional().default('').describe('The content of the message.'),
  reasoning_content: z.string().optional().describe('Optional hidden chain-of-thought or reasoning content.'),
  name: z.string().optional(),
  tool_call_id: z.string().optional(),
  tool_calls: z.array(ChatCompletionMessageToolCallSchema).optional(),
  function_call: FunctionCallSchema.optional(),
  control_plane_step: z.record(z.string(), z.any()).optional()
});

export const MetricResultSchema = z.object({
  is_score_valid: z.boolean().default(true),
  score: z.number().min(0.0).max(1.0),
  reason: z.string()
});

export const StepOutputSchema = z.object({
  step_index: z.union([z.number(), z.string()]).describe('User-defined index for the step (e.g., assistant message index, turn number). This is used by the system to map this output to the internal StepData.'),
  base_reward: z.number().describe('Base reward calculated by the user\'s reward function for this step.'),
  terminated: z.boolean().default(false).describe('Whether the environment signaled termination at this step.'),
  control_plane_info: z.record(z.string(), z.any()).optional().describe('Structured info from the environment\'s control plane.'),
  metrics: z.record(z.string(), z.any()).default({}).describe('Optional dictionary of custom metrics for this step.'),
  reason: z.string().optional().describe('Optional explanation for the step\'s base reward or metrics.')
});

export const EvaluateResultSchema = z.object({
  score: z.number().describe('The overall evaluation score, typically between 0.0 and 1.0.'),
  is_score_valid: z.boolean().default(true).describe('Whether the overall score is valid.'),
  reason: z.string().optional().describe('Optional explanation for the overall score.'),
  metrics: z.record(z.string(), MetricResultSchema).default({}).describe('Dictionary of component metrics for detailed breakdown.'),
  step_outputs: z.array(StepOutputSchema).optional().describe('For RL, a list of outputs for each conceptual step, providing base rewards.'),
  error: z.string().optional().describe('Optional error message if the evaluation itself encountered an issue.'),
  trajectory_info: z.record(z.string(), z.any()).optional().describe('Additional trajectory-level information (duration, steps, termination_reason, etc.).'),
  final_control_plane_info: z.record(z.string(), z.any()).optional().describe('The final control plane state that led to termination.'),
  agg_score: z.number().optional().describe('The aggregated score of the evaluation across all runs.'),
  standard_error: z.number().optional().describe('The standard error of the evaluation across all runs.')});

export const CompletionParamsSchema = z.record(z.string(), z.any());

// AIP-193 ErrorInfo model for structured error details
export const ErrorInfoSchema = z.object({
  reason: z.string().describe('Short snake_case description of the error cause'),
  domain: z.string().describe('Logical grouping for the error reason'),
  metadata: z.record(z.string(), z.any()).default({}).describe('Additional dynamic information as context')
});

// AIP-193 compatible Status model (matches Python Status)
export const StatusCodeSchema = z.enum([
  'OK', 'CANCELLED', 'UNKNOWN', 'INVALID_ARGUMENT', 'DEADLINE_EXCEEDED', 'NOT_FOUND',
  'ALREADY_EXISTS', 'PERMISSION_DENIED', 'RESOURCE_EXHAUSTED', 'FAILED_PRECONDITION',
  'ABORTED', 'OUT_OF_RANGE', 'UNIMPLEMENTED', 'INTERNAL', 'UNAVAILABLE', 'DATA_LOSS',
  'UNAUTHENTICATED', 'FINISHED', 'RUNNING'
]).describe('Common gRPC status codes as defined in google.rpc.Code');

// Mapping from integer status codes to their corresponding code names
export const STATUS_CODE_MAP: Record<number, StatusCode> = {
  0: 'OK',
  1: 'CANCELLED',
  2: 'UNKNOWN',
  3: 'INVALID_ARGUMENT',
  4: 'DEADLINE_EXCEEDED',
  5: 'NOT_FOUND',
  6: 'ALREADY_EXISTS',
  7: 'PERMISSION_DENIED',
  8: 'RESOURCE_EXHAUSTED',
  9: 'FAILED_PRECONDITION',
  10: 'ABORTED',
  11: 'OUT_OF_RANGE',
  12: 'UNIMPLEMENTED',
  13: 'INTERNAL',
  14: 'UNAVAILABLE',
  15: 'DATA_LOSS',
  16: 'UNAUTHENTICATED',
  100: 'FINISHED',
  101: 'RUNNING'
} as const;

// Helper function to get status code name from integer
export const getStatusCodeName = (code: number): StatusCode => {
  return STATUS_CODE_MAP[code] || 'UNKNOWN';
};

export const StatusSchema = z.object({
  code: z.number().describe('The status code (numeric value from google.rpc.Code enum)'),
  message: z.string().describe('Developer-facing, human-readable debug message in English'),
  details: z.array(z.record(z.string(), z.any())).default([]).describe('Additional error information, each packed in a google.protobuf.Any message format')
});

// Evaluation threshold configuration
export const EvaluationThresholdSchema = z.object({
  success: z.number().min(0.0).max(1.0).describe('Minimum success rate threshold (fraction of total score, 0.0 to 1.0)'),
  standard_error: z.number().min(0.0).max(1.0).optional().describe('Maximum standard error threshold (fraction of total score, 0.0 to 1.0)')
});

export const InputMetadataSchema = z.object({
  row_id: z.string().optional().describe('Unique string to ID the row. If not provided, a stable hash will be generated based on the row\'s content. The hash removes fields that are not typically stable across processes such as created_at, execution_metadata, and pid.'),
  completion_params: CompletionParamsSchema.describe('Completion endpoint parameters used'),
  dataset_info: z.record(z.string(), z.any()).optional().describe('Dataset row details: seed, system_prompt, environment_context, etc'),
  session_data: z.record(z.string(), z.any()).optional().describe('Session metadata like timestamp (input only, no duration/usage)')
}).loose(); // equivalent to extra="allow" in Pydantic

export const CompletionUsageSchema = z.object({
  prompt_tokens: z.number(),
  completion_tokens: z.number(),
  total_tokens: z.number()
});

export const EvalMetadataSchema = z.object({
  name: z.string().describe('Name of the evaluation'),
  description: z.string().optional().describe('Description of the evaluation'),
  version: z.string().describe('Version of the evaluation. Should be populated with a PEP 440 version string.'),
  status: StatusSchema.optional().describe('Status of the evaluation'),
  num_runs: z.number().int().describe('Number of times the evaluation was repeated'),
  aggregation_method: z.string().describe('Method used to aggregate scores across runs'),
  passed_threshold: EvaluationThresholdSchema.optional().describe('Threshold configuration for test success'),
  passed: z.boolean().optional().describe('Whether the evaluation passed based on the threshold')
});

export const ExecutionMetadataSchema = z.object({
  invocation_id: z.string().optional().describe('The ID of the invocation that this row belongs to.'),
  experiment_id: z.string().optional().describe('The ID of the experiment that this row belongs to.'),
  rollout_id: z.string().optional().describe('The ID of the rollout that this row belongs to.'),
  run_id: z.string().optional().describe('The ID of the run that this row belongs to.'),
});

export const EvaluationRowSchema = z.object({
  messages: z.array(MessageSchema).describe('List of messages in the conversation/trajectory.'),
  tools: z.array(z.record(z.string(), z.any())).optional().describe('Available tools/functions that were provided to the agent.'),
  input_metadata: InputMetadataSchema.describe('Metadata related to the input (dataset info, model config, session data, etc.).'),
  rollout_status: StatusSchema.describe('The status of the rollout following AIP-193 standards.'),
  execution_metadata: ExecutionMetadataSchema.optional().describe('Metadata about the execution of the evaluation.'),
  ground_truth: z.string().optional().describe('Optional ground truth reference for this evaluation.'),
  evaluation_result: EvaluateResultSchema.optional().describe('The evaluation result for this row/trajectory.'),
  usage: CompletionUsageSchema.optional().describe('Token usage statistics from LLM calls during execution.'),
  created_at: z.preprocess(
    (val) => typeof val === "string" ? new Date(val) : val,
    z.date()
  ).describe('The timestamp when the row was created. Accepts string and parses to Date.'),
  eval_metadata: EvalMetadataSchema.optional().describe('Metadata about the evaluation that was run.'),
  pid: z.number().optional().describe('The PID of the process that created the row. This is used by the evaluation watcher to detect stopped evaluations.')
});

// Agent Evaluation Framework (V2) schemas
export const ResourceServerConfigSchema = z.object({
  start_command: z.string().describe('The command to start the server. The string \'{port}\' will be replaced with a dynamically allocated free port.'),
  health_check_url: z.string().describe('The URL to poll to check if the server is ready. The string \'{port}\' will be replaced with the allocated port.')
});

export const EvaluationCriteriaModelSchema = z.object({
  final_state_query: z.string().optional().describe('A query (e.g., SQL) to run on the final state of the resource.'),
  expected_query_result_transform: z.string().optional().describe('A Python lambda string (e.g., \'lambda x: x > 0\') to transform and evaluate the query result to a boolean.'),
  ground_truth_function_calls: z.array(z.array(z.string())).optional().describe('Ground truth function calls for BFCL evaluation.'),
  ground_truth_comparable_state: z.record(z.string(), z.any()).optional().describe('Ground truth comparable state for BFCL evaluation.')
});

export const TaskDefinitionModelSchema = z.object({
  name: z.string().describe('Unique name for the task.'),
  description: z.string().optional().describe('A brief description of the task.'),
  resource_type: z.string().describe('The type of ForkableResource to use (e.g., \'SQLResource\', \'PythonStateResource\', \'FileSystemResource\', \'DockerResource\').'),
  base_resource_config: z.record(z.string(), z.any()).default({}).describe('Configuration dictionary passed to the base resource\'s setup() method.'),
  tools_module_path: z.string().optional().describe('Optional Python import path to a module containing custom tool functions for this task.'),
  reward_function_path: z.string().describe('Python import path to the reward function (e.g., \'my_module.my_reward_func\').'),
  goal_description: z.string().optional().describe('A human-readable description of the agent\'s goal for this task.'),
  evaluation_criteria: EvaluationCriteriaModelSchema.optional().describe('Criteria used by the Orchestrator to determine if the primary goal was achieved.'),
  initial_user_prompt: z.string().optional().describe('The initial prompt or message to start the agent interaction. Deprecated if \'messages\' field is used for multi-turn.'),
  messages: z.array(z.record(z.string(), z.any())).optional().describe('A list of messages to start the conversation, can represent multiple user turns for sequential processing.'),
  poc_max_turns: z.number().int().min(1).default(3).describe('For PoC Orchestrator, the maximum number of interaction turns.'),
  resource_server: ResourceServerConfigSchema.optional().describe('Configuration for a background server required for the task.'),
  num_rollouts: z.number().int().min(1).default(1).describe('Number of parallel rollouts to execute for this task definition.'),
  dataset_path: z.string().optional().describe('Path to dataset file (JSONL) containing experimental conditions for data-driven evaluation.'),
  num_rollouts_per_sample: z.number().int().min(1).default(1).describe('Number of rollouts to execute per sample from the dataset.')
}).loose(); // equivalent to extra="allow" in Pydantic

// MCP Configuration schemas
export const MCPConfigurationServerStdioSchema = z.object({
  command: z.string().describe('command to run the MCP server'),
  args: z.array(z.string()).default([]).describe('to pass to the command'),
  env: z.array(z.string()).default([]).describe('List of environment variables to verify exist in the environment')
});

export const MCPConfigurationServerUrlSchema = z.object({
  url: z.string().describe('url to the MCP server')
});

export const MCPMultiClientConfigurationSchema = z.object({
  mcpServers: z.record(z.string(), z.union([MCPConfigurationServerStdioSchema, MCPConfigurationServerUrlSchema]))
});

// Export TypeScript types derived from the schemas
export type ChatCompletionContentPartTextParam = z.infer<typeof ChatCompletionContentPartTextParamSchema>;
export type Message = z.infer<typeof MessageSchema>;
export type MetricResult = z.infer<typeof MetricResultSchema>;
export type StepOutput = z.infer<typeof StepOutputSchema>;
export type EvaluateResult = z.infer<typeof EvaluateResultSchema>;
export type CompletionParams = z.infer<typeof CompletionParamsSchema>;
export type InputMetadata = z.infer<typeof InputMetadataSchema>;
export type CompletionUsage = z.infer<typeof CompletionUsageSchema>;
export type EvalMetadata = z.infer<typeof EvalMetadataSchema>;
export type EvaluationRow = z.infer<typeof EvaluationRowSchema>;
export type Status = z.infer<typeof StatusSchema>;
export type StatusCode = z.infer<typeof StatusCodeSchema>;
export type ErrorInfo = z.infer<typeof ErrorInfoSchema>;
export type EvaluationThreshold = z.infer<typeof EvaluationThresholdSchema>;
export type ResourceServerConfig = z.infer<typeof ResourceServerConfigSchema>;
export type EvaluationCriteriaModel = z.infer<typeof EvaluationCriteriaModelSchema>;
export type TaskDefinitionModel = z.infer<typeof TaskDefinitionModelSchema>;
export type MCPConfigurationServerStdio = z.infer<typeof MCPConfigurationServerStdioSchema>;
export type MCPConfigurationServerUrl = z.infer<typeof MCPConfigurationServerUrlSchema>;
export type MCPMultiClientConfiguration = z.infer<typeof MCPMultiClientConfigurationSchema>;
