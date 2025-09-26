import z from "zod";
import type { ChatCompletionCreateParamsNonStreaming } from "openai/resources/chat/completions/completions";

// Zod schemas for validation
const roleSchema = z.enum(["system", "user", "assistant"]);
const messageSchema = z.union([
  z.object({
    role: roleSchema,
    content: z.string(),
  }),
  z.object({
    role: z.literal("tool"),
    content: z.string(),
    tool_call_id: z.string(),
  }),
]);

const functionDefinitionSchema = z
  .object({
    name: z.string().regex(/^[a-zA-Z0-9_-]{1,64}$/),
    description: z.string().optional(),
    // JSON Schema object; allow arbitrary keys
    parameters: z.object({}).loose().optional(),
  })
  .loose();

const toolSchema = z.object({
  type: z.literal("function"),
  function: functionDefinitionSchema,
});

const metadataSchema = z
  .object({
    invocation_id: z.string(),
    experiment_id: z.string(),
    rollout_id: z.string(),
    run_id: z.string(),
    row_id: z.string(),
  })
  .loose();

export const initRequestSchema = z.object({
  rollout_id: z.string(),
  model: z.string(),
  messages: z.array(messageSchema).min(1),
  tools: z.array(toolSchema).optional().nullable(),
  metadata: metadataSchema,
  model_base_url: z.string().optional().nullable(),
});

export const statusInfoSchema = z.record(z.string(), z.any());

export const statusResponseSchema = z.object({
  terminated: z.boolean(),
  info: statusInfoSchema.optional(),
});

// Infer types from schemas
export type Message = z.infer<typeof messageSchema>;
export type FunctionDefinition = z.infer<typeof functionDefinitionSchema>;
export type Tool = z.infer<typeof toolSchema>;
export type Metadata = z.infer<typeof metadataSchema>;
export type InitRequest = z.infer<typeof initRequestSchema>;
export type StatusInfo = z.infer<typeof statusInfoSchema>;
export type StatusResponse = z.infer<typeof statusResponseSchema>;

export function initRequestToCompletionParams(
  initRequest: InitRequest
): ChatCompletionCreateParamsNonStreaming {
  const toolsToOpenAI = initRequest.tools?.map((tool) => ({
    type: "function" as const,
    function: tool.function.description
      ? {
          name: tool.function.name,
          description: tool.function.description,
          parameters: tool.function.parameters || {},
        }
      : {
          name: tool.function.name,
          parameters: tool.function.parameters || {},
        },
  }));

  const completionParams = toolsToOpenAI
    ? {
        model: initRequest.model,
        messages: initRequest.messages,
        tools: toolsToOpenAI,
      }
    : {
        model: initRequest.model,
        messages: initRequest.messages,
      };
  return completionParams;
}

export function createLangfuseConfigTags(initRequest: InitRequest): string[] {
  return [
    `invocation_id:${initRequest.metadata.invocation_id}`,
    `experiment_id:${initRequest.metadata.experiment_id}`,
    `rollout_id:${initRequest.metadata.rollout_id}`,
    `run_id:${initRequest.metadata.run_id}`,
    `row_id:${initRequest.metadata.row_id}`,
  ];
}
