import z from "zod";

// Shared protocol schemas/types for Eval Protocol TypeScript support.

export const roleSchema = z.enum(["system", "user", "assistant"]);

export const messageSchema = z.union([
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

export const functionDefinitionSchema = z
  .object({
    name: z.string().regex(/^[a-zA-Z0-9_-]{1,64}$/),
    description: z.string().optional(),
    // JSON Schema object; allow arbitrary keys
    parameters: z.object({}).catchall(z.any()).optional(),
  })
  .catchall(z.any());

export const toolSchema = z.object({
  type: z.literal("function"),
  function: functionDefinitionSchema,
});

export const metadataSchema = z
  .object({
    invocation_id: z.string(),
    experiment_id: z.string(),
    rollout_id: z.string(),
    run_id: z.string(),
    row_id: z.string(),
  })
  .catchall(z.any());

export const initRequestSchema = z.object({
  completion_params: z
    .record(z.string(), z.any())
    .describe("Completion parameters including model and optional model_kwargs, temperature, etc."),
  messages: z.array(messageSchema).optional(),
  tools: z.array(toolSchema).optional().nullable(),
  metadata: metadataSchema,
  model_base_url: z.string().optional().nullable(),
  api_key: z.string().optional().nullable(),
});

export const statusInfoSchema = z.record(z.string(), z.any());

export const statusResponseSchema = z.object({
  terminated: z.boolean(),
  info: statusInfoSchema.optional(),
});

export type Message = z.infer<typeof messageSchema>;
export type FunctionDefinition = z.infer<typeof functionDefinitionSchema>;
export type Tool = z.infer<typeof toolSchema>;
export type Metadata = z.infer<typeof metadataSchema>;
export type InitRequest = z.infer<typeof initRequestSchema>;
export type StatusInfo = z.infer<typeof statusInfoSchema>;
export type StatusResponse = z.infer<typeof statusResponseSchema>;
