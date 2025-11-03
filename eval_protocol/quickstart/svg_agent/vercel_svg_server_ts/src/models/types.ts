/**
 * Minimal types for the TypeScript server
 */

import { z } from 'zod';

// Message schema with nullable optional fields
export const messageSchema = z.union([
  z.object({
    role: z.enum(['system', 'user', 'assistant']),
    content: z.union([z.string(), z.array(z.any())]).optional().nullable(),
    name: z.string().optional().nullable(),
    tool_call_id: z.string().optional().nullable(),
    tool_calls: z.array(z.any()).optional().nullable(),
    function_call: z.any().optional().nullable(),
    reasoning_content: z.string().optional().nullable(),
    control_plane_step: z.record(z.any()).optional().nullable(),
    weight: z.number().optional().nullable()
  }),
  z.object({
    role: z.literal('tool'),
    content: z.string(),
    tool_call_id: z.string(),
    name: z.string().optional().nullable()
  }),
]);

// Rollout metadata schema
export const rolloutMetadataSchema = z.object({
  invocation_id: z.string(),
  experiment_id: z.string(),
  rollout_id: z.string(),
  run_id: z.string(),
  row_id: z.string(),
});

// InitRequest schema
export const initRequestSchema = z.object({
  completion_params: z.record(z.string(), z.any()).describe('Completion parameters including model and optional model_kwargs, temperature, etc.'),
  messages: z.array(messageSchema).optional().nullable(),
  tools: z.array(z.any()).optional().nullable(),
  metadata: rolloutMetadataSchema,
  model_base_url: z.string().optional().nullable(),
  api_key: z.string().optional().nullable(),
});

// Infer TypeScript types from schemas
export type Message = z.infer<typeof messageSchema>;
export type RolloutMetadata = z.infer<typeof rolloutMetadataSchema>;
export type InitRequest = z.infer<typeof initRequestSchema>;
