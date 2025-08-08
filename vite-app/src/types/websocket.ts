// WebSocket message types based on logs_server.py
import { z } from 'zod';
import { EvaluationRowSchema } from './eval-protocol';

// Zod Schemas for runtime validation

// Initialize logs message schema
export const InitializeLogsMessageSchema = z.object({
  type: z.literal('initialize_logs'),
  logs: z.array(z.string()),
});

export const LogMessageSchema = z.object({
  type: z.literal('log'),
  row: EvaluationRowSchema,
});

// Union schema for all WebSocket server messages
export const WebSocketServerMessageSchema = z.discriminatedUnion('type', [
  InitializeLogsMessageSchema,
  LogMessageSchema,
] as const);

// Server status response schema
export const ServerStatusResponseSchema = z.object({
  status: z.literal('ok'),
  build_dir: z.string(),
  active_connections: z.number(),
  watch_paths: z.array(z.string()),
});

// Log entry schema
export const LogEntrySchema = z.object({
  id: z.string(),
  timestamp: z.string(),
  level: z.enum(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
  message: z.string(),
  metadata: z.record(z.string(), z.any()).optional(),
});

// Type inference from Zod schemas
export type InitializeLogsMessage = z.infer<typeof InitializeLogsMessageSchema>;
export type LogMessage = z.infer<typeof LogMessageSchema>;
export type WebSocketServerMessage = z.infer<typeof WebSocketServerMessageSchema>;
export type ServerStatusResponse = z.infer<typeof ServerStatusResponseSchema>;
export type LogEntry = z.infer<typeof LogEntrySchema>;
