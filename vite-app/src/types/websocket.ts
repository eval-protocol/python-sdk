// WebSocket message types based on logs_server.py
import { z } from 'zod';

// Zod Schemas for runtime validation

// File update message schema
export const FileUpdateMessageSchema = z.object({
  type: z.enum(['file_created', 'file_changed', 'file_deleted']),
  path: z.string(),
  timestamp: z.number(),
  contents: z.string().nullable().optional(),
});

// Initialize logs message schema
export const InitializeLogsMessageSchema = z.object({
  type: z.literal('initialize_logs'),
  logs: z.array(z.string()),
});

// Union schema for all WebSocket server messages
export const WebSocketServerMessageSchema = z.discriminatedUnion('type', [
  FileUpdateMessageSchema,
  InitializeLogsMessageSchema,
] as const);

// Server status response schema
export const ServerStatusResponseSchema = z.object({
  status: z.literal('ok'),
  build_dir: z.string(),
  active_connections: z.number(),
  watch_paths: z.array(z.string()),
});

// File system event schema
export const FileSystemEventSchema = z.object({
  type: z.enum(['file_created', 'file_changed', 'file_deleted']),
  path: z.string(),
  timestamp: z.number(),
  contents: z.string().nullable().optional(),
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
export type FileUpdateMessage = z.infer<typeof FileUpdateMessageSchema>;
export type InitializeLogsMessage = z.infer<typeof InitializeLogsMessageSchema>;
export type WebSocketServerMessage = z.infer<typeof WebSocketServerMessageSchema>;
export type ServerStatusResponse = z.infer<typeof ServerStatusResponseSchema>;
export type FileSystemEvent = z.infer<typeof FileSystemEventSchema>;
export type LogEntry = z.infer<typeof LogEntrySchema>;
