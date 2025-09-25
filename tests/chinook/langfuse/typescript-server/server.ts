import express, { Request, Response } from "express";
import cors from "cors";
import helmet from "helmet";
import { z } from "zod";
import { OpenAI } from "openai";
import { observeOpenAI } from "@langfuse/openai";
import "./instrumentation";
import "./env";

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
    parameters: z.object({}).passthrough().optional(),
  })
  .passthrough();

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
  .passthrough();

export const initRequestSchema = z.object({
  rollout_id: z.string(),
  model: z.string(),
  messages: z.array(messageSchema).min(1),
  tools: z.array(toolSchema).optional().nullable(),
  metadata: metadataSchema,
});

export const statusInfoSchema = z.object({
  reason: z.enum(["completed", "failed", "timeout", "cancelled"]),
  ended_at: z.string(),
  error: z.string().optional(),
});

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

// In-memory storage for rollout states
interface RolloutState {
  rollout_id: string;
  status: "running" | "completed" | "failed" | "timeout" | "cancelled";
  started_at: string;
  ended_at?: string;
  completed_turns: number;
  error?: string;
}

const rolloutStates = new Map<string, RolloutState>();

// Express app setup
const app: express.Application = express();
const PORT = process.env["PORT"] || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get("/health", (_req: Request, res: Response) => {
  res.json({ status: "healthy", timestamp: new Date().toISOString() });
});

// POST /init endpoint
app.post("/init", async (req: Request, res: Response) => {
  try {
    // Validate request body
    const validatedData = initRequestSchema.parse(req.body);
    const { rollout_id, model, messages, tools, metadata } = validatedData;

    console.log(`Initializing rollout ${rollout_id} with model ${model}`);

    // Create rollout state
    const rolloutState: RolloutState = {
      rollout_id,
      status: "running",
      started_at: new Date().toISOString(),
      completed_turns: 0,
    };

    rolloutStates.set(rollout_id, rolloutState);

    // Simulate async processing
    setTimeout(async () => {
      await simulateRolloutExecution(
        rollout_id,
        model,
        messages,
        tools || null,
        metadata
      );
    }, 100);

    res.status(200).json({
      status: "accepted",
      rollout_id,
      message: "Rollout initialized successfully",
    });
  } catch (error) {
    console.error("Error in /init endpoint:", error);

    if (error instanceof z.ZodError) {
      res.status(400).json({
        error: "Validation error",
        details: error.errors,
      });
    } else {
      res.status(500).json({
        error: "Internal server error",
        message: error instanceof Error ? error.message : "Unknown error",
      });
    }
  }
});

// GET /status endpoint
app.get("/status", (req: Request, res: Response) => {
  try {
    const { rollout_id } = req.query;

    if (!rollout_id || typeof rollout_id !== "string") {
      res.status(400).json({
        error: "Missing or invalid rollout_id parameter",
      });
      return;
    }

    const rolloutState = rolloutStates.get(rollout_id);

    if (!rolloutState) {
      res.status(404).json({
        error: "Rollout not found",
        rollout_id,
      });
      return;
    }

    const response: StatusResponse = {
      terminated: rolloutState.status !== "running",
    };

    if (rolloutState.status !== "running") {
      response.info = {
        reason: rolloutState.status as StatusInfo["reason"],
        ended_at: rolloutState.ended_at || new Date().toISOString(),
        ...(rolloutState.error && { error: rolloutState.error }),
      };
    }

    const validatedResponse = statusResponseSchema.parse(response);

    res.json(validatedResponse);
  } catch (error) {
    console.error("Error in /status endpoint:", error);
    res.status(500).json({
      error: "Internal server error",
      message: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Simulate rollout execution
async function simulateRolloutExecution(
  rollout_id: string,
  model: string,
  messages: Message[],
  tools: Tool[] | null,
  metadata: Metadata
): Promise<void> {
  const rolloutState = rolloutStates.get(rollout_id);
  if (!rolloutState) return;

  try {
    console.log(`Starting rollout execution for ${rollout_id}`);

    const openai = new OpenAI({
      apiKey: process.env["OPENAI_API_KEY"],
    });

    const tracedOpenAI = observeOpenAI(openai, {
      tags: [
        `invocation_id:${metadata.invocation_id}`,
        `experiment_id:${metadata.experiment_id}`,
        `rollout_id:${metadata.rollout_id}`,
        `run_id:${metadata.run_id}`,
        `row_id:${metadata.row_id}`,
      ],
    });

    const toolsToOpenAI = tools?.map((tool) => ({
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
          model,
          messages,
          tools: toolsToOpenAI,
        }
      : {
          model,
          messages,
        };

    await tracedOpenAI.chat.completions.create(completionParams);

    // Mark as completed
    rolloutState.status = "completed";
    rolloutState.ended_at = new Date().toISOString();
    rolloutState.completed_turns = 1;

    console.log(`Rollout ${rollout_id} completed successfully`);
  } catch (error) {
    console.error(`Error in rollout execution for ${rollout_id}:`, error);

    rolloutState.status = "failed";
    rolloutState.ended_at = new Date().toISOString();
    rolloutState.error =
      error instanceof Error ? error.message : "Unknown error";
  }
}

// Error handling middleware
app.use((error: Error, _req: Request, res: Response, _next: any) => {
  console.error("Unhandled error:", error);
  res.status(500).json({
    error: "Internal server error",
    message: error.message,
  });
});

// 404 handler
app.use((_req: Request, res: Response) => {
  res.status(404).json({
    error: "Not found",
    path: _req.originalUrl,
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 TypeScript Express server running on port ${PORT}`);
  console.log(`📋 Available endpoints:`);
  console.log(`   POST /init - Initialize a rollout`);
  console.log(`   GET /status?rollout_id={id} - Check rollout status`);
  console.log(`   GET http://localhost:${PORT}/health - Health check`);
});

export default app;
