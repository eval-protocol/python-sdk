import express, { Request, Response } from "express";
import cors from "cors";
import helmet from "helmet";
import { z } from "zod";
import { OpenAI } from "openai";
import { observeOpenAI } from "@langfuse/openai";
import "./instrumentation.js";
import "./env.js";
import {
  initRequestSchema,
  statusResponseSchema,
  StatusResponse,
  initRequestToCompletionParams,
  InitRequest,
  createLangfuseConfigTags,
} from "eval-protocol";

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
    const { completion_params, metadata } = validatedData;
    const rollout_id = metadata.rollout_id;
    const model = validatedData.completion_params?.['model'];
    if (!model) {
      throw new Error("model is required in completion_params");
    }
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
      await simulateRolloutExecution(validatedData);
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
        reason: rolloutState.status,
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
  initRequest: InitRequest
): Promise<void> {
  const rollout_id = initRequest.metadata.rollout_id;
  const rolloutState = rolloutStates.get(rollout_id);
  if (!rolloutState) return;

  try {
    console.log(`Starting rollout execution for ${rollout_id}`);

    const openai = new OpenAI({
      baseURL: initRequest.model_base_url || "https://api.fireworks.ai/inference/v1",
      apiKey: process.env["FIREWORKS_API_KEY"] || process.env["OPENAI_API_KEY"],
    });

    const tracedOpenAI = observeOpenAI(openai, {
      tags: createLangfuseConfigTags(initRequest),
    });

    const completionParams = initRequestToCompletionParams(initRequest);

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
