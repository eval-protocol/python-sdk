/**
 * Vercel serverless function for SVGBench remote evaluation
 *
 * TypeScript port of the Python Flask server with:
 * - Fire-and-forget async processing
 * - Comprehensive logging with Fireworks tracing
 * - Robust error handling with Status codes
 * - Full CORS support
 * - Vercel production optimization
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import OpenAI from 'openai';
import { initRequestSchema, InitRequest } from '../src/models/types.js';
import { Status } from '../src/models/status.js';
import { mapOpenAIErrorToStatus } from '../src/models/exceptions.js';
import { resolveApiKey } from '../src/config/environment.js';

// Simple Fireworks logging function
async function logToFireworks(rolloutId: string, message: string, status: Status, apiKey: string): Promise<void> {
  try {
    const payload = {
      program: "eval_protocol",
      message: message,
      tags: [`rollout_id:${rolloutId}`],
      extras: {
        logger_name: `__main__.${rolloutId}`,
        level: "INFO",
        timestamp: new Date().toISOString()
      },
      status: {
        code: status.code,
        message: status.message,
        details: status.details
      }
    };

    const url = process.env.FW_TRACING_GATEWAY_BASE_URL || 'https://tracing.fireworks.ai';

    const response = await fetch(`${url}/logs`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
        'User-Agent': 'typescript-fetch/1.0.0'
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      console.error(`[FIREWORKS] Failed to log rollout ${rolloutId}: ${response.status}`);
    }
  } catch (error: any) {
    console.error(`[FIREWORKS] Error logging rollout ${rolloutId}:`, error.message);
  }
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Set CORS headers for all responses
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return res.status(200).json({});
  }

  // Handle health check
  if (req.method === 'GET') {
    return res.status(200).json({
      status: 'ok',
      message: 'SVGBench Vercel TypeScript Serverless Function',
      endpoints: {
        'POST /init': 'Process SVGBench evaluation requests',
        'GET /': 'Health check endpoint'
      }
    });
  }

  // Only handle POST requests for the main functionality
  if (req.method !== 'POST') {
    return res.status(405).json({
      error: `Method ${req.method} not allowed`
    });
  }

  let rolloutId: string | undefined;
  let apiKey: string | null = null;

  try {
    // Parse and validate request body
    const parseResult = initRequestSchema.safeParse(req.body);

    if (!parseResult.success) {
      const errorMsg = `Invalid request format: ${parseResult.error.message}`;
      console.error(`INFO:rollout:unknown:${errorMsg}`);

      return res.status(400).json({
        error: errorMsg,
        details: parseResult.error.issues
      });
    }

    const initRequest: InitRequest = parseResult.data;
    rolloutId = initRequest.metadata.rollout_id;

    console.log(`INFO:rollout:${rolloutId}:Received rollout request`);

    // Validate required fields
    if (!initRequest.messages || initRequest.messages.length === 0) {
      const errorMsg = 'messages is required and cannot be empty';
      console.error(`INFO:rollout:${rolloutId}:${errorMsg}`);

      return res.status(400).json({
        error: errorMsg,
        rollout_id: rolloutId
      });
    }

    // Resolve API key with fallback chain
    apiKey = resolveApiKey(initRequest.api_key);
    if (!apiKey) {
      const errorMsg = 'API key not provided in request or FIREWORKS_API_KEY environment variable';
      console.error(`INFO:rollout:${rolloutId}:${errorMsg}`);

      return res.status(401).json({
        error: errorMsg,
        rollout_id: rolloutId
      });
    }

    const startTime = Date.now();

    // Create OpenAI client
    const openaiClient = new OpenAI({
      apiKey,
      baseURL: initRequest.model_base_url!,  // Always provided in your use case
    });

    const model = initRequest.completion_params?.model;
    console.log(`INFO:rollout:${rolloutId}:Sending completion request to model ${model}`);

    // Prepare completion arguments - sanitize messages
    const allowedMessageFields = new Set([
      'role', 'content', 'name', 'tool_call_id', 'tool_calls', 'function_call'
    ]);

    const sanitizedMessages = (initRequest.messages || []).map(message => {
      const sanitized: any = {};
      for (const [key, value] of Object.entries(message)) {
        if (allowedMessageFields.has(key) && value !== undefined && value !== null) {
          sanitized[key] = value;
        }
      }
      if (!sanitized.role) {
        throw new Error('Message role is required');
      }
      return sanitized as OpenAI.Chat.ChatCompletionMessageParam;
    });

    // Build completion parameters
    const completionParams: OpenAI.Chat.ChatCompletionCreateParams = {
      model: String(model),
      messages: sanitizedMessages,
      stream: false
    };

    // Add optional parameters
    if (initRequest.completion_params?.temperature !== undefined) {
      completionParams.temperature = Number(initRequest.completion_params.temperature);
    }
    if (initRequest.completion_params?.max_tokens !== undefined) {
      completionParams.max_tokens = Number(initRequest.completion_params.max_tokens);
    }
    if (initRequest.completion_params?.top_p !== undefined) {
      completionParams.top_p = Number(initRequest.completion_params.top_p);
    }
    if (initRequest.completion_params?.frequency_penalty !== undefined) {
      completionParams.frequency_penalty = Number(initRequest.completion_params.frequency_penalty);
    }
    if (initRequest.completion_params?.presence_penalty !== undefined) {
      completionParams.presence_penalty = Number(initRequest.completion_params.presence_penalty);
    }
    if (initRequest.completion_params?.stop !== undefined) {
      completionParams.stop = initRequest.completion_params.stop;
    }

    // Add tools if present
    if (initRequest.tools && initRequest.tools.length > 0) {
      completionParams.tools = initRequest.tools.map(tool => ({
        type: 'function' as const,
        function: {
          name: tool.function.name,
          ...(tool.function.description && { description: tool.function.description }),
          ...(tool.function.parameters && { parameters: tool.function.parameters })
        }
      }));
    }

    // Debug log showing what we're sending (similar to Python version)
    const maskedApiKey = apiKey ? `${apiKey.substring(0, 8)}...${apiKey.substring(apiKey.length - 4)}` : 'undefined';

    console.log(`INFO:rollout:${rolloutId}:DEBUG: ${initRequest.model_base_url}, COMPLETION_KWARGS: ${JSON.stringify(completionParams)}, API_KEY: ${maskedApiKey}, MODEL: ${model}, BASE_URL: ${initRequest.model_base_url}`);

    // Perform chat completion synchronously
    const completion = await openaiClient.chat.completions.create(completionParams);

    const duration = Date.now() - startTime;
    console.log(`INFO:rollout:${rolloutId}:Rollout ${rolloutId} completed successfully`);

    // Log to Fireworks tracing system synchronously
    const status = Status.rolloutFinished();
    await logToFireworks(rolloutId!, `Rollout ${rolloutId} completed`, status, apiKey);

    // Return successful response with completion
    return res.status(200).json({
      status: 'completed',
      rollout_id: rolloutId,
      message: 'Rollout completed successfully'
    });

  } catch (error: any) {
    // Handle all errors in one place
    const status = mapOpenAIErrorToStatus(error);
    console.error(`INFO:rollout:${rolloutId || 'unknown'}:Rollout ${rolloutId} failed: ${error.message}`);

    // Log error to Fireworks tracing system synchronously (only if we have rolloutId and apiKey)
    if (rolloutId && apiKey) {
      await logToFireworks(rolloutId, `Rollout ${rolloutId} failed: ${error.message}`, status, apiKey);
    }

    // Return appropriate HTTP status based on error type
    if (error instanceof OpenAI.AuthenticationError || error instanceof OpenAI.PermissionDeniedError) {
      return res.status(403).json({
        error: `Authentication failed: ${error.message}`,
        rollout_id: rolloutId
      });
    } else if (error instanceof OpenAI.NotFoundError) {
      return res.status(404).json({
        error: `Model not found: ${error.message}`,
        rollout_id: rolloutId
      });
    } else if (error instanceof OpenAI.RateLimitError) {
      return res.status(429).json({
        error: `Rate limit exceeded: ${error.message}`,
        rollout_id: rolloutId
      });
    } else if (error instanceof OpenAI.BadRequestError) {
      return res.status(400).json({
        error: `Bad request: ${error.message}`,
        rollout_id: rolloutId
      });
    } else if (error instanceof OpenAI.APIConnectionTimeoutError) {
      return res.status(408).json({
        error: `Request timeout: ${error.message}`,
        rollout_id: rolloutId
      });
    } else if (error instanceof OpenAI.InternalServerError) {
      return res.status(502).json({
        error: `Upstream server error: ${error.message}`,
        rollout_id: rolloutId
      });
    } else if (error instanceof OpenAI.UnprocessableEntityError) {
      return res.status(422).json({
        error: `Invalid request data: ${error.message}`,
        rollout_id: rolloutId
      });
    } else {
      // Network errors, parsing errors, and unexpected errors
      return res.status(500).json({
        error: `Internal error: ${error.message}`,
        rollout_id: rolloutId
      });
    }
  }
}
