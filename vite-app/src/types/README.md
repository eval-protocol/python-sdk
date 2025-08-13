# Evaluation Protocol Types

This directory contains Zod schemas and TypeScript types that mirror the Pydantic models from the Python `eval_protocol/models.py` file.

## Files

- `eval-protocol.ts` - Main Zod schemas and TypeScript types
- `eval-protocol-utils.ts` - Utility functions for working with evaluation data
- `index.ts` - Re-exports for easier importing

## Usage

### Basic Import

```typescript
import { EvaluationRow, Message, EvaluateResult } from '@/types';
```

### Using Zod Schemas for Validation

```typescript
import { EvaluationRowSchema, MessageSchema } from '@/types';

// Validate incoming data
const validateEvaluationRow = (data: unknown) => {
  return EvaluationRowSchema.parse(data);
};

// Safe parsing with error handling
const safeParseMessage = (data: unknown) => {
  const result = MessageSchema.safeParse(data);
  if (!result.success) {
    console.error('Validation failed:', result.error);
    return null;
  }
  return result.data;
};
```

### Using Utility Functions

```typescript
import { evalRowUtils, messageUtils, evaluateResultUtils } from '@/types/eval-protocol-utils';

// Check if evaluation is trajectory-based
const isTrajectory = evalRowUtils.isTrajectoryEvaluation(evaluationRow);

// Get conversation length
const length = evalRowUtils.getConversationLength(evaluationRow);

// Get system message
const systemMsg = evalRowUtils.getSystemMessage(evaluationRow);

// Check if message has tool calls
const hasTools = messageUtils.hasToolCalls(message);

// Get message content as string
const content = messageUtils.getContentAsString(message);
```

## Key Types

### Core Types

- `EvaluationRow` - Main data structure for evaluation results
- `Message` - Chat message with role, content, and optional metadata
- `EvaluateResult` - Evaluation results with scores and metrics
- `MetricResult` - Individual metric results
- `StepOutput` - Per-step evaluation output for RL scenarios

### Agent Evaluation Framework (V2)

- `TaskDefinitionModel` - Task configuration for agent evaluation
- `ResourceServerConfig` - Server configuration for tasks
- `EvaluationCriteriaModel` - Criteria for evaluating task success

### MCP Configuration

- `MCPMultiClientConfiguration` - MCP server configuration
- `MCPConfigurationServerStdio` - Stdio-based MCP server
- `MCPConfigurationServerUrl` - URL-based MCP server

## Validation Examples

### Validating API Responses

```typescript
import { EvaluationRowSchema } from '@/types';

async function fetchEvaluationData(): Promise<EvaluationRow> {
  const response = await fetch('/api/evaluation');
  const data = await response.json();

  // Validate the response
  return EvaluationRowSchema.parse(data);
}
```

### Creating New Evaluation Data

```typescript
import { EvaluationRowSchema, MessageSchema } from '@/types';

const newMessage: Message = {
  role: 'user',
  content: 'Hello, how are you?'
};

// Validate the message
const validatedMessage = MessageSchema.parse(newMessage);

const newEvaluationRow = {
  messages: [validatedMessage],
  input_metadata: {
    row_id: 'unique-id-123'
  },
  created_at: new Date()
};

// Validate the evaluation row
const validatedRow = EvaluationRowSchema.parse(newEvaluationRow);
```

## Type Safety

All types are derived from Zod schemas, ensuring runtime validation and compile-time type safety. The schemas include:

- Field validation (e.g., score ranges, required fields)
- Default values
- Optional fields
- Union types for flexible content
- Descriptive error messages

## Migration from Python

The TypeScript types closely mirror the Python Pydantic models:

- `BaseModel` → `z.object()`
- `Field()` → `z.string().describe()`
- `Optional[T]` → `z.optional()`
- `List[T]` → `z.array()`
- `Dict[str, Any]` → `z.record(z.any())`
- `extra="allow"` → `.passthrough()`
