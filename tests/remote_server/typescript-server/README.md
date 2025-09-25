# TypeScript Express Server for Remote Rollout Processor

This TypeScript Express server implements the Remote Rollout Processor API contract as specified in the Eval Protocol documentation.

## Features

- **POST /init** - Initialize a rollout with validation using Zod schemas
- **GET /status** - Check the status of a rollout
- **GET /health** - Health check endpoint
- Full TypeScript support with strict type checking
- Request validation using Zod
- Error handling and logging
- CORS and security middleware

## Installation

```bash
pnpm install
```

## Development

```bash
# Run in development mode with hot reload
pnpm run dev

# Build for production
pnpm run build

# Run production build
pnpm run start
```

## API Endpoints

### POST /init

Initialize a new rollout.

**Request Body:**
```json
{
  "rollout_id": "rll_ijkl",
  "model": "openai/gpt-4o",
  "messages": [
    { "role": "user", "content": "Hello" }
  ],
  "tools": null,
  "metadata": {
    "invocation_id": "ivk_abcd",
    "experiment_id": "exp_efgh",
    "rollout_id": "rll_ijkl",
    "run_id": "run_123",
    "row_id": "row_123"
  },
  "num_turns": 2
}
```

**Response:**
```json
{
  "status": "accepted",
  "rollout_id": "rll_ijkl",
  "message": "Rollout initialized successfully"
}
```

### GET /status

Check the status of a rollout.

**Query Parameters:**
- `rollout_id` (required): The ID of the rollout to check

**Response (Running):**
```json
{
  "terminated": false
}
```

**Response (Completed):**
```json
{
  "terminated": true,
  "info": {
    "reason": "completed",
    "ended_at": "2025-01-24T12:34:56Z",
    "num_turns": 2
  }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-24T12:34:56Z"
}
```

## Usage with Eval Protocol

This server can be used with the Eval Protocol's `RemoteRolloutProcessor`:

```python
from eval_protocol import (
    evaluation_test,
    DynamicDataLoader,
    RemoteRolloutProcessor,
)

@pytest.mark.parametrize("completion_params", [{"model": "openai/gpt-4o"}])
@evaluation_test(
    data_loaders=[InlineDataLoader(messages=[[Message(role="user", content="Hello")]])],
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url="http://localhost:3000",
        output_data_loader=create_output_data_loader,
    )
)
def test_remote_http(row: EvaluationRow) -> EvaluationRow:
    return row
```

## Configuration

The server runs on port 3000 by default. You can change this by setting the `PORT` environment variable:

```bash
PORT=8080 pnpm run dev
```

## Error Handling

The server includes comprehensive error handling:
- Request validation errors return 400 with detailed error messages
- Missing rollout IDs return 404
- Server errors return 500 with error details
- All errors are logged to the console

## Development Notes

- The server simulates async rollout execution with a 1-second delay per turn
- Rollout states are stored in memory (not persistent across restarts)
- All requests are validated using Zod schemas
- TypeScript strict mode is enabled for better type safety
