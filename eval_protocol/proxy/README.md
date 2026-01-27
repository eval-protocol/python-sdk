# LiteLLM Metadata Extraction Gateway

A FastAPI-based metadata extraction gateway that sits in front of LiteLLM to inject evaluation metadata into LLM requests and track completions for distributed evaluation workflows.

## Overview

The Metadata Gateway is a proxy service that enhances LiteLLM by:
- **Extracting metadata from URL paths** and injecting it as Langfuse tags
- **Managing Langfuse credentials** per-project without exposing them to clients
- **Tracking completion insertion IDs** in Redis for completeness verification
- **Fetching and validating traces** from Langfuse with built-in retry logic

This enables distributed evaluation systems to track which LLM completions belong to which evaluation runs, ensuring data completeness and proper attribution.

## Architecture

```
┌─────────────┐
│   Client    │
│  (SDK/CLI)  │
└──────┬──────┘
       │ Authorization: Bearer <api_key>
       │ POST /rollout_id/{id}/invocation_id/{id}/.../chat/completions
       ▼
┌─────────────────────────┐
│  Metadata Gateway       │
│  (FastAPI Service)      │
│  - Extract metadata     │
│  - Inject Langfuse keys │
│  - Generate UUID7 IDs   │
└──────┬──────────┬───────┘
       │          │
       ▼          ▼
  ┌────────┐  ┌─────────────┐
  │ Redis  │  │  LiteLLM    │
  │        │  │  Backend    │
  │ Track  │  │             │
  │ IDs    │  └──────┬──────┘
  └────────┘         │
                     ▼
              ┌─────────────┐
              │  Langfuse   │
              │  (Tracing)  │
              └─────────────┘
```

### Components

#### 1. **Metadata Gateway** (`proxy_core/`)
   - **`app.py`**: Main FastAPI application with route definitions
   - **`litellm.py`**: LiteLLM client for forwarding requests
   - **`langfuse.py`**: Langfuse trace fetching with retry logic
   - **`redis_utils.py`**: Redis operations for insertion ID tracking
   - **`models.py`**: Pydantic models for configuration and responses
   - **`auth.py`**: Authentication provider interface (extensible)
   - **`main.py`**: Entry point for running the service

#### 2. **Redis**
   - Stores insertion IDs per rollout for completeness checking
   - Uses Redis Sets: `rollout_id -> {insertion_id_1, insertion_id_2, ...}`

#### 3. **LiteLLM SDK (Direct)**
   - Uses LiteLLM SDK directly for LLM calls (no separate proxy server needed)
   - Integrated with Langfuse via `langfuse_otel` OpenTelemetry callback

## Key Features

### Metadata Injection
URL paths encode evaluation metadata that gets injected as Langfuse tags:
- `rollout_id`: Unique ID for a batch evaluation run
- `invocation_id`: ID for a single invocation within a rollout
- `experiment_id`: Experiment identifier
- `run_id`: Run identifier within an experiment
- `row_id`: Dataset row identifier
- `insertion_id`: Auto-generated UUID7 for this specific completion

### Completeness Tracking
1. **On chat completion**: Generate UUID7 insertion_id and store in Redis
2. **On trace fetch**: Verify all expected insertion_ids are present in Langfuse
3. **Retry logic**: Automatic retries with exponential backoff for incomplete traces

### Multi-Project Support
- Store Langfuse credentials for multiple projects in `secrets.yaml`
- Route requests to the correct project via `project_id` in URL or use default
- Credentials never exposed to clients

## Setup

### Prerequisites
- Docker and Docker Compose (recommended)
- Python 3.11+ (for local development)

### Local Development: Docker Compose

1. **Create secrets file:**
   ```bash
   cp proxy_core/secrets.yaml.example proxy_core/secrets.yaml
   ```

2. **Edit `proxy_core/secrets.yaml`** with your Langfuse credentials.
**Important**: use your real Langfuse project ID (e.g. `cmg00asdf0123...`).
   ```yaml
   langfuse_keys:
     my-project:
       public_key: pk-lf-...
       secret_key: sk-lf-...
   default_project_id: my-project
   ```

3. **Start services:**
   ```bash
   docker-compose up -d
   ```

4. **Verify services are running:**
   ```bash
   curl http://localhost:4000/health
   # Expected: {"status":"healthy","service":"metadata-proxy"}
   ```

The gateway will be available at `http://localhost:4000`.

## API Reference

### Chat Completions

#### With Full Metadata
```
POST /rollout_id/{rollout_id}/invocation_id/{invocation_id}/experiment_id/{experiment_id}/run_id/{run_id}/row_id/{row_id}/chat/completions
POST /project_id/{project_id}/rollout_id/{rollout_id}/.../chat/completions
```

**Features:**
- Extracts metadata from URL path
- Generates UUID7 insertion_id
- Injects Langfuse credentials
- Tracks insertion_id in Redis
- Forwards to LiteLLM

**Request:**
```bash
curl -X POST http://localhost:4000/rollout_id/abc123/invocation_id/inv1/experiment_id/exp1/run_id/run1/row_id/row1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-..." \
  -d '{
    "model": "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Response:** Standard OpenAI chat completion response

#### With Project Only
```
POST /project_id/{project_id}/chat/completions
```

For completions that don't need rollout tracking.

#### With Encoded Base URL
```
POST /rollout_id/{rollout_id}/.../encoded_base_url/{encoded_base_url}/chat/completions
```

The `encoded_base_url` is base64-encoded URL string injected into the request body as `base_url`.

### Trace Fetching

#### Fetch All Langfuse Traces
```
GET /traces?tags=rollout_id:abc123
GET /v1/traces?tags=rollout_id:abc123
GET /project_id/{project_id}/traces?tags=rollout_id:abc123
GET /v1/project_id/{project_id}/traces?tags=rollout_id:abc123
```

Waits for all expected insertion_ids to complete before returning all traces.

#### Fetch Latest Langfuse Trace (Pointwise)
```
GET /traces/pointwise?tags=rollout_id:abc123
GET /v1/traces/pointwise?tags=rollout_id:abc123
GET /project_id/{project_id}/traces/pointwise?tags=rollout_id:abc123
GET /v1/project_id/{project_id}/traces/pointwise?tags=rollout_id:abc123
```

Returns only the latest trace (UUID v7 time-ordered). Much faster for pointwise evaluations where you only need the final accumulated result.

**Required Query Parameters:**
- `tags`: Array of tags (must include at least one `rollout_id:*` tag)

**Optional Query Parameters:**
- `limit`: Max traces to fetch (default: 100)
- `sample_size`: Random sample size if more traces found
- `user_id`, `session_id`, `name`, `environment`, `version`, `release`: Langfuse filters
- `fields`: Comma-separated fields to include
- `hours_back`: Fetch traces from last N hours
- `from_timestamp`, `to_timestamp`: ISO datetime strings for time range
- `sleep_between_gets`: Delay between trace.get calls (default: 2.5s)
- `max_retries`: Retry attempts for incomplete traces (default: 3)

**Completeness Logic:**
1. Fetches traces from Langfuse matching tags
2. Extracts insertion_ids from trace tags
3. Compares with expected insertion_ids in Redis
4. Retries with exponential backoff if incomplete
5. Returns 404 if still incomplete after max_retries

**Response:**
```json
{
  "project_id": "my-project",
  "total_traces": 42,
  "traces": [
    {
      "id": "trace-123",
      "name": "chat-completion",
      "tags": ["rollout_id:abc123", "insertion_id:uuid7..."],
      "input": {...},
      "output": {...},
      "observations": [...]
    }
  ]
}
```

### Health Check
```
GET /health
```

Returns service health status.

### Catch-All Proxy
```
ANY /{path}
```

Forwards any other request to LiteLLM backend with API key injection.

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_HOST` | Yes | - | Redis hostname |
| `REDIS_PORT` | No | 6379 | Redis port |
| `REDIS_PASSWORD` | No | - | Redis password |
| `SECRETS_PATH` | No | `proxy_core/secrets.yaml` | Path to secrets file (YAML) |
| `LANGFUSE_HOST` | No | `https://us.cloud.langfuse.com` | Langfuse OTEL host for tracing |
| `REQUEST_TIMEOUT` | No | 300.0 | Request timeout (LLM calls) in seconds |
| `LOG_LEVEL` | No | INFO | Logging level |
| `PORT` | No | 4000 | Gateway port |

### Secrets Configuration

Create `proxy_core/secrets.yaml`:
```yaml
langfuse_keys:
  project-1:
    public_key: pk-lf-...
    secret_key: sk-lf-...
  project-2:
    public_key: pk-lf-...
    secret_key: sk-lf-...
default_project_id: project-1
```

**Security:** `secrets.yaml` is ignored via `.gitignore`.

### LiteLLM Configuration

The `config_no_cache.yaml` configures LiteLLM (only needed if running a standalone LiteLLM proxy):
```yaml
model_list:
  - model_name: "*"
    litellm_params:
      model: "*"
litellm_settings:
  callbacks: ["langfuse_otel"]
  drop_params: True
general_settings:
  allow_client_side_credentials: true
```

Key settings:
- **Wildcard model support**: Route any model to any provider
- **Langfuse OTEL**: OpenTelemetry-based tracing via `langfuse_otel` callback
- **Client-side credentials**: Accept API keys from request body

**Note:** The proxy now uses the LiteLLM SDK directly with `langfuse_otel` integration, so a separate LiteLLM proxy server is no longer required.

## Security Considerations

### Authentication
- **Default**: No authentication (`NoAuthProvider`)
- **Extensible**: Implement custom `AuthProvider` for production
- **API Keys**: Client API keys forwarded to LiteLLM, never stored

### Trace Fetching Security
- **Required rollout_id tag**: Prevents fetching all traces
- **Project isolation**: Projects can only access their own Langfuse data
- **Optional auth**: `/traces` endpoint can require authentication

### Best Practices
1. **Never commit `secrets.json`** - use environment variables in production
2. **Use HTTPS** in production deployments
3. **Implement proper authentication** for production use
4. **Rotate Langfuse keys** regularly
5. **Monitor Redis memory** usage for large rollouts

## Deployment

### Docker Compose (Development)
```bash
docker-compose up -d
```

### Kubernetes
Create deployment with:
- Secrets for `secrets.json` and Redis credentials
- Service for internal/external access
- ConfigMap for LiteLLM config
- Redis StatefulSet or managed Redis service

## Development

### Project Structure
```
eval_protocol/proxy/
├── proxy_core/              # Main application package
│   ├── __init__.py
│   ├── app.py              # FastAPI routes
│   ├── litellm.py          # LiteLLM client
│   ├── langfuse.py         # Langfuse integration
│   ├── redis_utils.py      # Redis operations
│   ├── models.py           # Pydantic models
│   ├── auth.py             # Authentication
│   ├── main.py             # Entry point
│   └── secrets.yaml.example
├── docker-compose.yml       # Local development stack
├── Dockerfile.gateway       # Gateway container
├── config_no_cache.yaml     # LiteLLM config
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Testing

#### Test chat completion:
```bash
curl -X POST http://localhost:4000/rollout_id/test123/invocation_id/inv1/experiment_id/exp1/run_id/run1/row_id/row1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $FIREWORKS_API_KEY" \
  -d '{
    "model": "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

#### Test trace fetching:
```bash
curl "http://localhost:4000/traces?tags=rollout_id:test123" \
  -H "Authorization: Bearer your-auth-token"
```

#### Check Redis:
```bash
redis-cli
> SMEMBERS test123  # View insertion_ids for rollout
```
