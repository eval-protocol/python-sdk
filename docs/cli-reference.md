# CLI Reference

**ep** - eval-protocol: Tools for evaluation and reward modeling

## Global Options

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `--verbose`, `-v` |  | false | No | Enable verbose logging |
| `--profile` |  | - | No | Fireworks profile to use (reads ~/.fireworks/profiles/<name>/auth.ini and settings.ini) |
| `--server` |  | - | No | Fireworks API server hostname or URL (e.g., dev.api.fireworks.ai or https://dev.api.fireworks.ai) |

## Commands

### `ep logs`

Serve logs with file watching and real-time updates

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `--port` | int | `8000` | No | Port to bind to (default: 8000) |
| `--debug` |  | false | No | Enable debug mode |
| `--disable-elasticsearch-setup` |  | false | No | Disable Elasticsearch setup |
| `--use-env-elasticsearch-config` |  | false | No | Use env vars for Elasticsearch config (requires ELASTICSEARCH_URL, ELASTICSEARCH_API_KEY, ELASTICSEARCH_INDEX_NAME) |
| `--use-fireworks` |  | false | No | Force Fireworks tracing backend for logs UI (overrides env auto-detection) |
| `--use-elasticsearch` |  | false | No | Force Elasticsearch backend for logs UI (overrides env auto-detection) |

### `ep upload`

Scan for evaluation tests, select, and upload as Fireworks evaluators

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `--path` |  | `.` | No | Path to search for evaluation tests (default: current directory) |
| `--entry` |  | - | No | Entrypoint of evaluation test to upload (module:function or path::function). For multiple, separate by commas. |
| `--id` |  | - | No | Evaluator ID to use (if multiple selections, a numeric suffix is appended) |
| `--display-name` |  | - | No | Display name for evaluator (defaults to ID) |
| `--description` |  | - | No | Description for evaluator |
| `--force` |  | false | No | Overwrite existing evaluator with the same ID |
| `--yes`, `-y` |  | false | No | Non-interactive: upload all discovered evaluation tests |
| `--env-file` |  | - | No | Path to .env file containing secrets to upload (default: .env in current directory) |

### `ep create`

Resource creation commands

#### `ep create rft`

Create a Reinforcement Fine-tuning Job on Fireworks

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `--evaluator` |  | - | No | Evaluator ID or fully-qualified resource (accounts/{acct}/evaluators/{id}); if omitted, derive from local tests |
| `--dataset` |  | - | No | Use existing dataset (ID or resource 'accounts/{acct}/datasets/{id}') to skip local materialization |
| `--dataset-jsonl` |  | - | No | Path to JSONL to upload as a new Fireworks dataset |
| `--dataset-builder` |  | - | No | Explicit dataset builder spec (module::function or path::function) |
| `--dataset-display-name` |  | - | No | Display name for dataset on Fireworks (defaults to dataset id) |
| `--base-model` |  | - | No | Base model resource id |
| `--warm-start-from` |  | - | No | Addon model to warm start from |
| `--output-model` |  | - | No | Output model id (defaults from evaluator) |
| `--epochs` | int | `1` | No | - |
| `--batch-size` | int | `128000` | No | - |
| `--learning-rate` | float | `3e-05` | No | - |
| `--max-context-length` | int | `65536` | No | - |
| `--lora-rank` | int | `16` | No | - |
| `--gradient-accumulation-steps` | int | - | No | Number of gradient accumulation steps |
| `--learning-rate-warmup-steps` | int | - | No | Number of LR warmup steps |
| `--accelerator-count` | int | - | No | - |
| `--region` |  | - | No | Fireworks region enum value |
| `--display-name` |  | - | No | RFT job display name |
| `--evaluation-dataset` |  | - | No | Optional separate eval dataset id |
| `--eval-auto-carveout` |  | true | No | - |
| `--no-eval-auto-carveout` |  | true | No | - |
| `--chunk-size` | int | `100` | No | Data chunk size for rollout batching |
| `--temperature` | float | - | No | - |
| `--top-p` | float | - | No | - |
| `--top-k` | int | - | No | - |
| `--max-output-tokens` | int | `32768` | No | - |
| `--response-candidates-count` | int | `8` | No | - |
| `--extra-body` |  | - | No | JSON string for extra inference params |
| `--mcp-server` |  | - | No | The MCP server resource name to use for the reinforcement fine-tuning job. |
| `--wandb-enabled` |  | false | No | - |
| `--wandb-project` |  | - | No | - |
| `--wandb-entity` |  | - | No | - |
| `--wandb-run-id` |  | - | No | - |
| `--wandb-api-key` |  | - | No | - |
| `--job-id` |  | - | No | Specify an explicit RFT job id |
| `--yes`, `-y` |  | false | No | Non-interactive mode |
| `--dry-run` |  | false | No | Print planned REST calls without sending |
| `--force` |  | false | No | Overwrite existing evaluator with the same ID |
| `--skip-validation` |  | false | No | Skip local dataset and evaluator validation before creating the RFT job |
| `--ignore-docker` |  | false | No | Ignore Dockerfile even if present; run pytest on host during evaluator validation |
| `--docker-build-extra` |  | `` | No | Extra flags to pass to 'docker build' when validating evaluator (quoted string, e.g. "--no-cache --pull --progress=plain") |
| `--docker-run-extra` |  | `` | No | Extra flags to pass to 'docker run' when validating evaluator (quoted string, e.g. "--env-file .env --memory=8g") |

### `ep local-test`

Select an evaluation test and run it locally. If a Dockerfile exists, build and run via Docker; otherwise run on host.

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `--entry` |  | - | No | Entrypoint to run (path::function or path). If not provided, a selector will be shown (unless --yes). |
| `--ignore-docker` |  | false | No | Ignore Dockerfile even if present; run pytest on host |
| `--yes`, `-y` |  | false | No | Non-interactive: if multiple tests exist and no --entry, fails with guidance |
| `--docker-build-extra` |  | `` | No | Extra flags to pass to 'docker build' (quoted string, e.g. "--no-cache --pull --progress=plain") |
| `--docker-run-extra` |  | `` | No | Extra flags to pass to 'docker run' (quoted string, e.g. "--env-file .env --memory=8g") |
