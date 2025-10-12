# Eval Protocol (EP)

[![PyPI - Version](https://img.shields.io/pypi/v/eval-protocol)](https://pypi.org/project/eval-protocol/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/eval-protocol/python-sdk)

**Stop guessing which AI model to use. Build a data-driven model leaderboard.**

With hundreds of models and configs, you need objective data to choose the right one for your use case. EP helps you evaluate real traces, compare models, and visualize results locally.

## 🚀 Features

- **Pytest authoring**: `@evaluation_test` decorator to configure evaluations
- **Robust rollouts**: Handles flaky LLM APIs and parallel execution
- **Integrations**: Works with Langfuse, LangSmith, Braintrust, Responses API
- **Agent support**: LangGraph and Pydantic AI
- **MCP RL envs**: Build reinforcement learning environments with MCP
- **Built-in benchmarks**: AIME, tau-bench
- **LLM judge**: Stack-rank models using pairwise Arena-Hard-Auto
- **Local UI**: Pivot/table views for real-time analysis

## ⚡ Quickstart (local traces + local models)

This end-to-end uses a local Langfuse (Docker Compose), seeds app traces, then runs a model picker with a Fireworks-based judge and your local models (Ollama or llama.cpp). See `examples/local_langfuse_litellm_ollama/README.md` for a full guide.

### 1) Start Langfuse locally (compose file included)

```bash
# From repo root
docker compose -f examples/local_langfuse_litellm_ollama/langfuse-docker-compose.yml up -d
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=...  # create in Langfuse UI
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_ENVIRONMENT=local
```

Open `http://localhost:3000` and confirm the UI loads.

### 2) Seed traces (PydanticAgent, no external DB required)

```bash
export FIREWORKS_API_KEY=...
export CHINOOK_USE_STUB_DB=1
make -C . local-generate-chinook
```

Optionally verify the adapter can fetch rows:

```bash
make -C . local-adapter-smoke
```

### 3) Evaluate with local models

Ollama only, direct (bypass LiteLLM):

```bash
export DIRECT_OLLAMA=1
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODELS='ollama/llama3.1'   # comma-separated to compare multiple
export FIREWORKS_API_KEY=...
# Optional debug to verify calls and logging
export EP_DEBUG=1
pytest eval_protocol/quickstart/llm_judge_langfuse_local.py -k test_llm_judge_local -q
```

Optional: via LiteLLM router (Ollama/llama.cpp):

```bash
export LITELLM_API_KEY=local-demo-key
litellm --config examples/local_langfuse_litellm_ollama/litellm-config.yaml --port 4000
export LITELLM_BASE_URL=http://127.0.0.1:4000
export OLLAMA_MODELS='ollama/llama3.1,ollama/llama3.2:1b'
# Optional debug to verify router calls and logging
export EP_DEBUG=1
pytest eval_protocol/quickstart/llm_judge_langfuse_local.py -k test_llm_judge_local -q
```

The pytest output includes local links for a leaderboard and row-level traces at `http://localhost:8000`.

## Basic AHA judge example (remote APIs)

Install with your tracing platform extras and set API keys:

```bash
pip install 'eval-protocol[langfuse]'

# Model API keys (set what you need)
export OPENAI_API_KEY=...
export FIREWORKS_API_KEY=...
export GEMINI_API_KEY=...

# Platform keys
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_HOST=https://your-deployment.com  # optional
```

Minimal evaluation using the built-in AHA judge:

```python
from datetime import datetime
import pytest

from eval_protocol import (
    evaluation_test,
    aha_judge,
    EvaluationRow,
    SingleTurnRolloutProcessor,
    DynamicDataLoader,
    create_langfuse_adapter,
)


def langfuse_data_generator() -> list[EvaluationRow]:
    adapter = create_langfuse_adapter()
    return adapter.get_evaluation_rows(
        to_timestamp=datetime.utcnow(),
        limit=20,
        sample_size=5,
    )


@pytest.mark.parametrize(
    "completion_params",
    [
        {"model": "openai/gpt-4.1"},
        {"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"},
    ],
)
@evaluation_test(
    data_loaders=DynamicDataLoader(generators=[langfuse_data_generator]),
    rollout_processor=SingleTurnRolloutProcessor(),
)
async def test_llm_judge(row: EvaluationRow) -> EvaluationRow:
    return await aha_judge(row)
```

Run it:

```bash
pytest -q -s
```

The pytest output includes local links for a leaderboard and row-level traces (pivot/table) at `http://localhost:8000`.

## Installation

This library requires Python >= 3.10.

### pip

```bash
pip install eval-protocol
```

### uv (recommended)

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to your project
uv add eval-protocol
```

## 🧑‍💻 Developer notes

- The `eval-protocol logs` command currently may show no rows in some local setups even when Langfuse traces exist; use the local UI links printed by pytest and the Langfuse UI to inspect results. We’re tracking improvements to unify local logs with external trace sources.
- For Langfuse seeding, prefer `tests/chinook/langfuse/generate_traces.py` with `CHINOOK_USE_STUB_DB=1` to avoid external DBs.
- To compare multiple local models, set `OLLAMA_MODELS` (comma-separated) or use the LiteLLM config for mix-and-match backends.

## 📚 Resources

- **[Documentation](https://evalprotocol.io)** – Guides and API reference
- **[Discord](https://discord.com/channels/1137072072808472616/1400975572405850155)** – Community
- **[GitHub](https://github.com/eval-protocol/python-sdk)** – Source and examples

## License

[MIT](LICENSE)
