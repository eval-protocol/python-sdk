"""Fully local Langfuse + LiteLLM example for the aha judge.

This example shows how to run the Arena-Hard-Auto ("aha") judge entirely on
local infrastructure. It reuses the Langfuse adapter to pull traces from a
self-hosted Langfuse deployment and evaluates them with a local LiteLLM router
that fronts both `ollama` and `llama.cpp` backends.

Prerequisites
-------------
1. Start Langfuse locally and export the usual environment variables so the
   SDK can connect::

       docker compose up -d
       export LANGFUSE_PUBLIC_KEY=local
       export LANGFUSE_SECRET_KEY=local
       export LANGFUSE_HOST=http://localhost:3000

   Replace the credentials with whatever you configured for your local
   deployment.

2. Launch the model backends. The example below assumes:

   * ``ollama`` is running on ``http://127.0.0.1:11434`` with the model
     ``llama3.1`` pulled.
   * A ``llama.cpp`` server is running on ``http://127.0.0.1:8080`` that serves
     ``Meta-Llama-3-8B-Instruct`` (adjust the path/model name for your set-up).

3. Start a LiteLLM router that proxies both backends. Save the following to
   ``litellm-config.yaml`` (change model names as desired)::

       model_list:
         - model_name: "judge/llama3.1"
           litellm_params:
             model: "ollama/llama3.1"
             api_base: "http://127.0.0.1:11434"
         - model_name: "candidate/llama3.8b"
           litellm_params:
             model: "llama.cpp"
             api_base: "http://127.0.0.1:8080/v1"
             model_path: "/path/to/Meta-Llama-3-8B-Instruct.gguf"

       litellm_settings:
         drop_params: true
         telemetry: false

   Then launch the router::

       export LITELLM_API_KEY=local-demo-key
       litellm --config litellm-config.yaml --port 4000

4. Point the example at the router. The defaults below expect the router on
   ``http://127.0.0.1:4000`` and use ``judge/llama3.1`` as the judge model.
   Override them via ``LITELLM_BASE_URL`` and ``LOCAL_JUDGE_MODEL`` if your
   configuration is different.

Running the example
-------------------
With the services running, execute::

    pytest eval_protocol/quickstart/llm_judge_langfuse_local.py -k test_llm_judge_local

The test will fetch traces from the local Langfuse instance, convert each
assistant turn into an ``EvaluationRow``, and score them with the local judge.
"""

from __future__ import annotations

from datetime import datetime
import os

import pytest

from eval_protocol import (
    DynamicDataLoader,
    EvaluationRow,
    SingleTurnRolloutProcessor,
    aha_judge,
    create_langfuse_adapter,
    evaluation_test,
    multi_turn_assistant_to_ground_truth,
)
from eval_protocol.quickstart.utils import JUDGE_CONFIGS

# ---------------------------------------------------------------------------
# Local judge configuration
# ---------------------------------------------------------------------------
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://127.0.0.1:4000")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "local-demo-key")
LOCAL_JUDGE_MODEL = os.getenv("LOCAL_JUDGE_MODEL", "judge/llama3.1")
LOCAL_JUDGE_TEMPERATURE = float(os.getenv("LOCAL_JUDGE_TEMPERATURE", "0.0"))
LOCAL_JUDGE_MAX_TOKENS = int(os.getenv("LOCAL_JUDGE_MAX_TOKENS", "4096"))

# Register a judge profile that points to the local LiteLLM router. Importing
# the module is enough for other quickstart helpers to discover it.
JUDGE_CONFIGS.setdefault(
    "local-litellm",
    {
        "model": LOCAL_JUDGE_MODEL,
        "temperature": LOCAL_JUDGE_TEMPERATURE,
        "max_tokens": LOCAL_JUDGE_MAX_TOKENS,
        "api_key": LITELLM_API_KEY,
        "base_url": LITELLM_BASE_URL,
    },
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def langfuse_local_data_generator() -> list[EvaluationRow]:
    """Fetch evaluation rows from a local Langfuse deployment."""

    adapter = create_langfuse_adapter()
    return adapter.get_evaluation_rows(
        environment=os.getenv("LANGFUSE_ENVIRONMENT", "local"),
        limit=int(os.getenv("LANGFUSE_LIMIT", "200")),
        sample_size=int(os.getenv("LANGFUSE_SAMPLE_SIZE", "20")),
        include_tool_calls=bool(int(os.getenv("LANGFUSE_INCLUDE_TOOL_CALLS", "1"))),
        sleep_between_gets=float(os.getenv("LANGFUSE_SLEEP", "0.5")),
        max_retries=int(os.getenv("LANGFUSE_MAX_RETRIES", "3")),
        from_timestamp=None,
        to_timestamp=datetime.utcnow(),
    )


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip local example in CI")
@pytest.mark.skipif(
    not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"),
    reason="LANGFUSE credentials not configured",
)
@pytest.mark.parametrize(
    "completion_params",
    [
        {
            "model": "candidate/llama3.8b",
            "api_key": LITELLM_API_KEY,
            "base_url": LITELLM_BASE_URL,
            "temperature": float(os.getenv("LOCAL_CANDIDATE_TEMPERATURE", "0.2")),
        },
        {
            "model": "ollama/llama3.1",
            "api_key": LITELLM_API_KEY,
            "base_url": LITELLM_BASE_URL,
            "extra_body": {"stream": False},
        },
    ],
)
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[langfuse_local_data_generator],
        preprocess_fn=multi_turn_assistant_to_ground_truth,
    ),
    rollout_processor=SingleTurnRolloutProcessor(),
    max_concurrent_evaluations=int(os.getenv("LOCAL_MAX_CONCURRENCY", "1")),
)
async def test_llm_judge_local(row: EvaluationRow) -> EvaluationRow:
    """Evaluate one Langfuse trace row with the local aha judge."""

    return await aha_judge(row, judge_name="local-litellm")
