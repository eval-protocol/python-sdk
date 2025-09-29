"""Fully local Langfuse + LiteLLM example with Fireworks judge.

This example shows how to evaluate local model responses (served via a local
LiteLLM router in front of `ollama` and/or `llama.cpp`) using the default
Arena-Hard-Auto ("aha") judge, which runs on Fireworks. Traces are pulled from
your self-hosted Langfuse instance using the built-in adapter.

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

4. Export your Fireworks credentials for the LLM judge::

       export FIREWORKS_API_KEY=...  # required for the judge
       # optional if using organization-scoped models
       export FIREWORKS_ACCOUNT_ID=...

5. Point the example at the router. The defaults below expect the router on
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
from eval_protocol.quickstart.utils import assistant_to_ground_truth
# Note: We keep the default aha judge (Fireworks) from utils.JUDGE_CONFIGS.

# ---------------------------------------------------------------------------
# Force direct Ollama usage (no LiteLLM router) for this example
# ---------------------------------------------------------------------------
# Avoid unexpected input param overrides in local runs
os.environ.pop("EP_INPUT_PARAMS_JSON", None)

# ---------------------------------------------------------------------------
# Hardcoded local configuration (no env required for models/routing)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODELS = [
    "ollama/llama3.1",
]
LANGFUSE_TAGS = ["chinook_sql"]
LANGFUSE_LIMIT = 200
LANGFUSE_SAMPLE_SIZE = 20
LANGFUSE_SLEEP_BETWEEN_GETS = 1.0
LANGFUSE_MAX_RETRIES = 6
LANGFUSE_HOURS_BACK = 48


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def langfuse_local_data_generator() -> list[EvaluationRow]:
    """Fetch evaluation rows from a local Langfuse deployment."""

    adapter = create_langfuse_adapter()
    print("[EP-Debug] Pulling rows from Langfuse with hardcoded config:")
    print(
        f"  tags={LANGFUSE_TAGS}, limit={LANGFUSE_LIMIT}, sample_size={LANGFUSE_SAMPLE_SIZE}, include_tool_calls=True"
    )

    rows = adapter.get_evaluation_rows(
        environment=None,
        tags=LANGFUSE_TAGS,
        limit=LANGFUSE_LIMIT,
        sample_size=LANGFUSE_SAMPLE_SIZE,
        include_tool_calls=True,
        sleep_between_gets=LANGFUSE_SLEEP_BETWEEN_GETS,
        max_retries=LANGFUSE_MAX_RETRIES,
        hours_back=LANGFUSE_HOURS_BACK,
        from_timestamp=None,
        to_timestamp=datetime.utcnow(),
    )
    print(f"[EP-Debug] Langfuse adapter returned rows (preprocess pending): {len(rows)}")
    return rows


def _preprocess_rows(data: list[EvaluationRow]) -> list[EvaluationRow]:
    """Mirror quickstart pattern: run multi_turn split, then drop empties with debug."""
    split_rows = multi_turn_assistant_to_ground_truth(data)
    print(f"[EP-Debug] After multi_turn_assistant_to_ground_truth: {len(split_rows)} rows")

    # Keep only rows that have at least one message before assistant turn
    filtered = [r for r in split_rows if r.messages and len(r.messages) > 0]
    if len(filtered) != len(split_rows):
        print(f"[EP-Debug] Dropped {len(split_rows) - len(filtered)} rows with empty messages after split")

    # Show a small sample for inspection
    for r in filtered[:2]:
        try:
            roles = [m.role for m in r.messages]
            gt_repr = str(r.ground_truth or "")
            print(f"[EP-Debug] Row sample: msg_count={len(r.messages)} roles={roles} gt_len={len(gt_repr)}")
        except Exception:
            pass
    if filtered:
        return filtered

    # Fallback: use last assistant as ground truth without split
    print("[EP-Debug] Fallback preprocess: applying assistant_to_ground_truth")
    fallback_rows = assistant_to_ground_truth(data)
    fallback_filtered = [r for r in fallback_rows if r.messages and len(r.messages) > 0]
    if len(fallback_filtered) != len(fallback_rows):
        print(f"[EP-Debug] Fallback dropped {len(fallback_rows) - len(fallback_filtered)} rows with empty messages")
    for r in fallback_filtered[:2]:
        try:
            roles = [m.role for m in r.messages]
            gt_repr = str(r.ground_truth or "")
            print(f"[EP-Debug] Fallback sample: msg_count={len(r.messages)} roles={roles} gt_len={len(gt_repr)}")
        except Exception:
            pass
    return fallback_filtered


# Hardcoded completion params for local Ollama via LiteLLM SDK (no proxy)
_PARAMS = [
    {
        "model": m,
        "base_url": OLLAMA_BASE_URL,
        "extra_body": {"stream": False},
    }
    for m in OLLAMA_MODELS
]


@pytest.mark.parametrize("completion_params", _PARAMS)
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip local example in CI")
@pytest.mark.skipif(
    not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"),
    reason="LANGFUSE credentials not configured",
)
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[langfuse_local_data_generator],
        preprocess_fn=_preprocess_rows,
    ),
    rollout_processor=SingleTurnRolloutProcessor(),
    max_concurrent_evaluations=1,
)
async def test_llm_judge_local(row: EvaluationRow) -> EvaluationRow:
    """Evaluate one Langfuse trace row with the local aha judge."""
    # Use default Fireworks-based judge and push score back to Langfuse
    adapter = create_langfuse_adapter()
    if os.getenv("EP_DEBUG", "0").strip() == "1":
        try:
            cp = row.input_metadata.completion_params
            print(
                f"[EP-Debug] Starting judge for row: rollout_id={row.execution_metadata.rollout_id}, model={cp.get('model') if cp else 'n/a'}"
            )
        except Exception:
            pass
    return await aha_judge(row, adapter=adapter)
