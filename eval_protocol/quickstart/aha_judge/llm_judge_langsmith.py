"""
LLM Judge quickstart that PULLS DATA FROM LANGSMITH and persists results locally via Eval Protocol.

This mirrors `eval_protocol/quickstart/llm_judge.py` (Langfuse source), but uses
LangSmith datasets/examples as the source of evaluation rows.

Setup:
  pip install -U langsmith

Env vars:
  export LANGSMITH_API_KEY=...             # required to fetch examples
  export LS_DATASET="ep_langsmith_demo_ds"  # dataset to pull examples from

Judge model keys:
  - Default judge is "gemini-2.5-pro" from utils; requires GEMINI_API_KEY
  - Or set judge in the code to "gpt-4.1" and export OPENAI_API_KEY

Run:
  pytest python-sdk/eval_protocol/quickstart/llm_judge_langsmith.py -q -s
"""

import os
from typing import List, Optional

import pytest

from eval_protocol import (
    evaluation_test,
    aha_judge,
    EvaluationRow,
    SingleTurnRolloutProcessor,
    LangSmithAdapter,
    DynamicDataLoader,
    multi_turn_assistant_to_ground_truth,
)


def langsmith_data_generator() -> List[EvaluationRow]:
    """Fetch LangSmith root runs and convert to EvaluationRow, mirroring Langfuse adapter shape.

    - Extract messages from run.inputs and run.outputs
    - Append assistant message from outputs so we can derive ground_truth
    - Store run_id in input_metadata.session_data
    """
    project = os.getenv("LS_PROJECT", "ep-langgraph-examples")
    try:
        adapter = LangSmithAdapter()
        return adapter.get_evaluation_rows(project_name=project, limit=20, include_tool_calls=True)
    except Exception as e:
        print(f"❌ LangSmithAdapter failed: {e}")
        return []


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI")
@pytest.mark.parametrize(
    "completion_params",
    [
        {
            "model": "fireworks_ai/accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
        },
        {
            "max_tokens": 131000,
            "extra_body": {"reasoning_effort": "low"},
            "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
        },
    ],
)
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[langsmith_data_generator],
        preprocess_fn=multi_turn_assistant_to_ground_truth,
    ),
    rollout_processor=SingleTurnRolloutProcessor(),
    preprocess_fn=multi_turn_assistant_to_ground_truth,
    max_concurrent_evaluations=2,
)
async def test_llm_judge_langsmith(row: EvaluationRow) -> EvaluationRow:
    """LLM Judge evaluation over LangSmith-sourced rows, persisted locally by Eval Protocol.

    Mirrors quickstart/llm_judge.py, using Arena-Hard-Auto style pairwise judgment.
    """
    return await aha_judge(row)
