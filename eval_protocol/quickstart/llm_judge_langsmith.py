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
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
import pytest

from eval_protocol.models import EvaluationRow, Message, EvaluateResult, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor
from eval_protocol.quickstart.utils import (
    split_multi_turn_rows,
    JUDGE_CONFIGS,
    calculate_bootstrap_scores,
    run_judgment_async,
)
from eval_protocol.adapters.langsmith import LangSmithAdapter


def fetch_langsmith_traces_as_evaluation_rows(
    project_name: Optional[str] = None,
    limit: int = 20,
) -> List[EvaluationRow]:
    """Fetch LangSmith root runs and convert to EvaluationRow, mirroring Langfuse adapter shape.

    - Extract messages from run.inputs and run.outputs
    - Append assistant message from outputs so split_multi_turn_rows can derive ground_truth
    - Store run_id in input_metadata.session_data
    """
    project = project_name or os.getenv("LS_PROJECT", "ep-langgraph-examples")
    try:
        adapter = LangSmithAdapter()
        return adapter.get_evaluation_rows(project_name=project, limit=limit, include_tool_calls=True)
    except Exception as e:
        print(f"❌ LangSmithAdapter failed: {e}")
        return []


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI")
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[fetch_langsmith_traces_as_evaluation_rows()],
    completion_params=[
        {
            "model": "fireworks_ai/accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
        },
        {
            "max_tokens": 131000,
            "extra_body": {"reasoning_effort": "low"},
            "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
        },
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    preprocess_fn=split_multi_turn_rows,
    mode="all",
)
async def test_llm_judge_langsmith(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """LLM Judge evaluation over LangSmith-sourced rows, persisted locally by Eval Protocol.

    Mirrors quickstart/llm_judge.py, using Arena-Hard-Auto style pairwise judgment.
    """

    judge_name = "gemini-2.5-pro"

    if not rows:
        print("❌ No evaluation rows provided")
        return rows

    print(f"🔄 Processing {len(rows)} evaluation rows for LLM judging (LangSmith source)...")

    model_name = rows[0].input_metadata.completion_params.get("model", "unknown_model")

    judgments: List[Dict[str, Any]] = []

    judge_config = JUDGE_CONFIGS[judge_name]

    async with AsyncOpenAI(
        api_key=judge_config.get("api_key"), base_url=judge_config.get("base_url")
    ) as shared_client:
        for row in rows:
            result = await run_judgment_async(row, model_name, judge_name, shared_client)
            if result and result["games"][0] and result["games"][1]:
                judgments.append(result)

    if not judgments:
        print("❌ No valid judgments generated")
        return rows

    print(f"✅ Generated {len(judgments)} valid judgments")

    result = calculate_bootstrap_scores(judgments)
    if not result:
        print("❌ No valid scores extracted")
        return rows

    mean_score, lower_score, upper_score = result
    if mean_score == 0.0:
        print("❌ No valid scores extracted")
        return rows

    print("\n##### LLM Judge Results (90th percentile CI) #####")
    clean_model_name = model_name.split("/")[-1]
    print(f"{clean_model_name}: {mean_score:.1%} (CI: {lower_score:.1%} - {upper_score:.1%})")
    print("original: 50.0% (CI: 50.0% - 50.0%)")

    for row in rows:
        if row.evaluation_result:
            row.evaluation_result.score = mean_score
            row.evaluation_result.standard_error = (upper_score - lower_score) / (2 * 1.645)
        else:
            row.evaluation_result = EvaluateResult(
                score=mean_score,
                reason="Aggregated LLM judge score",
                metrics={
                    "summary": MetricResult(score=mean_score, reason="Aggregated over judgments"),
                },
            )

    return rows
