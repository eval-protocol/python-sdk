"""
Default LLM judge for Eval Protocol. Inspired by Arena-Hard-Auto.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

import pytest

from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor
from eval_protocol.quickstart.utils import pairwise_judgment, split_multi_turn_rows, serialize_message
from eval_protocol.adapters.langfuse import create_langfuse_adapter

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

JUDGE_CONFIGS = {
    "gpt-4.1": {
        "model": "gpt-4.1",
        "temperature": 0.0,
        "max_tokens": 16000,
        "max_concurrency": 64,
    },
    "gemini-2.5-pro": {
        "model": "gemini-2.5-pro",
        "temperature": 1.0,
        "max_tokens": 32000,
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "max_concurrency": 32,
    },
}


def fetch_langfuse_traces_as_evaluation_rows(
    limit: int = 100,
    tags: Optional[List[str]] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    hours_back: Optional[int] = None,
    include_tool_calls: bool = True,
) -> List[EvaluationRow]:
    try:
        adapter = create_langfuse_adapter()

        return adapter.get_evaluation_rows(
            limit=limit,
            tags=tags,
            user_id=user_id,
            session_id=session_id,
            hours_back=hours_back,
            include_tool_calls=include_tool_calls,
        )

    except Exception as e:
        print(f"❌ LangfuseAdapter failed: {e}")
        return []


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI")
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[fetch_langfuse_traces_as_evaluation_rows()],
    completion_params=[
        {"model": "gpt-5"},
        {
            # "max_tokens": 131000,
            # "extra_body": {"reasoning_effort": "low"},
            "model": "fireworks_ai/accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
        },
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    preprocess_fn=split_multi_turn_rows,
    mode="all",
)
async def test_llm_judge(rows: list[EvaluationRow]) -> list[EvaluationRow]:
    """
    Simplified LLM Judge for Arena-Hard-Auto style pairwise comparisons.

    Each row contains:
    - messages[:-1]: Question/prompt (conversation context)
    - messages[-1]: Model B's answer (comparison model response)
    - ground_truth: Model A's answer (original assistant response)
    """

    judge_name = "gemini-2.5-pro"  # Edit to which judge you'd like to use. Configs at top of file.

    if not rows:
        print("❌ No evaluation rows provided")
        return rows

    print(f"🔄 Processing {len(rows)} evaluation rows for LLM judging...")

    model_name = rows[0].input_metadata.completion_params.get("model", "unknown_model")

    def run_judgment(row: EvaluationRow) -> Optional[Dict[str, Any]]:
        """Run pairwise judgment for a single evaluation row."""
        if not row.messages:
            return None

        question_text = "\n".join([serialize_message(msg) for msg in row.messages[:-1]])
        model_a_answer = row.ground_truth
        model_b_answer = serialize_message(row.messages[-1])

        games = []

        # Round 1: A vs B (original vs comparison)
        result1 = pairwise_judgment(
            question_text=question_text,
            answer_a=model_a_answer,
            answer_b=model_b_answer,
            tools=row.tools,
            judge_config=JUDGE_CONFIGS[judge_name],
        )
        games.append(result1)

        # Round 2: B vs A (comparison vs original)
        result2 = pairwise_judgment(
            question_text=question_text,
            answer_a=model_b_answer,
            answer_b=model_a_answer,
            tools=row.tools,
            judge_config=JUDGE_CONFIGS[judge_name],
        )
        games.append(result2)

        row.evaluation_result = EvaluateResult(
            score=0.0,
            reason=f"LLM Judge comparison: Round 1: {result1['score']}, Round 2: {result2['score']}"
            if result1 and result2
            else "Failed to get judgement scores",
            metrics={
                "round1_judgment": MetricResult(
                    score=0.0, reason=result1["judgment"] if result1 else "Failed to get judgment reason"
                ),
                "round2_judgment": MetricResult(
                    score=0.0, reason=result2["judgment"] if result2 else "Failed to get judgment reason"
                ),
            },
        )

        return {"model": model_name, "games": games}

    judgments = []
    max_concurrency = JUDGE_CONFIGS[judge_name]["max_concurrency"]

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = [executor.submit(run_judgment, row) for row in rows]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating judgments"):
            result = future.result()
            if result and result["games"][0] and result["games"][1]:
                judgments.append(result)

    if not judgments:
        print("❌ No valid judgments generated")
        return rows

    print(f"✅ Generated {len(judgments)} valid judgments")

    # Convert to scores for leaderboard
    label_to_score = {
        "A>B": [1],
        "A>>B": [1] * 3,
        "A=B": [0.5],
        "A<<B": [0] * 3,
        "A<B": [0],
        "B>A": [0],
        "B>>A": [0] * 3,
        "B=A": [0.5],
        "B<<A": [1] * 3,
        "B<A": [1],
    }

    # Extract scores from judgments
    scores_data = []
    for judgment in judgments:
        game1, game2 = judgment["games"]
        if game1 and game2 and game1.get("score") and game2.get("score"):
            # Convert judgment scores to numerical scores
            scores = label_to_score[game2["score"]] + [1 - s for s in label_to_score[game1["score"]]]
            for score in scores:
                scores_data.append(score)

    if not scores_data:
        print("❌ No valid scores extracted")
        return rows

    # Create DataFrame (single column of scores)
    battles = pd.DataFrame({"score": scores_data})

    # Bootstrap sampling for calculating relative performance to original model at fixed 50%
    bootstrap_means = [battles.sample(frac=1.0, replace=True)["score"].mean() for _ in range(100)]

    # Calculate final scores
    bootstraps = pd.Series(bootstrap_means)
    mean_score = bootstraps.mean()
    lower_score = bootstraps.quantile(0.05)
    upper_score = bootstraps.quantile(0.95)

    # Print leaderboard
    print("\n##### LLM Judge Results (90th percentile CI) #####")

    clean_model_name = model_name.split("/")[-1]  # Clean model name

    print(f"{clean_model_name}: {mean_score:.1%} (CI: {lower_score:.1%} - {upper_score:.1%})")
    print("original: 50.0% (CI: 50.0% - 50.0%)")

    for row in rows:
        if row.evaluation_result:
            row.evaluation_result.score = mean_score
            row.evaluation_result.standard_error = (upper_score - lower_score) / (
                2 * 1.645
            )  # Standard error approximation from 90% CI

    return rows
