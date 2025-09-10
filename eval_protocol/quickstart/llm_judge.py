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
from eval_protocol.quickstart.utils import pairwise_judgment

# Langfuse client setup
try:
    from langfuse import get_client  # pyright: ignore[reportPrivateImportUsage]

    LANGFUSE_AVAILABLE = True
    langfuse = get_client()
except ImportError:
    LANGFUSE_AVAILABLE = False
    langfuse = None


def fetch_langfuse_traces_as_evaluation_rows(
    hours_back: int = 168, tags: Optional[List[str]] = None
) -> List[EvaluationRow]:
    try:
        from eval_protocol.adapters.langfuse import create_langfuse_adapter

        if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
            raise ValueError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set")

        adapter = create_langfuse_adapter(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),  # pyright: ignore[reportArgumentType]
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),  # pyright: ignore[reportArgumentType]
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        now = datetime.now()
        from_timestamp = now - timedelta(hours=hours_back)

        return adapter.get_evaluation_rows(
            limit=20, from_timestamp=from_timestamp, to_timestamp=now, include_tool_calls=True, tags=tags
        )

    except Exception as e:
        print(f"❌ LangfuseAdapter failed: {e}")
        return []


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI")
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[fetch_langfuse_traces_as_evaluation_rows()],
    completion_params=[{"model": "gpt-4o"}],
    rollout_processor=SingleTurnRolloutProcessor(),
    split_multi_turn=True,
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

    if not rows:
        print("❌ No evaluation rows provided")
        return rows

    print(f"🔄 Processing {len(rows)} evaluation rows for LLM judging...")

    model_name = rows[0].input_metadata.completion_params.get("model", "unknown_model")

    # Generate judgments directly from rows
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor

    def run_judgment(row: EvaluationRow) -> Optional[Dict[str, Any]]:
        """Run pairwise judgment for a single evaluation row."""
        if not row.messages:
            return None

        # Extract question and answers
        question_text = "\n".join([f"{msg.role}: {msg.content}" for msg in row.messages[:-1]])
        model_a_answer = row.ground_truth  # Original response
        model_b_answer = row.messages[-1].content  # Comparison model response

        games = []

        # Round 1: A vs B (original vs comparison)
        result1 = pairwise_judgment(
            question_text=question_text,
            answer_a=model_a_answer,
            answer_b=model_b_answer,
        )
        games.append(result1)

        # Round 2: B vs A (comparison vs original)
        result2 = pairwise_judgment(
            question_text=question_text,
            answer_a=model_b_answer,
            answer_b=model_a_answer,
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
    max_workers = 64

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
    bootstrap_means = [
        battles.sample(frac=1.0, replace=True)["score"].mean() for _ in tqdm(range(100), desc="Bootstrap sampling")
    ]

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
        # This is hacky, but it's the only way to get the score into the evaluation result in our current pattern
        if row.evaluation_result:
            row.evaluation_result.score = mean_score
            # Standard error approximation from 90% CI: SE ≈ (upper - lower) / (2 × 1.645), but this is not quite right bc it assumes a normal distribution
            row.evaluation_result.standard_error = (upper_score - lower_score) / (2 * 1.645)

    return rows
