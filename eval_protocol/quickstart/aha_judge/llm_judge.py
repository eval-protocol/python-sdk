"""
Default LLM judge for Eval Protocol. Inspired by Arena-Hard-Auto.
"""

from typing import Optional

from eval_protocol.models import EvaluationRow, EvaluateResult, MetricResult
from eval_protocol.adapters.base import BaseAdapter
from eval_protocol.quickstart.aha_judge.utils import (
    JUDGE_CONFIGS,
    LABEL_TO_SCORE,
    run_single_judgment,
)
from eval_protocol.utils.evaluation_row_utils import serialize_message

from openai import AsyncOpenAI


async def aha_judge(
    row: EvaluationRow, judge_name: str = "kimi-k2-instruct-0905", adapter: Optional[BaseAdapter] = None
) -> EvaluationRow:
    """
    LLM Judge evaluation using Arena-Hard-Auto style pairwise comparisons for a single row.

    Compares model response against ground truth using an LLM judge:
    1. Extracts the question from messages[:-1]
    2. Compares messages[-1] (new model response) vs ground_truth (baseline response)
    3. Runs two judgment rounds (A vs B, B vs A) to reduce position bias
    4. Returns individual scores for bootstrap aggregation

    Args:
        row: Single EvaluationRow object with messages, ground_truth, and tools
        judge_name: Name of the judge configuration to use
        adapter: Optional adapter to push scores back to (if provided)

    Returns:
        Same row with updated evaluation_result containing individual judgment scores
    """

    if not row.messages:
        return row

    judge_config = JUDGE_CONFIGS[judge_name]

    # Extract question and answers
    question_text = "\n".join([serialize_message(msg) for msg in row.messages[:-1]])
    model_a_answer = str(row.ground_truth)
    model_b_answer = serialize_message(row.messages[-1])

    async with AsyncOpenAI(api_key=judge_config.get("api_key"), base_url=judge_config.get("base_url")) as client:
        # Run two judgment rounds in sequence (A vs B, then B vs A)
        result1 = await run_single_judgment(
            question_text, model_a_answer, model_b_answer, row.tools, judge_config, client
        )
        result2 = await run_single_judgment(
            question_text, model_b_answer, model_a_answer, row.tools, judge_config, client
        )

    if not result1 or not result2 or not result1.get("score") or not result2.get("score"):
        # If either judgment failed, mark as invalid (don't include in distribution)
        final_score = 0.0
        reason = "Failed to get judgment scores"
        metrics = {}
        is_score_valid = False
    else:
        # Convert judgment scores to numerical scores
        game1_score = 1 - LABEL_TO_SCORE[result1["score"]]
        game2_score = LABEL_TO_SCORE[result2["score"]]
        final_score = (game1_score + game2_score) / 2

        reason = f"LLM Judge comparison: Round 1: {result1['score']}, Round 2: {result2['score']}"
        metrics = {
            "round1_judgment": MetricResult(score=game1_score, reason=result1["judgment"]),
            "round2_judgment": MetricResult(score=game2_score, reason=result2["judgment"]),
        }
        is_score_valid = True

    row.evaluation_result = EvaluateResult(
        score=final_score,
        reason=reason,
        metrics=metrics,
        is_score_valid=is_score_valid,
    )

    # Upload score to adapter if provided
    if adapter and row.evaluation_result and row.evaluation_result.is_score_valid:
        model_name = row.input_metadata.completion_params.get("model", "unknown_model")
        adapter.upload_score(row, model_name)

    return row
