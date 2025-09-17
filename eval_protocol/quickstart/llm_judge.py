"""
Default LLM judge for Eval Protocol. Inspired by Arena-Hard-Auto.
"""

from tqdm import tqdm
from typing import Optional

from eval_protocol.models import EvaluationRow
from eval_protocol.adapters.base import BaseAdapter
from eval_protocol.quickstart.utils import (
    JUDGE_CONFIGS,
    calculate_bootstrap_scores,
    run_judgment_async,
)
import asyncio
from openai import AsyncOpenAI


async def aha_judge(
    rows: list[EvaluationRow], judge_name: str = "gemini-2.5-pro", adapter: Optional[BaseAdapter] = None
) -> list[EvaluationRow]:
    """
    LLM Judge evaluation using Arena-Hard-Auto style pairwise comparisons.

    Compares model responses against ground truth using an LLM judge. For each row:
    1. Extracts the question from messages[:-1]
    2. Compares messages[-1] (new model response) vs ground_truth (baseline response)
    3. Runs two judgment rounds (A vs B, B vs A) to reduce position bias
    4. Calculates bootstrap scores across all comparisons
    5. Updates evaluation_result with final scores and confidence intervals

    Args:
        rows: List of EvaluationRow objects with messages, ground_truth, and tools
        judge_name: Name of the judge configuration to use
        adapter: Optional adapter to push scores back to (if provided)

    Returns:
        Same rows with updated evaluation_result containing scores and judgments
    """

    if not rows:
        print("❌ No evaluation rows provided")
        return rows

    print(f"🔄 Processing {len(rows)} evaluation rows for LLM judging...")

    model_name = rows[0].input_metadata.completion_params.get("model", "unknown_model")

    judgments = []
    max_concurrency = JUDGE_CONFIGS[judge_name]["max_concurrency"]

    judge_config = JUDGE_CONFIGS[judge_name]

    async with AsyncOpenAI(
        api_key=judge_config.get("api_key"), base_url=judge_config.get("base_url")
    ) as shared_client:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_judgment(row):
            async with semaphore:
                return await run_judgment_async(row, model_name, judge_name, shared_client)

        tasks = [run_judgment(row) for row in rows]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating judgments"):
            result = await coro
            if result and result["games"][0] and result["games"][1]:
                judgments.append(result)

    if not judgments:
        print("❌ No valid judgments generated")
        return rows

    print(f"✅ Generated {len(judgments)} valid judgments")

    # Calculate bootstrap scores
    result = calculate_bootstrap_scores(judgments)
    if not result:
        print("❌ No valid scores extracted")
        return rows

    mean_score, lower_score, upper_score = result

    # Print leaderboard
    print("\n##### LLM Judge Results (90th percentile CI) #####")

    clean_model_name = model_name.split("/")[-1]  # Clean model name

    print(f"{clean_model_name}: {mean_score:.1%} (CI: {lower_score:.1%} - {upper_score:.1%})")
    print("original: 50.0% (CI: 50.0% - 50.0%)")

    for row in rows:
        if row.evaluation_result:
            row.evaluation_result.score = mean_score

    # Push scores back to adapter if provided
    if adapter:
        adapter.upload_scores(rows, model_name, mean_score)

    return rows
