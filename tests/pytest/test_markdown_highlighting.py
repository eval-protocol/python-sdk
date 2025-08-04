"""
Pytest test for markdown highlighting validation using the evaluation_test decorator.

This test demonstrates how to check if model responses contain the required number of highlighted sections.
"""

import re
from typing import Any, Dict, List, Optional

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import default_single_turn_rollout_processor, evaluation_test


def markdown_dataset_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert entries from markdown dataset to EvaluationRow objects.
    """
    return [
        EvaluationRow(messages=[Message(role="user", content=row["prompt"])], ground_truth=str(row["num_highlights"]))
        for row in data
    ]


@evaluation_test(
    input_dataset=["tests/pytest/data/markdown_dataset.jsonl"],
    dataset_adapter=markdown_dataset_to_evaluation_row,
    model=["accounts/fireworks/models/kimi-k2-instruct"],
    rollout_input_params=[{"temperature": 0.0, "max_tokens": 4096}],
    threshold_of_success=0.5,
    rollout_processor=default_single_turn_rollout_processor,
    num_runs=1,
    mode="pointwise",
)
def test_markdown_highlighting_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Evaluation function that checks if the model's response contains the required number of formatted sections.
    """

    assistant_response = row.messages[-1].content

    if not assistant_response:
        return EvaluateResult(score=0.0, reason="❌ No assistant response found")

    required_highlights = int(row.ground_truth)

    # Check if the response contains the required number of formatted sections
    # e.g. **bold** or *italic*

    actual_count = 0
    highlights = re.findall(r"\*[^\n\*]*\*", assistant_response)
    double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", assistant_response)

    for highlight in highlights:
        if highlight.strip("*").strip():
            actual_count += 1
    for highlight in double_highlights:
        if highlight.removeprefix("**").removesuffix("**").strip():
            actual_count += 1

    meets_requirement = actual_count >= required_highlights

    if meets_requirement:
        row.evaluation_result = EvaluateResult(
            score=1.0, reason=f"✅ Found {actual_count} highlighted sections (required: {required_highlights})"
        )
    else:
        row.evaluation_result = EvaluateResult(
            score=0.0, reason=f"❌ Only found {actual_count} highlighted sections (required: {required_highlights})"
        )
    return row
