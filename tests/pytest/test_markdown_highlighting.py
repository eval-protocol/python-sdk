"""
Pytest test for markdown highlighting validation using the evaluation_test decorator.

This test demonstrates how to check if model responses contain the required number of highlighted sections.
"""

import json
import re
from typing import Any, Dict, List, Optional

from eval_protocol.models import EvaluateResult, EvaluationRow, Message, InputMetadata, CompletionParams
from eval_protocol.pytest import evaluation_test, default_single_turn_rollout_processor, evaluate


def markdown_dataset_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert entries from markdown dataset to EvaluationRow objects.
    """    
    return [
        EvaluationRow(
            messages=[Message(role="user", content=row["prompt"])], 
            ground_truth=str(row["num_highlights"])
        )
        for row in data
    ]


def markdown_format_evaluate(messages: List[Message], ground_truth: Optional[str]=None, **kwargs) -> EvaluateResult:
    """
    Evaluation function that checks if the model's response contains the required number of formatted sections.
    """
    
    assistant_response = messages[-1].content
    
    if not assistant_response:
        return EvaluateResult(
            score=0.0,
            reason="❌ No assistant response found"
        )
    
    required_highlights = int(ground_truth)

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
        return EvaluateResult(
            score=1.0,
            reason=f"✅ Found {actual_count} highlighted sections (required: {required_highlights})"
        )
    else:
        return EvaluateResult(
            score=0.0,
            reason=f"❌ Only found {actual_count} highlighted sections (required: {required_highlights})"
        )


@evaluation_test(
    input_dataset=["tests/pytest/data/markdown_dataset.jsonl"],
    dataset_adapter=markdown_dataset_to_evaluation_row,
    model=["accounts/fireworks/models/llama-v3p1-8b-instruct"],
    input_params=[{"temperature": 0.0, "max_tokens": 4096}],  
    threshold_of_success=1.0,
    rollout_processor=default_single_turn_rollout_processor,
    num_runs=1
)
def test_markdown_highlighting_evaluation(input_dataset, input_params, model):
    """
    Test markdown highlighting validation using batch mode with evaluate().
    """
    return evaluate(input_dataset, markdown_format_evaluate) 