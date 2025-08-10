"""
Pytest test for coding code evaluation using the evaluation_test decorator.

This test demonstrates how to evaluate code correctness by executing Python code locally
and comparing the output against expected results in a pointwise manner.
"""

from typing import Any, Dict, List

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import default_single_turn_rollout_processor, evaluation_test
from eval_protocol.rewards.code_execution import execute_python_code, extract_code_blocks


def coding_dataset_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert entries from coding dataset to EvaluationRow objects.
    """
    return [
        EvaluationRow(
            messages=[Message(role="user", content=f"{row['prompt']} Input: {row['input']}")],
            ground_truth=row["expected_output"],
        )
        for row in data
    ]


@evaluation_test(
    input_dataset=["tests/pytest/data/basic_coding_dataset.jsonl"],
    dataset_adapter=coding_dataset_to_evaluation_row,
    model=["fireworks_ai/accounts/fireworks/models/kimi-k2-instruct"],
    rollout_input_params=[{"temperature": 0.0, "max_tokens": 4096}],
    threshold_of_success=0.8,
    rollout_processor=default_single_turn_rollout_processor,
    num_runs=1,
    mode="pointwise",
)
def test_coding_code_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Evaluation function that tests code correctness by executing it locally.

    This function:
    1. Extracts Python code from the assistant's response
    2. Executes the code locally with timeout=10
    3. Compares the output to ground_truth
    4. Returns a score of 1.0 if output matches, 0.0 otherwise

    Args:
        row: EvaluationRow containing the conversation messages and expected_output in ground_truth

    Returns:
        EvaluationRow with the evaluation result
    """
    # Check if we have an assistant response
    if len(row.messages) < 2 or row.messages[-1].role != "assistant":
        row.evaluation_result = EvaluateResult(score=0.0, reason="No assistant response found")
        return row

    assistant_content = row.messages[-1].content or ""
    expected_output = (row.ground_truth or "").strip()

    # Extract Python code blocks
    code_blocks = extract_code_blocks(assistant_content, language="python")
    if not code_blocks:
        row.evaluation_result = EvaluateResult(score=0.0, reason="No Python code block found")
        return row

    code = code_blocks[0]["code"]

    # Execute the code locally
    execution_result = execute_python_code(code, timeout=10)

    if not execution_result.get("success", False):
        error_msg = execution_result.get("error", "Code execution failed")
        row.evaluation_result = EvaluateResult(score=0.0, reason=f"Execution error: {error_msg}")
        return row

    # Compare output with expected
    actual_output = (execution_result.get("output", "") or "").strip()

    if actual_output == expected_output:
        row.evaluation_result = EvaluateResult(score=1.0, reason=f"✅ Output matches: '{actual_output}'")
    else:
        row.evaluation_result = EvaluateResult(
            score=0.0, reason=f"❌ Expected: '{expected_output}', Got: '{actual_output}'"
        )

    return row
