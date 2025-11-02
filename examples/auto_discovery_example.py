"""
Auto Test Discovery Example

This example demonstrates that @evaluation_test decorated functions
are automatically discoverable by pytest, regardless of naming.

Run with:
    pytest examples/auto_discovery_example.py -v
"""

from eval_protocol.models import EvaluationRow, EvaluateResult
from eval_protocol.pytest import evaluation_test


# Example 1: Function without 'test_' prefix
# This will be automatically registered as 'test_math_evaluation'
@evaluation_test(
    input_rows=[[
        EvaluationRow(messages=[{"role": "user", "content": "What is 2+2?"}])
    ]]
)
async def math_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Evaluate math responses.
    
    Even though this function doesn't start with 'test_',
    pytest will discover it as 'test_math_evaluation'.
    """
    # Simple evaluation logic
    row.evaluation_result = EvaluateResult(
        score=1.0,
        reason="Evaluation completed"
    )
    return row


# Example 2: Function with proper naming
@evaluation_test(
    input_rows=[[
        EvaluationRow(messages=[{"role": "user", "content": "Hello!"}])
    ]]
)
async def test_greeting_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    This already follows pytest conventions.
    Will be discovered normally.
    """
    row.evaluation_result = EvaluateResult(
        score=1.0,
        reason="Greeting evaluation completed"
    )
    return row


# Example 3: Another function without 'test_' prefix
@evaluation_test(
    input_rows=[[
        EvaluationRow(messages=[{"role": "user", "content": "Write a function"}])
    ]]
)
async def coding_task_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Automatically registered as 'test_coding_task_evaluation'.
    """
    row.evaluation_result = EvaluateResult(
        score=1.0,
        reason="Coding task evaluated"
    )
    return row


if __name__ == "__main__":
    print("="*70)
    print("Auto Test Discovery Example")
    print("="*70)
    print()
    print("All functions decorated with @evaluation_test will be discovered")
    print("by pytest, regardless of their naming:")
    print()
    print("  • math_evaluation           → test_math_evaluation")
    print("  • test_greeting_evaluation  → test_greeting_evaluation")
    print("  • coding_task_evaluation    → test_coding_task_evaluation")
    print()
    print("Run pytest to see all tests:")
    print("  pytest examples/auto_discovery_example.py --collect-only")
    print()
    print("Run the tests:")
    print("  pytest examples/auto_discovery_example.py -v")
    print("="*70)

