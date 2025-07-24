"""
APPS Coding Evaluation Example

This example shows how to create a custom reward function for code generation
that reuses existing reward-kit functionality. Here we import and use the
built-in APPS coding evaluation to test code correctness against test cases.
"""

from typing import Any, Dict, List, Optional, Union

from eval_protocol import EvaluateResult, reward_function
from eval_protocol.models import Message

# Import the existing reward function from reward-kit
from eval_protocol.rewards.apps_coding_reward import evaluate_apps_solution


@reward_function
def evaluate(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[str] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate code generation accuracy using execution against test cases.

    This function demonstrates how to reuse existing reward-kit functions
    for code evaluation while allowing for future customization if needed.

    Args:
        messages: The conversation messages including the generated code
        ground_truth: JSON string with test cases (inputs/outputs)
        **kwargs: Additional parameters (like execution_timeout)

    Returns:
        EvaluateResult with score and metrics
    """
    # For now, we directly use the built-in APPS evaluation function
    # In a real scenario, you might add preprocessing, custom logic, or
    # combine multiple reward functions here

    return evaluate_apps_solution(messages=messages, ground_truth=ground_truth, **kwargs)
