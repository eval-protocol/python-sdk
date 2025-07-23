"""
Tool Calling Evaluation Example

This example shows how to create a custom reward function that reuses
existing reward-kit functionality. Here we import and use the built-in
exact_tool_match_reward function to evaluate tool calling accuracy.
"""

from typing import Any, Dict, List, Optional, Union

from eval_protocol import EvaluateResult, reward_function
from eval_protocol.models import Message

# Import the existing reward function from reward-kit
from eval_protocol.rewards.function_calling import exact_tool_match_reward


@reward_function
def evaluate_tool_calling(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate tool calling accuracy using exact match comparison.

    This function demonstrates how to reuse existing reward-kit functions
    while allowing for future customization if needed.

    Args:
        messages: The conversation messages including tool calls
        ground_truth: Expected tool calls for comparison
        **kwargs: Additional parameters

    Returns:
        EvaluateResult with score and metrics
    """
    # For now, we directly use the built-in exact_tool_match_reward
    # In a real scenario, you might add preprocessing, custom logic, or
    # combine multiple reward functions here

    return exact_tool_match_reward(
        messages=messages, ground_truth=ground_truth, **kwargs
    )
