from typing import Any, Dict, List, Optional, Union

from reward_kit.models import EvaluateResult, Message, MetricResult
from reward_kit.typed_interface import reward_function


@reward_function
def simple_echo_reward(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ground_truth: Optional[str] = None,
    **kwargs: Any,
) -> EvaluateResult:
    """
    A simple reward function that returns a fixed score and echoes some input.
    """
    last_message_content = ""
    if messages:
        if isinstance(messages[-1], Message):
            last_message_content = messages[-1].content
        elif isinstance(messages[-1], dict) and "content" in messages[-1]:
            last_message_content = messages[-1].get("content", "")

    reason_str = f"Evaluated based on simple echo. Last message: '{last_message_content}'. Ground truth: '{ground_truth}'. Kwargs: {kwargs}"

    return EvaluateResult(
        score=0.75,
        reason=reason_str,
        is_score_valid=True,
        metrics={
            "echo_check": MetricResult(
                score=1.0,
                is_score_valid=True,
                reason="Echo check always passes for this dummy function.",
            )
        },
    )


@reward_function
def error_reward(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ground_truth: Optional[str] = None,
    **kwargs: Any,
) -> EvaluateResult:
    """
    A dummy reward function that always raises an error.
    """
    raise ValueError("This is a deliberate error from error_reward function.")
