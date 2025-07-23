from typing import Any, Dict, List, Optional, Union

from eval_protocol.models import EvaluateResult, Message, MetricResult
from eval_protocol.typed_interface import reward_function


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


@reward_function
def length_based_reward(messages: List[Message], **kwargs: Any) -> EvaluateResult:
    """Reward based on the length of the assistant's last reply."""
    if not messages or messages[-1].role != "assistant":
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found.",
            metrics={
                "length": MetricResult(
                    score=0.0, success=False, reason="No assistant response."
                )
            },
        )

    assistant_message_content = messages[-1].content or ""
    length = len(assistant_message_content)
    score = min(1.0, length / 100.0)

    return EvaluateResult(
        score=score,
        reason=f"Assistant response length: {length} characters.",
        metrics={
            "length": MetricResult(
                score=score, success=length > 0, reason=f"Length: {length}"
            )
        },
    )
