from typing import Any, Dict, List, Optional, Union

from eval_protocol import reward_function
from eval_protocol.models import (  # Ensure MetricResult is imported
    EvaluateResult,
    Message,
    MetricResult,
)


@reward_function
def dummy_reward_func(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ground_truth: Optional[str] = None,
    **kwargs: Any,
) -> EvaluateResult:
    """A simple dummy reward function for testing."""
    score = 0.5
    if ground_truth == "success":
        score = 1.0
    elif ground_truth == "failure":
        score = 0.0

    reason = f"Dummy function processed {len(messages)} messages."
    if kwargs:
        reason += f" With kwargs: {kwargs}"

    return EvaluateResult(
        score=score,
        reason=reason,
        is_score_valid=True,
        metrics={
            "dummy_metric": MetricResult(
                score=0.75, is_score_valid=True, reason="A dummy metric"
            )
        },  # Correctly use MetricResult
    )


def not_a_reward_function():
    """A plain function for testing loading non-reward functions."""
    return "This is not an EvaluateResult object"


@reward_function
def dummy_reward_func_error(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ground_truth: Optional[str] = None,
    **kwargs: Any,
) -> EvaluateResult:
    """A dummy reward function that intentionally raises an error."""
    raise ValueError("Intentional error in dummy_reward_func_error")


@reward_function
def dummy_reward_func_invalid_return(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ground_truth: Optional[str] = None,
    **kwargs: Any,
) -> Dict:  # Intentionally wrong return type hint for testing server's handling
    """
    A dummy reward function that returns a dict instead of EvaluateResult.
    The @reward_function decorator might handle this, or the server's check will.
    """
    return {
        "score": 0.1,
        "reason": "This is a dict, not EvaluateResult",
        "is_score_valid": True,
        "metrics": {},
    }


def dummy_accepts_args_returns_string(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ground_truth: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """A dummy function that accepts standard reward func args but returns a string."""
    return f"Processed {len(messages)} messages, but returning a string."
