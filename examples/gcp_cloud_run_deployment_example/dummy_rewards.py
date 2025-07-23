from typing import Any, Dict, List, Optional

from eval_protocol.models import EvaluateResult, MetricResult
from eval_protocol.reward_function import (  # Assuming @reward_function is available
    reward_function,
)


@reward_function
def hello_world_reward(
    messages: List[Dict[str, Any]], ground_truth: Optional[str] = None, **kwargs: Any
) -> EvaluateResult:
    """
    A simple dummy reward function that always returns a fixed score.
    """
    print(f"Dummy hello_world_reward called with messages: {messages}")
    print(f"Ground truth: {ground_truth}")
    print(f"Additional kwargs: {kwargs}")

    return EvaluateResult(
        score=0.75,
        reason="This is a dummy reward from hello_world_reward.",
        is_score_valid=True,
        metrics={
            "dummy_metric": MetricResult(
                score=1.0, is_score_valid=True, reason="Dummy metric always passes."
            )
        },
    )


if __name__ == "__main__":
    # Example of how it might be called (for local testing of the function itself)
    test_messages = [
        {"role": "user", "content": "Hello there!"},
        {"role": "assistant", "content": "General Kenobi!"},
    ]
    result = hello_world_reward(
        messages=test_messages,
        ground_truth="Some expected answer",
        extra_param="test_value",
    )
    print(result)
