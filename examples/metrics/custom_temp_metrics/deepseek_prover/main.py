import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from examples.basic_reward import combined_reward

# Assuming combined_reward from basic_reward.py expects messages as List[Message]
# and returns an EvaluateResult (which can be accessed like a dict).
# The 'messages' here would be List[Dict[str, Any]] if this 'evaluate' is called
# before Pydantic conversion by a decorator. If combined_reward handles that, it's fine.


def evaluate(messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
    """
    Custom evaluation function for DeepSeek Prover, using combined_reward.
    Args:
        messages: List of conversation messages, typically List[Dict[str, Any]].
        **kwargs: Additional arguments passed to combined_reward.

    Returns:
        A dictionary with "score", "reason", and "metrics".
    """
    # Evaluate the messages using the combined_reward function
    # combined_reward is decorated, so it will handle conversion of messages
    # from List[Dict[str, Any]] to List[Message] if needed,
    # or expect List[Message] if this 'evaluate' function is already passing that.
    # Given basic_reward.py's combined_reward takes List[Message],
    # this implies 'messages' here should conform or be convertible.
    # For simplicity, assuming combined_reward handles List[Dict[str, Any]] input
    # due to its own @reward_function decorator.
    result = combined_reward(messages=messages, **kwargs)  # type: ignore

    # result is an EvaluateResult. Use attribute access.
    return {
        "score": result.score,
        "reason": result.reason,
        "metrics": result.metrics,
    }
