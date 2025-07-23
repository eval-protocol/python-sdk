"""Braintrust scorer wrapped for Reward Kit."""

from eval_protocol.adapters.braintrust import scorer_to_reward_fn
from eval_protocol.typed_interface import reward_function


def equality_scorer(input: str, output: str, expected: str) -> float:
    """Return ``1.0`` if ``output`` exactly matches ``expected``."""

    return 1.0 if output.strip() == expected.strip() else 0.0


_reward_fn = scorer_to_reward_fn(equality_scorer)


@reward_function
def evaluate(messages, ground_truth=None, **kwargs):
    """Reward Kit evaluate function calling the Braintrust scorer."""

    return _reward_fn(messages=messages, ground_truth=ground_truth)
