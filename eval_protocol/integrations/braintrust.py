"""Adapters for integrating Eval Protocol with Braintrust scoring functions."""

from typing import Any, Callable, List, Optional, cast

from eval_protocol.models import EvaluateResult, Message
from eval_protocol.typed_interface import reward_function

# Type alias for Braintrust scoring functions
BraintrustScorer = Callable[[Any, Any, Any], float]


def scorer_to_reward_fn(
    scorer: BraintrustScorer,
    *,
    messages_to_input: Optional[Callable[[List[Message]], Any]] = None,
    ground_truth_to_expected: Optional[Callable[[List[Message]], Any]] = None,
) -> Callable[[List[Message], Optional[List[Message]]], EvaluateResult]:
    """Wrap a Braintrust scorer as an Eval Protocol reward function."""

    def reward_fn_core(
        messages: List[Message], ground_truth: Optional[List[Message]] = None, **kwargs: Any
    ) -> EvaluateResult:
        input_val = messages_to_input(messages) if messages_to_input else messages[0].content
        output_val = messages[-1].content
        expected_val = None
        if ground_truth:
            expected_val = (
                ground_truth_to_expected(ground_truth) if ground_truth_to_expected else ground_truth[-1].content
            )
        score = scorer(input_val, output_val, expected_val)
        return EvaluateResult(score=float(score))

    # Wrap with reward_function decorator while preserving precise callable type for type checker
    wrapped = reward_function(reward_fn_core)
    return cast(Callable[[List[Message], Optional[List[Message]]], EvaluateResult], wrapped)


def reward_fn_to_scorer(
    reward_fn: Callable[[List[Message], Optional[List[Message]]], EvaluateResult],
) -> BraintrustScorer:
    """Create a Braintrust-compatible scorer from an Eval Protocol reward function."""

    def scorer(input_val: Any, output: Any, expected: Any) -> float:
        messages = [
            Message(role="user", content=str(input_val)),
            Message(role="assistant", content=str(output)),
        ]
        ground_truth = None
        if expected is not None:
            ground_truth = [Message(role="assistant", content=str(expected))]
        result = reward_fn(messages, ground_truth)
        return float(result.score)

    return scorer
