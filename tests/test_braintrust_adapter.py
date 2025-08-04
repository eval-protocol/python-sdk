import pytest

from eval_protocol.adapters.braintrust import reward_fn_to_scorer, scorer_to_reward_fn
from eval_protocol.models import EvaluateResult, Message
from eval_protocol.typed_interface import reward_function


def simple_scorer(input, output, expected):
    return 1.0 if output == expected else 0.0


def test_scorer_to_reward_fn():
    reward_fn = scorer_to_reward_fn(simple_scorer)
    messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hi"),
    ]
    ground_truth = [Message(role="assistant", content="hi")]
    result = reward_fn(messages=messages, ground_truth=ground_truth)
    assert isinstance(result, EvaluateResult)
    assert result.score == 1.0


@reward_function
def my_reward(messages, ground_truth=None, **kwargs):
    expected = ground_truth[-1].content if ground_truth else ""
    score = 1.0 if messages[-1].content == expected else 0.0
    return EvaluateResult(score=score)


def test_reward_fn_to_scorer():
    scorer = reward_fn_to_scorer(my_reward)
    score = scorer("foo", "bar", "bar")
    assert score == 1.0
