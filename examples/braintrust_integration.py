from braintrust import Eval

from eval_protocol.adapters.braintrust import scorer_to_reward_fn


def equality_scorer(input: str, output: str, expected: str) -> float:
    return 1.0 if output == expected else 0.0


reward_fn = scorer_to_reward_fn(equality_scorer)


def hi_bot_task(name: str) -> str:
    """Simple placeholder task that echoes the user's name."""
    return "Hi " + name


Eval(
    "Reward Kit Braintrust Example",
    data=lambda: [
        {"input": "Foo", "expected": "Hi Foo"},
        {"input": "Bar", "expected": "Hello Bar"},
    ],
    task=hi_bot_task,
    scores=[reward_fn],
)
