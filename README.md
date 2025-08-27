# Eval Protocol (EP)

[![PyPI - Version](https://img.shields.io/pypi/v/eval-protocol)](https://pypi.org/project/eval-protocol/)

**The open-source toolkit for building your internal model leaderboard.**

When you have multiple AI models to choose from—different versions, providers, or configurations—how do you know which one is best for your use case?

## Quick Example

Compare models on a simple formatting task:

```python test_bold_format.py
from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import default_single_turn_rollout_processor, evaluation_test

@evaluation_test(
    input_messages=[
        [
            Message(role="system", content="Use bold text to highlight important information."),
            Message(role="user", content="Explain why evaluations matter for AI agents. Make it dramatic!"),
        ],
    ],
    model=[
        "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct",
        "openai/gpt-4",
        "anthropic/claude-3-sonnet"
    ],
    rollout_processor=default_single_turn_rollout_processor,
    mode="pointwise",
)
def test_bold_format(row: EvaluationRow) -> EvaluationRow:
    """Check if the model's response contains bold text."""
    assistant_response = row.messages[-1].content

    if assistant_response is None:
        row.evaluation_result = EvaluateResult(score=0.0, reason="No response")
        return row

    has_bold = "**" in str(assistant_response)
    score = 1.0 if has_bold else 0.0
    reason = "Contains bold text" if has_bold else "No bold text found"

    row.evaluation_result = EvaluateResult(score=score, reason=reason)
    return row
```

## 📚 Resources

- **[Documentation](https://evalprotocol.io)** - Complete guides and API reference
- **[Discord](https://discord.com/channels/1137072072808472616/1400975572405850155)** - Community discussions

## Installation

**This library requires Python >= 3.10.**

Install with pip:

```
pip install eval-protocol
```

## License

[MIT](LICENSE)
