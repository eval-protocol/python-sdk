# Eval Protocol (EP)

[![PyPI - Version](https://img.shields.io/pypi/v/eval-protocol)](https://pypi.org/project/eval-protocol/)

**Eval Protocol (EP) is the open-source standard and toolkit for practicing Eval-Driven Development.**

Building with AI is different. Traditional software is deterministic, but AI systems are probabilistic. How do you ship new features without causing silent regressions? How do you prove a new prompt is actually better?

The answer is a new engineering discipline: **Eval-Driven Development (EDD)**. It adapts the rigor of Test-Driven Development for the uncertain world of AI. With EDD, you define your AI's desired behavior as a suite of executable tests, creating a safety net that allows you to innovate with confidence.

EP provides a consistent way to write evals, store traces, and analyze results.

<p align="center">
	<img src="https://raw.githubusercontent.com/eval-protocol/python-sdk/refs/heads/main/assets/ui.png" alt="UI" />
	<br>
	<sub><b>Log Viewer: Monitor your evaluation rollouts in real time.</b></sub>
</p>

## Quick Example

Here's a simple test function that checks if a model's response contains **bold** text formatting:

```python test_bold_format.py
from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test

@evaluation_test(
    input_messages=[
        [
            Message(role="system", content="You are a helpful assistant. Use bold text to highlight important information."),
            Message(role="user", content="Explain why **evaluations** matter for building AI agents. Make it dramatic!"),
        ],
    ],
    completion_params=[{"model": "accounts/fireworks/models/llama-v3p1-8b-instruct"}],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
)
def test_bold_format(row: EvaluationRow) -> EvaluationRow:
    """
    Simple evaluation that checks if the model's response contains bold text.
    """

    assistant_response = row.messages[-1].content

    # Check if response contains **bold** text
    has_bold = "**" in assistant_response

    if has_bold:
        result = EvaluateResult(score=1.0, reason="✅ Response contains bold text")
    else:
        result = EvaluateResult(score=0.0, reason="❌ No bold text found")

    row.evaluation_result = result
    return row
```

## Documentation

See our [documentation](https://evalprotocol.io) for more details.

## Installation

**This library requires Python >= 3.10.**

Install with pip:

```
pip install eval-protocol
```

## License

[MIT](LICENSE)
