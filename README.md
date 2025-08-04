# Eval Protocol (EP)

[![PyPI - Version](https://img.shields.io/pypi/v/eval-protocol)](https://pypi.org/project/eval-protocol/)

EP is an open protocol that standardizes how developers author evals for large
language model (LLM) applications.

## Quick Example

Here's a simple test function that checks if a model's response contains **bold** text formatting:

```python test_bold_format.py
from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest import default_single_turn_rollout_processor, evaluation_test

@evaluation_test(
    input_messages=[
        [
            Message(role="system", content="You are a helpful assistant. Use bold text to highlight important information."),
            Message(role="user", content="Explain why **evaluations** matter for building AI agents. Make it dramatic!"),
        ],
    ],
    model=["accounts/fireworks/models/llama-v3p1-8b-instruct"],
    rollout_processor=default_single_turn_rollout_processor,
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
