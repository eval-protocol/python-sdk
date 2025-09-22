# Eval Protocol (EP)

[![PyPI - Version](https://img.shields.io/pypi/v/eval-protocol)](https://pypi.org/project/eval-protocol/)

**The open-source toolkit for building your internal model leaderboard.**

When you have multiple AI models to choose from—different versions, providers,
or configurations—how do you know which one is best for your use case?

## 🚀 Features

- **Custom Evaluations**: Write evaluations tailored to your specific business needs
- **Auto-Evaluation**: Stack-rank models using LLMs as judges with just model traces using out-of-the-box evaluators
- **RL Environments via MCP**: Build reinforcement learning environments using the Model Control Protocol (MCP) to simulate user interactions and advanced evaluation scenarios
- **Consistent Testing**: Test across various models and configurations with a unified framework
- **Resilient Runtime**: Automatic retries for unstable LLM APIs and concurrent execution for long-running evaluations
- **Rich Visualizations**: Built-in pivot tables and visualizations for result analysis
- **Data-Driven Decisions**: Make informed model deployment decisions based on comprehensive evaluation results

## Quick Examples

### Basic Model Comparison

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
    completion_params=[
        {"model": "fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct"},
        {"model": "openai/gpt-4"},
        {"model": "anthropic/claude-3-sonnet"}
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

### Using Datasets

Evaluate models on existing datasets:

```python
from eval_protocol.pytest import evaluation_test
from eval_protocol.adapters.huggingface import create_gsm8k_adapter

@evaluation_test(
    input_dataset=["development/gsm8k_sample.jsonl"],  # Local JSONL file
    dataset_adapter=create_gsm8k_adapter(),  # Adapter to convert data
    completion_params=[
        {"model": "openai/gpt-4"},
        {"model": "anthropic/claude-3-sonnet"}
    ],
    mode="pointwise"
)
def test_math_reasoning(row: EvaluationRow) -> EvaluationRow:
    # Your evaluation logic here
    return row
```


## 📚 Resources

- **[Documentation](https://evalprotocol.io)** - Complete guides and API reference
- **[Discord](https://discord.com/channels/1137072072808472616/1400975572405850155)** - Community discussions
- **[GitHub](https://github.com/eval-protocol/python-sdk)** - Source code and examples

## Installation

**This library requires Python >= 3.10.**

### Basic Installation

Install with pip:

```bash
pip install eval-protocol
```

### Recommended Installation with uv

For better dependency management and faster installs, we recommend using [uv](https://docs.astral.sh/uv/):

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install eval-protocol
uv add eval-protocol
```

### Optional Dependencies

Install with additional features:

```bash
# For Langfuse integration
pip install 'eval-protocol[langfuse]'

# For HuggingFace datasets
pip install 'eval-protocol[huggingface]'

# For all adapters
pip install 'eval-protocol[adapters]'

# For development
pip install 'eval-protocol[dev]'
```

## License

[MIT](LICENSE)
