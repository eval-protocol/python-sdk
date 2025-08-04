# Math with Format and Length Example

This example extends the basic math evaluation by also checking that responses follow
`<think>...</think><answer>...</answer>` formatting and by rewarding concise answers.

## Quick Start
```bash
# Run the evaluation
python -m eval_protocol.cli run --config-name simple_math_format_length_eval
```

The framework will download a small GSM8K subset, generate answers using a reasoning
model, and evaluate them for numerical accuracy, format compliance, and length.
