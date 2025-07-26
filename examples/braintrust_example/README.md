# Braintrust Example

A minimal example showing how to evaluate a Braintrust-style scorer end to end with Reward Kit and the Fireworks API.

## Quick Start

```bash
python -m eval_protocol.cli run --config-name simple_braintrust_eval
```

## Files

- `main.py` - Equality scorer wrapped as a Reward Kit reward function.
- `conf/simple_braintrust_eval.yaml` - Configuration using the `accounts/fireworks/models/qwen3-235b-a22b` model and the GSM8K dataset.
- `README.md` - This file.

## Data

This example reuses the **GSM8K** dataset from the math example.


## Output

Results are saved to `outputs/braintrust_eval/<timestamp>/eval_results.jsonl`.
