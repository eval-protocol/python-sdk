# N-Variant Generation to Batch Evaluation Guide

This guide explains how to use N-variant generation with batch reward functions in Eval Protocol.

## Overview

N-variant generation allows you to generate multiple response variants for each input sample, which can then be evaluated together using batch reward functions. This is useful for:

- Comparing different response strategies
- Finding the best response among multiple candidates
- Collecting diverse training data for RL
- A/B testing different model configurations

## Workflow

The complete workflow consists of three main steps:

1. **Generate N Variants**: Use the evaluation pipeline with `n > 1` to generate multiple responses per sample
2. **Transform to Batch Format**: Convert the N-variant JSONL output into batch evaluation format
3. **Run Batch Evaluation**: Use batch reward functions to evaluate all variants together

## Step 1: Generate N Variants

Configure your evaluation pipeline to generate multiple variants:

```yaml
# config.yaml
generation:
  enabled: true
  model_name: "your-model"
  n: 5  # Generate 5 variants per sample
  temperature: 0.8  # Higher temperature for diversity

reward:
  function_path: "your.pointwise.reward.function"

dataset:
  # Your dataset configuration

output:
  results_file: "n_variant_results.jsonl"
```

Run the evaluation:

```bash
eval-protocol run --config config.yaml
```

This produces a JSONL file where each line represents one variant with:
- `request_id`: Original sample ID (shared across variants)
- `response_id`: Variant index (0, 1, 2, ...)
- `id`: Unique variant ID (`{request_id}_v{response_id}`)
- Standard evaluation fields

## Step 2: Transform to Batch Format

Use the transformation utility to group variants by request:

```python
from eval_protocol.utils.batch_transformation import transform_n_variant_jsonl_to_batch_format

# Transform N-variant output to batch format
batch_data = transform_n_variant_jsonl_to_batch_format(
    input_file_path="n_variant_results.jsonl",
    output_file_path="batch_input.jsonl"
)
```

This creates batch evaluation entries with:
- `request_id`: Original sample ID
- `rollouts_messages`: List of conversation histories for all variants
- `num_variants`: Number of variants
- Other metadata from the original sample

## Step 3: Run Batch Evaluation

Create a batch reward function and run evaluation:

```python
from eval_protocol.typed_interface import reward_function
from eval_protocol.models import EvaluateResult, Message
from eval_protocol.utils.batch_evaluation import run_batch_evaluation

@reward_function(mode="batch")
def my_batch_reward(
    rollouts_messages: List[List[Message]],
    ground_truth_for_eval: str = None,
    **kwargs
) -> List[EvaluateResult]:
    """Compare all variants and return scores."""
    results = []

    # Process all variants together
    for i, rollout in enumerate(rollouts_messages):
        # Extract assistant response
        assistant_response = ""
        for msg in rollout:
            if msg.role == "assistant":
                assistant_response = msg.content
                break

        # Your scoring logic here
        score = calculate_score(assistant_response, ground_truth_for_eval)

        result = EvaluateResult(
            score=score,
            reason=f"Variant {i} analysis",
            is_score_valid=True
        )
        results.append(result)

    return results

# Run batch evaluation
batch_results = run_batch_evaluation(
    batch_jsonl_path="batch_input.jsonl",
    reward_function_path="my_module.my_batch_reward",
    output_path="batch_results.jsonl"
)
```

## Key Features

### Request/Response ID System

- **`request_id`**: Groups all variants from the same original sample
- **`response_id`**: Identifies individual variants within a request (0, 1, 2, ...)
- Enables easy grouping and comparison of variants

### Standalone Transformation Function

The transformation function is designed to be reusable:

```python
# Basic usage
transform_n_variant_jsonl_to_batch_format(
    input_file_path="variants.jsonl",
    output_file_path="batch.jsonl"
)

# Advanced usage with custom field names
transform_n_variant_jsonl_to_batch_format(
    input_file_path="variants.jsonl",
    output_file_path="batch.jsonl",
    request_id_field="original_sample_id",
    response_id_field="variant_num",
    messages_field="conversation_history"
)
```

### Batch Evaluation Utilities

The batch evaluation utility handles:
- Loading and validating batch reward functions
- Processing grouped variants
- Error handling for individual variants
- Structured output with original metadata

## Example: Complete Workflow

See `examples/n_variant_to_batch_example.py` for a complete working example that demonstrates:

1. Sample N-variant data creation
2. Transformation to batch format
3. Batch reward function implementation
4. Results analysis and comparison

## Configuration Options

### N-Variant Generation
- `generation.n`: Number of variants to generate (default: 1)
- `generation.temperature`: Sampling temperature for diversity
- Standard generation parameters apply to all variants

### Transformation
- `request_id_field`: Field containing the original sample ID
- `response_id_field`: Field containing the variant index
- `messages_field`: Field containing conversation messages
- `fallback_messages_fields`: Alternative fields to construct messages

### Batch Evaluation
- Reward function must use `@reward_function(mode="batch")`
- Input: `rollouts_messages: List[List[Message]]`
- Output: `List[EvaluateResult]`
- All variants are processed together for comparative scoring

## Testing

Comprehensive tests are available:
- `tests/test_n_variant_integration.py`: N-variant generation tests
- `tests/test_n_variant_batch_integration.py`: End-to-end batch evaluation tests

Run tests with:
```bash
pytest tests/test_n_variant_integration.py tests/test_n_variant_batch_integration.py -v
```
