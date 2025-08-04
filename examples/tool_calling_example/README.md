# Tool Calling Example

A simplified example showing how to evaluate function/tool calling capabilities using eval-protocol.

## Quick Start

```bash
# Run the evaluation
python -m eval_protocol.cli run --config-name simple_tool_calling_eval
```

That's it! The framework automatically:
- Detects the local `conf/` directory
- Adds the current directory to Python path
- Loads the dataset and reward function
- Evaluates tool calling conversations and produces real scores

## What This Example Does

This example evaluates tool calling capabilities by:

1. **Loading conversations** from `dataset.jsonl` (user queries with expected tool calls)
2. **Comparing tool calls** using exact match scoring against ground truth
3. **Producing results** with scores (0.0 to 1.0) indicating tool calling accuracy

## Files

- `main.py` - Custom reward function that reuses built-in tool calling evaluation
- `conf/simple_tool_calling_eval.yaml` - Simplified configuration (no complex inheritance)
- `dataset.jsonl` - Sample tool calling conversations with ground truth
- `README.md` - This file

## Configuration

The `simple_tool_calling_eval.yaml` config is self-contained and includes:
- Dataset loading from `dataset.jsonl`
- Evaluation mode (no generation needed - using existing conversations)
- Automatic mapping of ground truth for evaluation
- 3 sample limit for quick testing (remove `limit_samples` to run all samples)

## Data Format

Each example in `dataset.jsonl` contains:
- `messages`: Conversation with user query and assistant tool calls
- `tools`: Available function definitions
- `ground_truth`: Expected assistant response with correct tool calls

## Output

Results are saved to `outputs/tool_calling_eval/[timestamp]/eval_results.jsonl` with:
- Exact match scores for tool calling accuracy (1.0 = perfect match, 0.0 = no match)
- Score distribution and statistics
- Detailed evaluation metrics

## About Tool Calling Evaluation

This example demonstrates how to create custom reward functions that reuse existing eval-protocol functionality. The `main.py` file imports and wraps the built-in `exact_tool_match_reward` function, showing the recommended pattern for:

- **Reusing existing functions**: Import from `eval_protocol.rewards.function_calling`
- **Adding customization**: Easy to extend with preprocessing or custom logic
- **Maintaining simplicity**: Keep the core evaluation logic while allowing flexibility

The evaluation performs precise comparison of:
- Function names
- Function arguments
- Tool call structure

The reward function returns 1.0 for perfect matches and 0.0 for mismatches, making it ideal for evaluating function calling accuracy.

The original complex setup with multiple config files, custom processors, and manual inheritance has been simplified to work with a single command while maintaining full evaluation functionality.
