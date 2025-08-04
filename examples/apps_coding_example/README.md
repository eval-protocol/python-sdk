# APPS Coding Example

A simplified example showing how to evaluate code generation models using eval-protocol on programming problems.

## Quick Start

```bash
# Run the evaluation
python -m eval_protocol.cli run --config-name simple_apps_eval
```

That's it! The framework automatically:
- Detects the local `conf/` directory
- Adds the current directory to Python path
- Loads the dataset and reward function
- Generates and evaluates code solutions with real execution against test cases

## What This Example Does

This example evaluates code generation by:

1. **Loading programming problems** from `development/CODING_DATASET.jsonl` (problems with test cases)
2. **Generating code solutions** using AI models to solve the programming problems
3. **Executing and testing code** against provided test cases to measure correctness
4. **Producing results** with scores (0.0 to 1.0) based on the percentage of test cases passed

## Files

- `main.py` - Custom reward function that reuses built-in APPS coding evaluation
- `conf/simple_apps_eval.yaml` - Simplified configuration (no complex inheritance)
- `development/CODING_DATASET.jsonl` - Sample programming problems with test cases
- `README.md` - This file

## Configuration

The `simple_apps_eval.yaml` config is self-contained and includes:
- Dataset loading from `development/CODING_DATASET.jsonl`
- Code generation using DeepSeek model
- Automatic execution against test cases
- 3 sample limit for quick testing (remove `limit_samples` to run all samples)

## Data Format

Each example in the dataset contains:
- `user_query`: Programming problem description
- `ground_truth_for_eval`: JSON with test cases (inputs/outputs) for validation

## Output

Results are saved to `outputs/apps_coding_eval/[timestamp]/eval_results.jsonl` with:
- Code correctness scores (1.0 = all tests pass, 0.0 = all tests fail)
- Pass rate for individual test cases
- Detailed execution metrics and error information

## About Code Evaluation

This example demonstrates how to create custom reward functions that reuse existing eval-protocol functionality. The `main.py` file imports and wraps the built-in `evaluate_apps_solution` function, showing the recommended pattern for:

- **Reusing existing functions**: Import from `eval_protocol.rewards.apps_coding_reward`
- **Adding customization**: Easy to extend with preprocessing or custom logic
- **Maintaining simplicity**: Keep the core evaluation logic while allowing flexibility

The evaluation performs:
- Python code extraction from AI responses
- Safe code execution with timeout protection
- Test case validation against expected outputs
- Detailed error reporting and debugging information

The reward function returns the percentage of test cases passed (0.0 to 1.0), making it ideal for measuring code generation accuracy on algorithmic problems.

The original complex setup with multiple config files and manual dataset preparation has been simplified to work with a single command while maintaining full evaluation functionality.
