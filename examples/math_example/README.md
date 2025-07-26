# Math Example

A simplified example showing how to evaluate mathematical reasoning using eval-protocol.

## Quick Start

```bash
# Run the evaluation
python -m eval_protocol.cli run --config-name simple_math_eval
```

That's it! The framework automatically:
- Detects the local `conf/` directory
- Adds the current directory to Python path
- Loads the dataset and reward function
- Evaluates math solutions and produces real scores

## What This Example Does

This example evaluates mathematical reasoning by:

1. **Loading GSM8K dataset** from HuggingFace (`openai/gsm8k`)
2. **Generating responses** to math problems using LLM with math-specific system prompt
3. **Extracting numerical answers** from model responses using built-in parsing
4. **Comparing answers** against ground truth with configurable tolerance
5. **Producing results** with scores (0.0 to 1.0) indicating mathematical accuracy

## Files

- `main.py` - Custom reward function that reuses built-in math evaluation
- `conf/simple_math_eval.yaml` - Simplified configuration (loads from HuggingFace GSM8K)
- `README.md` - This file

## Configuration

The `simple_math_eval.yaml` config is self-contained and includes:
- Dataset loading from HuggingFace `openai/gsm8k` (test split)
- Response generation with math-specific system prompt
- Automatic column mapping (`question` → `user_query`, `answer` → `ground_truth_for_eval`)
- Configurable tolerance for numerical comparison (0.001)
- 5 samples from dataset, 2 processed for quick testing

## Data Source

This example uses the **GSM8K dataset** from HuggingFace (`openai/gsm8k`):
- **Questions**: Grade school math word problems
- **Answers**: Step-by-step solutions with final numerical answers
- **System Prompt**: Guides model to include final answer in `<answer>72</answer>` tags

The generated responses are automatically evaluated for numerical accuracy using sophisticated answer extraction.

## Output

Results are saved to `outputs/math_eval/[timestamp]/eval_results.jsonl` with:
- Numerical accuracy scores (1.0 = correct answer, 0.0 = incorrect)
- Score distribution and statistics
- Detailed evaluation metrics including answer extraction

## About Math Evaluation

This example demonstrates how to create custom reward functions that reuse existing eval-protocol functionality. The `main.py` file imports and wraps the built-in `math_reward` function, showing the recommended pattern for:

- **Reusing existing functions**: Import from `eval_protocol.rewards.math`
- **Adding customization**: Easy to extend with preprocessing or custom logic
- **Maintaining simplicity**: Keep the core evaluation logic while allowing flexibility

The evaluation performs sophisticated numerical answer extraction including:
- HTML tag parsing (`<answer>` tags)
- LaTeX boxed expressions (`\\boxed{}`)
- Fraction conversion (`\\frac{a}{b}`)
- General number extraction from text
- Configurable numerical tolerance for comparison

The reward function returns 1.0 for correct numerical answers and 0.0 for incorrect ones, making it ideal for evaluating mathematical reasoning accuracy.

The original complex setup with multiple config files, dataset inheritance, and manual dataset processing has been simplified to work with a single command while maintaining full evaluation functionality.
