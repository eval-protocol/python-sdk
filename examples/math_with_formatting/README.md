# Math with Formatting Example

A simplified example showing how to evaluate mathematical reasoning with specific response formatting requirements using eval-protocol.

## Quick Start

```bash
# Run the evaluation
python -m eval_protocol.cli run --config-name simple_math_formatting_eval
```

That's it! The framework automatically:
- Detects the local `conf/` directory
- Adds the current directory to Python path
- Loads the dataset and reward function
- Evaluates math solutions considering both accuracy and formatting

## What This Example Does

This example evaluates mathematical reasoning with formatting requirements by:

1. **Loading GSM8K dataset** from HuggingFace (`openai/gsm8k`)
2. **Generating responses** to math problems using LLM with formatting-specific system prompt
3. **Extracting numerical answers** from model responses using built-in parsing
4. **Checking format compliance** for `<think>...</think><answer>...</answer>` structure
5. **Combining scores** by averaging accuracy and format compliance
6. **Producing results** with detailed metrics for both criteria

## Files

- `main.py` - Custom reward function combining math evaluation with format checking
- `conf/simple_math_formatting_eval.yaml` - Simplified configuration (loads from HuggingFace GSM8K)
- `README.md` - This file

## Configuration

The `simple_math_formatting_eval.yaml` config is self-contained and includes:
- Dataset loading from HuggingFace `openai/gsm8k` (test split)
- Response generation with formatting-specific system prompt
- Automatic column mapping (`question` → `user_query`, `answer` → `ground_truth_for_eval`)
- Configurable tolerance for numerical comparison (0.001)
- 5 samples from dataset, 2 processed for quick testing

## Data Source

This example uses the **GSM8K dataset** from HuggingFace (`openai/gsm8k`):
- **Questions**: Grade school math word problems
- **Answers**: Step-by-step solutions with final numerical answers
- **System Prompt**: Simply requests the final answer in `<answer>` tags. A reasoning model automatically includes `<think>` tags for its thought process.

The generated responses are evaluated for both numerical accuracy and strict format compliance.

## Output

Results are saved to `outputs/math_formatting_eval/[timestamp]/eval_results.jsonl` with:
- Combined scores averaging accuracy (0.0-1.0) and format compliance (0.0-1.0)
- Detailed metrics showing individual accuracy and format scores
- Score distribution and statistics

## About Multi-Criteria Evaluation

This example demonstrates how to create custom reward functions that combine multiple evaluation criteria. The `main.py` file shows the recommended pattern for:

- **Reusing existing functions**: Import `math_reward` from `eval_protocol.rewards.math`
- **Adding custom criteria**: Implement format checking with regex patterns
- **Combining metrics**: Average multiple scores for overall evaluation
- **Detailed reporting**: Provide separate metrics for each criterion

The evaluation performs:

**Accuracy Assessment:**
- HTML tag parsing (`<answer>` tags)
- LaTeX boxed expressions (`\\boxed{}`)
- Fraction conversion (`\\frac{a}{b}`)
- General number extraction from text
- Configurable numerical tolerance for comparison

**Format Assessment:**
- Regex pattern matching for `<think>...</think><answer>...</answer>` structure
- Binary scoring (1.0 for correct format, 0.0 for incorrect)

The final score is the average of accuracy and format scores, encouraging models to be both mathematically correct and follow instructions precisely.

The original complex setup with multiple config files, dataset inheritance, and manual dataset processing has been simplified to work with a single command while maintaining full multi-criteria evaluation functionality.
