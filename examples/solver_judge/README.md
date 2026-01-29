# Solver-Judge Evaluation Example

This example demonstrates a two-phase evaluation workflow using the Eval Protocol framework:

1. **Solver Phase**: Generate multiple candidate solutions for a problem using an LLM
2. **Judge Phase**: Use an LLM judge to select the best solution from the candidates

## Overview

The Solver-Judge pattern is commonly used in:

- **Best-of-N sampling**: Generate N solutions and select the best one
- **Self-consistency**: Use majority voting or intelligent selection among multiple solutions
- **Verifier-guided generation**: Use a separate model to verify/score solutions
- **MCTS-style exploration**: Generate candidates and use value models for selection

## Files

- `main.py` - Main implementation with reward functions and evaluation tests
- `conf/solver_judge_eval.yaml` - Hydra configuration for running evaluations

## Running the Example

### Option 1: Run with pytest (recommended)

```bash
# Run the basic solver test
pytest examples/solver_judge/main.py::test_solver_single -v

# Run the full solver-judge test
pytest examples/solver_judge/main.py::test_solver_judge -v
```

### Option 2: Run with Hydra configuration

```bash
eval-protocol run --config-path examples/solver_judge/conf --config-name solver_judge_eval
```

### Option 3: Run as standalone script

```bash
python examples/solver_judge/main.py
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FIREWORKS_API_KEY` | API key for Fireworks AI | Required |
| `FIREWORKS_API_BASE` | API base URL | `https://api.fireworks.ai/inference/v1` |
| `JUDGE_MODEL` | Model for the judge LLM | `accounts/fireworks/models/llama-v3p1-70b-instruct` |
| `N_SOLUTIONS` | Number of candidate solutions | `3` |

### Customizing the Judge

You can customize the judge configuration by modifying `JUDGE_CONFIG` in `main.py`:

```python
JUDGE_CONFIG = {
    "model": "your-preferred-model",
    "temperature": 0.0,
    "max_tokens": 4096,
    "api_key": os.getenv("YOUR_API_KEY"),
    "base_url": "https://your-api-endpoint",
}
```

## How It Works

### 1. Solver Phase

The solver generates N candidate solutions for each problem:

```python
@evaluation_test(
    completion_params=[
        {"model": "...", "temperature": 0.3 + i * 0.2}
        for i in range(N_SOLUTIONS)
    ],
    mode="groupwise",  # Process all N solutions together
)
async def test_solver_judge(rows: List[EvaluationRow], **kwargs):
    # rows contains N solutions for the same problem
    ...
```

### 2. Judge Phase

The judge evaluates all solutions and selects the best one:

```python
judge_result = await run_judge(problem, solutions)
selected_index = judge_result["selected_index"]
```

### 3. Metrics

The evaluation reports:

- **solver_accuracy**: Fraction of individual solutions that are correct
- **judge_accuracy**: Whether the judge selected a correct solution
- **individual_scores**: Scores for each candidate solution

## Extending the Example

### Custom Problem Domain

1. Modify `parse_solver_answer()` to handle your answer format
2. Update `check_answer_correct()` with domain-specific validation
3. Customize `create_judge_prompt()` for your evaluation criteria

### Custom Dataset

Replace `DEMO_ROWS` with your own dataset:

```python
MY_ROWS = [
    EvaluationRow(
        messages=[Message(role="user", content="Your problem here")],
        ground_truth="expected_answer",
    ),
    # ... more rows
]

@evaluation_test(input_rows=[MY_ROWS], ...)
async def test_my_solver_judge(...):
    ...
```

### Using External Datasets

Load from JSONL files:

```python
@evaluation_test(
    input_dataset=["path/to/your/dataset.jsonl"],
    ...
)
async def test_solver_judge_external(...):
    ...
```

## Programmatic Usage

The workflow can also be used outside of pytest:

```python
import asyncio
from examples.solver_judge.main import run_solver_judge_workflow

result = asyncio.run(run_solver_judge_workflow(
    problem="What is 2 + 2?",
    ground_truth="4",
    n_solutions=5,
))

print(f"Solver Accuracy: {result['solver_accuracy']:.2%}")
print(f"Judge Accuracy: {result['judge_accuracy']:.2%}")
print(f"Selected: {result['selected_solution']}")
```

## Comparison with Original RLLM Implementation

This eval_protocol implementation provides equivalent functionality to the RLLM Solver-Judge workflow:

| RLLM Concept | Eval Protocol Equivalent |
|--------------|-------------------------|
| `Trajectory` | `EvaluationRow` |
| `Step` | Individual message in `EvaluationRow.messages` |
| `Episode` | Collection of `EvaluationRow` objects with shared `row_id` |
| `RolloutEngine` | `SingleTurnRolloutProcessor` + `AsyncOpenAI` |
| `RewardFunction` | `@reward_function` decorator |
| `Workflow.run()` | `@evaluation_test` with `mode="groupwise"` |

## Metrics Output

Example output from the evaluation:

```json
{
  "score": 1.0,
  "reason": "Judge selected solution 2 (1-indexed). Correct",
  "metrics": {
    "solver_accuracy": {
      "score": 0.67,
      "reason": "Solver accuracy: 66.67% (2/3 correct)"
    },
    "judge_accuracy": {
      "score": 1.0,
      "reason": "Judge selection correct"
    },
    "individual_scores": {
      "score": 0.67,
      "data": {"scores": [1.0, 1.0, 0.0]}
    }
  }
}
```
