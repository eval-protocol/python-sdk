# Solver-Judge Evaluation Example (Countdown Task)

This example demonstrates a two-phase evaluation workflow using the Eval Protocol framework:

1. **Solver Phase**: Generate multiple candidate solutions for a problem using an LLM
2. **Judge Phase**: Use an LLM judge to select the best solution from the candidates

## The Countdown Task

The Countdown task is a classic arithmetic puzzle where:
- You're given a **target number** and a set of **available numbers**
- You must create an arithmetic expression that equals the target
- Each available number can only be used **exactly once**
- Only basic arithmetic operations are allowed: `+`, `-`, `*`, `/`
- The answer should be provided in `<answer>...</answer>` tags

Example:
- Target: 24
- Numbers: [1, 2, 3, 4]
- Solution: `<answer>(1 + 2 + 3) * 4</answer>` (equals 24)

## Files

- `main.py` - Main implementation with reward functions and evaluation tests
- `conf/solver_judge_eval.yaml` - Hydra configuration for running evaluations
- `tests/test_solver_judge.py` - Unit tests for the countdown reward functions

## Running the Example

### Option 1: Run with pytest (recommended)

```bash
# Run the basic solver test
pytest examples/solver_judge/main.py::test_solver_single -v

# Run the full solver-judge test
pytest examples/solver_judge/main.py::test_solver_judge -v
```

### Option 2: Run as standalone script

```bash
python examples/solver_judge/main.py
```

### Option 3: Run unit tests (no API key needed)

```bash
pytest examples/solver_judge/tests/test_solver_judge.py -v
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FIREWORKS_API_KEY` | API key for Fireworks AI | Required |
| `FIREWORKS_API_BASE` | API base URL | `https://api.fireworks.ai/inference/v1` |
| `JUDGE_MODEL` | Model for the judge LLM | `accounts/fireworks/models/llama-v3p1-70b-instruct` |
| `N_SOLUTIONS` | Number of candidate solutions | `3` |

## How It Works

### 1. Solver Phase

The solver generates N candidate solutions for each countdown problem:

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

### 2. Scoring Phase

Each solution is scored using `countdown_reward()`:
- **1.0**: Correct equation that evaluates to target
- **0.1**: Valid format but wrong result (partial credit)
- **0.0**: No answer tags or invalid format

### 3. Judge Phase

The LLM judge evaluates all solutions and selects the best one:

```python
judge_result = await run_judge(target, numbers, solutions)
selected_index = judge_result["selected_index"]
```

### 4. Final Scoring

The evaluation reports:
- **solver_accuracy**: Fraction of solutions that are fully correct (score=1.0)
- **judge_accuracy**: Whether the judge selected a correct solution
- **individual_scores**: Scores for each candidate solution

## Reward Function Details

The countdown reward function (`countdown_reward`) performs these checks:

1. **Format Check**: Does the response contain `<answer>...</answer>` tags?
2. **Validation Check**: Does the equation use exactly the available numbers?
3. **Correctness Check**: Does the equation evaluate to the target?

```python
from examples.solver_judge import countdown_reward

result = countdown_reward(
    solution_str="<answer>(1 + 2 + 3) * 4</answer>",
    ground_truth={"target": 24, "numbers": [1, 2, 3, 4]}
)
print(result.score)  # 1.0 if correct
print(result.metrics)  # Detailed breakdown
```

## Extending the Example

### Custom Dataset

```python
from eval_protocol.models import EvaluationRow, Message
from examples.solver_judge import get_countdown_prompt

MY_ROWS = [
    EvaluationRow(
        messages=[Message(role="user", content=get_countdown_prompt(100, [25, 50, 2, 4]))],
        ground_truth={"target": 100, "numbers": [25, 50, 2, 4]},
    ),
]

@evaluation_test(input_rows=[MY_ROWS], ...)
async def my_test(...):
    ...
```

### Programmatic Usage

```python
import asyncio
from examples.solver_judge import run_solver_judge_workflow

result = asyncio.run(run_solver_judge_workflow(
    target=24,
    numbers=[1, 2, 3, 4],
    n_solutions=5,
))

print(f"Solver Accuracy: {result['solver_accuracy']:.2%}")
print(f"Judge Accuracy: {result['judge_accuracy']:.2%}")
print(f"Selected: {result['selected_solution']}")
```

## Comparison with Original RLLM Implementation

| RLLM Concept | Eval Protocol Equivalent |
|--------------|-------------------------|
| `Trajectory` | `EvaluationRow` |
| `Step` | Individual `Message` in `EvaluationRow.messages` |
| `Episode` | Collection of `EvaluationRow` objects |
| `RolloutEngine` | `SingleTurnRolloutProcessor` + `AsyncOpenAI` |
| `RewardFunction` / `RewardOutput` | `@reward_function` decorator / `EvaluateResult` |
| `Workflow.run()` | `@evaluation_test` with `mode="groupwise"` |

## Example Output

```
Solver-Judge Workflow Demo (Countdown Task)
============================================================

Target: 24
Available Numbers: [1, 2, 3, 4]

Running workflow...

----------------------------------------
Results:
----------------------------------------

Solver Accuracy: 66.67%
Individual Scores: [1.0, 1.0, 0.1]

Judge Selected: Solution 1
Judge Accuracy: 100.00%

----------------------------------------
Solutions Generated:
----------------------------------------

✓ Solution 1 (score: 1.00) [SELECTED]:
Let me work through this step by step...
<answer>(1 + 2 + 3) * 4</answer>

✓ Solution 2 (score: 1.00):
I'll try different combinations...
<answer>4 * 3 * 2 * 1</answer>

~ Solution 3 (score: 0.10):
<answer>1 + 2 + 3 + 4</answer>
```
