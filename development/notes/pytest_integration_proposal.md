# Proposal: Pytest-Based Evaluation Workflow

This document outlines a workflow for authoring evaluations as pytest tests and
aggregating their results for RL fine-tuning (RFT) or supervised fine-tuning
(SFT) using the unified `EvaluationRow` data model.

## 1. Evaluation Decorator

* Provide a new `evaluation_test` decorator in `eval_protocol.pytest_utils`.
* Accept optional arguments:
  * `samples`: path to a JSONL dataset or HF dataset name
  * `batch`: whether to call the wrapped reward function in batch mode
  * `max_samples`: limit the number of samples loaded
  * `env_url`: optional URL for an RL environment which is started and torn down
    around the test
* The decorator internally wraps the user function with `reward_function` to
  validate that it returns an `EvaluateResult`.
* During the test, data is loaded and each sample is evaluated. Any invalid score
  or exceptions will cause the test to fail.
* The decorator automatically constructs `EvaluationRow` objects containing:
  * `messages`: conversation history from the dataset
  * `ground_truth`: reference data from the dataset (top-level field)
  * `evaluation_result`: the result from the reward function
  * `input_metadata`: dataset and test configuration information

## 2. Unified Data Model Integration

The decorator leverages the unified `EvaluationRow` model which supports both:
* **Single-turn evaluations**: Standard row-wise batch evaluation
* **Trajectory evaluations**: Multi-step RL scenarios with step outputs

Key features:
* `ground_truth` is a top-level field in `EvaluationRow`, making reference data easily accessible
* `evaluation_result` contains the `EvaluateResult` with scores, metrics, and optional step outputs
* `is_trajectory_evaluation()` method automatically detects trajectory vs. single-turn scenarios
* Full serialization support for CI/CD integration

## 3. RL Environment Support

When `env_url` is provided, the decorator performs setup/teardown by launching
and polling the environment URL. Step data from the environment can be attached
via `EvaluateResult.step_outputs` as usual. This allows multi-turn RL scenarios
in regular pytest tests.

The resulting `EvaluationRow` objects will have:
* `is_trajectory_evaluation()` returning `True`
* `evaluation_result.step_outputs` containing per-step rewards and control plane data
* `evaluation_result.trajectory_info` with duration, steps, and termination reason

## 4. CI/CD Workflow

1. Developers write evaluation tests using `@evaluation_test`.
2. CI runs `pytest` to execute all evaluations whenever the system prompt or
   reward code changes.
3. A pytest plugin collects `EvaluationRow` objects from all tests and writes a
   single JSON file summarizing scores and metrics.
4. This file can be used to detect regressions (single aggregated score or
   per-test scores) and also serves as input for SFT/RFT pipelines.

The same serialized results together with the input datasets and reward code can
be uploaded for controlled rollouts or fine-tuning. This unifies local
validation, CI checks, and training data generation.

## 5. Example Usage

```python
import pytest
from eval_protocol.pytest_utils import evaluation_test
from eval_protocol.models import EvaluationRow, EvaluateResult, MetricResult

@evaluation_test(
    samples="path/to/dataset.jsonl",
    max_samples=100,
    env_url="http://localhost:8000/mcp/"  # Optional RL environment
)
def test_math_accuracy(messages, ground_truth, **kwargs):
    """Test mathematical accuracy with ground truth comparison."""
    
    # Access ground truth directly from the top-level field
    expected_answer = ground_truth
    
    # Your evaluation logic here
    score = calculate_accuracy(messages, expected_answer)
    
    return EvaluateResult(
        score=score,
        reason="Mathematical accuracy evaluation",
        metrics={
            "accuracy": MetricResult(score=score, reason="Direct comparison")
        }
    )

# The decorator automatically creates EvaluationRow objects:
# - messages: from dataset
# - ground_truth: from dataset (top-level)
# - evaluation_result: from your function
# - input_metadata: test configuration
```

## 6. Data Flow

1. **Dataset Loading**: JSONL files are loaded and parsed into `EvaluationRow` objects
2. **Test Execution**: Each row is processed by the decorated test function
3. **Result Collection**: `EvaluationRow` objects are collected with evaluation results
4. **Aggregation**: Results are aggregated for CI/CD reporting and training data generation
5. **Serialization**: Unified JSON format supports both local validation and remote training

This approach provides a clean, consistent interface for both traditional evaluation and RL scenarios while maintaining the flexibility to handle different types of ground truth data at the top level.
