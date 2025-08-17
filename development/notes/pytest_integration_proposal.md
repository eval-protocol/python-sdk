# Proposal: A Comprehensive Pytest-Based Evaluation Workflow

This document outlines an enhanced, comprehensive workflow for authoring, running, and managing evaluations as `pytest` tests. It aims to provide a unified and powerful experience for various evaluation scenarios, including single-turn evaluation, multi-turn RL rollouts, and rejection sampling.

## 1. High-Level Vision

The core idea is to leverage `pytest` as a powerful framework for defining and running evaluations. By using a dedicated `@evaluation_test` decorator, developers can parameterize evaluations across different models, datasets, and inference parameters. The framework will handle data loading, environment setup, result collection, and aggregation, allowing developers to focus on the core logic of their evaluations.

This approach provides a clean, consistent interface for both traditional evaluation and complex RL scenarios, unifying local validation, CI checks, and training data generation into a single, cohesive workflow.

## 2. Core Components

The new evaluation experience is built around a few key components:

*   **`@evaluation_test` decorator**: The central entry point for creating an evaluation test. It handles parameterization, execution control, and result aggregation.
*   **The Test Function**: The decorated function orchestrates the evaluation run. It receives parameterized inputs (like model name and dataset) and uses helper functions to perform rollouts or evaluations.
*   **`@reward_function` decorator**: A way to define reusable evaluation logic (e.g., a reward model) that can be applied to evaluation results.
*   **`eval_protocol` helpers**: A suite of utility functions, such as `eval_protocol.rollout()` for running interactions with an environment and `eval_protocol.evaluate()` for applying a reward function.
*   **`eval_protocol.init()`**: An initialization function to configure global settings like local recording directories.

## 3. The `@evaluation_test` Decorator

The `@evaluation_test` decorator is a powerful tool that transforms a standard Python function into a `pytest`-powered evaluation. It uses `pytest.mark.parametrize` behind the scenes to create multiple test runs based on the provided arguments.

### Parameters:

*   `input_dataset` (List[str]): A list of paths to JSONL datasets or Hugging Face dataset names. The framework handles loading the data.
*   `model` (List[str]): A list of model identifiers to be tested. Each model will trigger a separate parameterized test run.
*   `input_params` (List[Dict]): A list of dictionaries, where each dictionary specifies inference parameters (`temperature`, `max_tokens`, etc.).
*   `env_urls` (List[str], optional): A list of URLs for RL environments. If provided, the framework will manage the setup and teardown of these environments for the duration of the test.
*   `num_runs` (int, optional): The number of times to execute the test logic for each parameterized run. Defaults to 1.
*   `num_retries` (int, optional): The number of times to retry a failed run. Defaults to 0.
*   `aggregation_method` (str, optional): The method for aggregating scores from multiple runs or dataset samples (`"mean"`, `"max"`, `"min"`). This is crucial for CI checks.
*   `threshold_of_success` (float, optional): A score threshold. If the aggregated score is below this value, the `pytest` test will fail.
*   `version` (str, optional): A version string for the evaluation. For better reproducibility, this can be tied to a git commit hash. We are exploring using `versioneer` for automatic versioning.
*   `upload_to_fw` (bool, optional): If `True`, results will be uploaded to a central service (e.g., Fireworks). Defaults to `False`.

## 4. Decoupling Logic with Rollout Processors

To provide maximum flexibility and modularity, the evaluation workflow is split into two logical stages:

1.  **Rollout/Generation**: This stage is responsible for taking an input prompt, a model, and an environment and generating a result (e.g., a single response or a multi-step trajectory).
2.  **Evaluation**: This stage takes the result of the rollout and applies one or more reward functions to score it.

The `@evaluation_test` decorator manages this via the `rollout_processor` parameter. A "processor" is simply a callable function that encapsulates the logic for the rollout stage.

This design allows developers to:
*   **Reuse Logic**: Standard rollout procedures can be defined once and reused across many tests.
*   **Isolate Complexity**: Complex, custom rollout logic (e.g., with specific post-processing) can be isolated from the evaluation logic.
*   **Prepare for Scale**: A self-contained rollout processor is a unit of work that can be more easily offloaded to a remote, scalable batch inference service.

### `rollout_processor` Parameter

*   **Type**: `Callable`
*   **Default**: `ep.default_rollout_processor`

If the `rollout_processor` argument is not provided to the decorator, a default implementation is used. This default processor handles standard, single-turn request-response interactions, similar to calling a standard completions API. For any multi-turn or custom interaction logic, you provide your own function.

## 5. Workflow and Examples

First, initialize the evaluation environment:
```python
import eval_protocol as ep

# Configure where to save evaluation artifacts locally
ep.init(record_dir="./records")
```

### Example 1: Simple API-Style Evaluation (Default Rollout)

This is the most straightforward case. For simple prompt-response evaluations where no complex interaction is needed, you can omit the `rollout_processor`. The framework's `default_rollout_processor` will be used automatically to get a model's response.

```python
import pytest
from typing import List, Dict
from eval_protocol.testing import evaluation_test, reward_function
from eval_protocol.models import EvaluationRow, EvaluateResult, LMessage, Tool

@evaluation_test(
    input_dataset=["./simple_prompts.jsonl"],
    model=["kimi-k2-instruct"],
    aggregation_method="mean",
    threshold_of_success=1.0,
)
def test_simple_response(dataset: List[EvaluationRow]) -> List[EvaluationRow]:
    """
    This test uses the default rollout processor. The `dataset` that is
    passed in already contains the model's response. The test function's
    only job is to apply a final scoring logic.
    """
    return ep.evaluate(dataset, evaluate_if_response_is_not_empty)

@reward_function
def evaluate_if_response_is_not_empty(messages: List[LMessage], **kwargs) -> EvaluateResult:
    """A simple reward function to check for a non-empty response."""
    last_message = messages[-1].content if messages else ""
    score = 1.0 if last_message and last_message.strip() else 0.0
    return EvaluateResult(score=score, reason=f"Response was non-empty: {score == 1.0}")
```

### Example 2: Complex Benchmark with a Custom Rollout (e.g., Tau2)

More complex benchmarks like `tau2` often require specific, multi-step logic to simulate tool use or user turns based on the dataset. This is a perfect use case for a custom `rollout_processor`.

```python
def tau2_rollout_processor(row: EvaluationRow, model: str, input_params: Dict, **kwargs) -> List[EvaluationRow]:
    """
    A custom processor for the Tau2 benchmark. It simulates the specific
    multi-turn and tool-use evaluation flow required by the benchmark.
    """
    # In a real scenario, this would contain complex logic to:
    # 1. Take the initial user prompt from `row`.
    # 2. Call the model.
    # 3. If the model returns a tool call, check it against the `ground_truth`
    #    from the dataset and provide a simulated tool response.
    # 4. Call the model again with the tool response.
    # 5. Construct a final EvaluationRow with the full transcript.

    # The logic is encapsulated here, away from the test definition.
    processed_row = ep.default_rollout_processor(row, model, input_params)[0] # Simplified for example
    return [processed_row]

@evaluation_test(
    input_dataset=["hf:fireworks-ai/tau2_bench"],
    model=["claude-3-5-sonnet-20240620"],
    input_params=[{"temperature": 0, "max_tokens": 2048}],
    # Pass our custom Tau2 processor to the decorator
    rollout_processor=tau2_rollout_processor,
)
def test_tau2_benchmark(evaluation_rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """
    This test receives rows that have been fully processed by our
    `tau2_rollout_processor`. Its job is to apply the final scoring.
    """
    return ep.evaluate(evaluation_rows, evaluate_helpfulness)

# The `evaluate_helpfulness` reward function would be defined here...
```

### Example 3: RL Evaluation with a Custom Rollout Processor

For RL tasks that interact with a live environment, a `rollout_processor` is used to handle the interaction loop. This is cleaner than having the loop inside the test function.

```python
def frozen_lake_rollout_processor(row: EvaluationRow, model: str, input_params: Dict, env_urls: List[str]) -> List[EvaluationRow]:
    """
    A custom processor for a simple RL task like Frozen Lake. It performs
    a standard rollout against the environment and returns the trajectory.
    """
    env_url = env_urls[0] if env_urls else None
    # ep.rollout handles the core interaction loop with the game environment.
    trajectories = ep.rollout(row, model, input_params, env_url)
    return [t.to_evaluation_row() for t in trajectories]

@evaluation_test(
    input_dataset=["./frozen_lake_prompts.jsonl"],
    model=["my-frozen-lake-agent-v1"],
    env_urls=["http://localhost:8000/frozen_lake_env/"],
    rollout_processor=frozen_lake_rollout_processor,
    aggregation_method="mean",
    threshold_of_success=0.78, # Standard success rate for Frozen Lake
)
def test_frozen_lake_performance(evaluation_rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """
    Receives completed game trajectories from the Frozen Lake processor.
    The rewards are already part of the trajectory data, so we just
    return the rows for aggregation and assertion against the threshold.
    """
    # No additional evaluation is needed unless we want to add a meta-reward.
    return evaluation_rows
```
Here, all the complex rollout logic is neatly encapsulated in `frozen_lake_rollout_processor`, which can be tested independently and reused.

### Example 4: Rejection Sampling via a Custom Processor

This pattern is also powerful enough to handle sophisticated sampling strategies like Best-of-N.

```python
def best_of_n_processor(row: EvaluationRow, model: str, input_params: Dict, **kwargs) -> List[EvaluationRow]:
    """
    A processor that generates N candidates and selects the best one.
    """
    # The `ep.evaluate` helper can be designed to understand `n > 1`
    # and return multiple candidates per input row.
    candidate_rows = ep.evaluate([row], None, model, input_params)

    # Then, apply a reward function to score each candidate.
    scored_rows = ep.evaluate(candidate_rows, score_politeness)

    # Finally, select the best row.
    # This logic could be encapsulated in a helper, e.g., ep.select_best().
    best_row = select_best_by_group(scored_rows, score_key='politeness')

    return [best_row]

@evaluation_test(
    input_dataset=["./customer_service_prompts.jsonl"],
    model=["customer-service-bot-v3"],
    # Generate 4 candidate responses for each prompt
    input_params=[{"temperature": 0.8, "n": 4}],
    rollout_processor=best_of_n_processor,
)
def test_best_of_n_sampling(evaluation_rows: List[EvaluationRow]):
    """
    The test function receives only the final, "best" rows selected
    by the processor. The test can then perform final assertions.
    """
    # Assert that all returned rows have a politeness score, for example.
    for row in evaluation_rows:
        assert "politeness" in row.evaluation_result.metrics
    return evaluation_rows

@reward_function
def score_politeness(messages: List[LMessage], **kwargs) -> EvaluateResult:
    """... logic to score the politeness of the final response ..."""
    score = 1.0 # Placeholder
    return EvaluateResult(score=score, metrics={"politeness": score})
```

## 6. CI/CD and Data-for-Tuning Workflow

This `pytest`-based system is designed for seamless integration into CI/CD pipelines.

1.  **Execution**: On code changes (e.g., to a model, prompt, or evaluation logic), CI runs `pytest`.
2.  **Assertion**: The tests will pass or fail based on the `threshold_of_success` and `aggregation_method`, providing a clear signal for regressions.
3.  **Result Collection**: The framework collects all generated `EvaluationRow` objects, which contain the inputs, outputs, scores, and metadata. These are saved to a file in the directory specified by `ep.init(record_dir=...)`.
4.  **Downstream Use**: The collected data is in a standardized format, ready to be used for:
    *   **Detailed analysis**: Deeper inspection of model performance.
    *   **Fine-tuning**: The data can be used to create datasets for SFT or RFT. A CLI command could facilitate this:
        ```bash
        eval_protocol cli create rft_dataset --from-test=test_agent_performance --output=rft_data.jsonl
        ```

This unified workflow ensures that the same code that validates models during development and CI is used to generate high-quality data for improving them.
