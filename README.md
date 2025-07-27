# Eval Protocol

**Eval Protocol: Author, reproduce, and evaluate reward functions seamlessly on Fireworks, TRL, and your own infrastructure.**

## Key Features

*   **Easy-to-use Decorator**: Define reward functions with a simple `@reward_function` decorator.
*   **Local Testing**: Quickly test your reward functions with sample data.
*   **Flexible Evaluation**: Evaluate model outputs based on single or multiple custom metrics.
*   **Seamless Deployment**: Deploy your reward functions to platforms like Fireworks AI.
*   **Comprehensive CLI**: Manage reward functions, preview evaluations (`eval-protocol preview`), deploy (`eval-protocol deploy`), and run complex evaluation pipelines (`eval-protocol run`).
*   **Simplified Dataset Integration**: Direct integration with HuggingFace datasets and on-the-fly format conversion.
*   **Extensible**: Designed to be adaptable for various LLM evaluation scenarios.

## Installation

```bash
pip install eval-protocol
```

### Optional TRL Extras

Install the additional dependencies required for running the TRL-based training
examples:

```bash
pip install "eval-protocol[trl]"
```

## Getting Started

Eval Protocol simplifies the creation and deployment of reward functions for evaluating AI model outputs.

### 1. Creating a Reward Function for Tool Calling

Eval Protocol allows you to define custom logic to evaluate model responses. Here's an example of how you might use the built-in `exact_tool_match_reward` for evaluating tool/function calls. This reward function checks if the model's generated tool calls exactly match the expected ones.

```python
# This is a conceptual example of how exact_tool_match_reward is defined and used.
# You would typically import it from eval_protocol.rewards.function_calling.
# For actual usage, you configure it in your YAML files for `eval-protocol run`.

from eval_protocol import reward_function
from eval_protocol.models import EvaluateResult, Message, MetricResult
from typing import List, Dict, Any, Optional, Union

# Definition of exact_tool_match_reward (simplified for brevity, see source for full details)
# from eval_protocol.rewards.function_calling import exact_tool_match_reward, eval_tool_call

@reward_function
def exact_tool_match_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> EvaluateResult:
    if not messages:
        return EvaluateResult(
            score=0.0, reason="No messages provided for evaluation.", metrics={}
        )

    generation_message_obj = messages[-1]
    generation_dict: Dict[str, Any]

    if isinstance(generation_message_obj, Message):
        generation_dict = {
            "role": generation_message_obj.role,
            "content": generation_message_obj.content,
        }
        if generation_message_obj.tool_calls:
            generation_dict["tool_calls"] = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in generation_message_obj.tool_calls
            ]
    elif isinstance(generation_message_obj, dict):
        generation_dict = generation_message_obj
    else:
        # Handle error for unexpected type
        return EvaluateResult(score=0.0, reason="Unexpected generation message type.", metrics={})

    if ground_truth is None:
        # Handle missing ground truth (e.g., score 0 if generation has tool calls, 1 if not)
        # This logic is simplified here.
        has_gen_tc = bool(generation_dict.get("tool_calls") or "<tool_call>" in generation_dict.get("content", ""))
        score = 0.0 if has_gen_tc else 1.0
        return EvaluateResult(score=score, reason="Ground truth not provided.", metrics={})

    # Ensure ground_truth is a dict (it might be a JSON string from some datasets)
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return EvaluateResult(score=0.0, reason="Ground truth string failed to parse.", metrics={})

    if not isinstance(ground_truth, dict):
         return EvaluateResult(score=0.0, reason="Ground truth is not a dictionary.", metrics={})

    # This simplified check compares generated tool calls with the expected ones.
    expected_tcs = ground_truth.get("tool_calls", [])
    generated_tcs = generation_dict.get("tool_calls", [])

    # This is a highly simplified check. The actual function is much more robust.
    is_match = (len(expected_tcs) == len(generated_tcs)) # Placeholder
    score = 1.0 if is_match else 0.0

    reason = f"Exact tool match evaluation score: {score}"
    return EvaluateResult(score=score, reason=reason, metrics={
        "tool_call_match": MetricResult(score=score, success=is_match, reason=reason)
    })

```
This example illustrates the structure. The actual `exact_tool_match_reward` in `eval_protocol.rewards.function_calling` handles complex parsing and comparison of tool calls.

### 2. Testing Your Reward Function with a Dataset

Effective testing of a reward function involves evaluating it against a representative dataset. The key is the **dataset/reward function pair**: your dataset should provide the necessary `ground_truth` information that your reward function expects.

**Crafting Your Dataset:**

1.  **Define `ground_truth`**: For each sample in your dataset, the `ground_truth_for_eval` (or a similarly named field specified in your dataset configuration) must contain the information your reward function needs to make a judgment.
    *   For `exact_tool_match_reward`, `ground_truth` should be a dictionary, often with a `tool_calls` key. This key would hold a list of expected tool calls, each specifying the `name` and `arguments` of the function call. Example:
        ```json
        {
          "role": "assistant",
          "tool_calls": [
            {
              "name": "get_weather",
              "arguments": {"location": "San Francisco, CA", "unit": "celsius"}
            }
          ]
        }
        ```
2.  **Format**: Datasets are typically JSONL files, where each line is a JSON object representing a sample. Each sample should include:
    *   `messages`: The input conversation history for the model.
    *   `tools` (optional, for tool calling): A list of available tools the model can use.
    *   `ground_truth_for_eval`: The expected output or data for the reward function (e.g., the structure shown above for tool calling).
    *   An `id` for tracking.

**Example Test Snippet (Conceptual):**

While `eval-protocol run` is the primary way to evaluate with datasets, here's a conceptual local test:

```python
from eval_protocol.rewards.function_calling import exact_tool_match_reward # Import the actual function
from eval_protocol.models import Message

# Sample 1: Correct tool call
test_messages_correct = [
    Message(role="user", content="What's the weather in SF?"),
    Message(role="assistant", tool_calls=[ # Model's generated tool call
        {"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "San Francisco, CA", "unit": "celsius"}'}}
    ])
]
ground_truth_correct = { # Expected tool call for the reward function
    "tool_calls": [
        {"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "celsius"}}
    ]
}

# Sample 2: Incorrect tool call
test_messages_incorrect = [
    Message(role="user", content="What's the weather in SF?"),
    Message(role="assistant", tool_calls=[
        {"id": "call_456", "type": "function", "function": {"name": "get_current_time", "arguments": '{}'}}
    ])
]
# Ground truth remains the same as we expect get_weather

# Test with the actual reward function
result_correct = exact_tool_match_reward(messages=test_messages_correct, ground_truth=ground_truth_correct)
print(f"Correct Call - Score: {result_correct.score}, Reason: {result_correct.reason}")

result_incorrect = exact_tool_match_reward(messages=test_messages_incorrect, ground_truth=ground_truth_correct)
print(f"Incorrect Call - Score: {result_incorrect.score}, Reason: {result_incorrect.reason}")
```
This local test helps verify the reward function's logic with specific inputs. For comprehensive evaluation, use `eval-protocol run` with a full dataset (see next section).

### 3. Running Local Evaluations with `eval-protocol run`

For comprehensive local evaluations, especially when working with datasets and complex configurations, the `eval-protocol run` command is the recommended tool. It leverages Hydra for configuration management, allowing you to define your evaluation pipeline (dataset, model, reward function, etc.) in YAML files.

**Example: Math Evaluation using `codeparrot/gsm8k`**

The `examples/math_example` demonstrates evaluating models on math word problems.

```bash
# Ensure you are in the repository root
# cd /path/to/eval-protocol

# Run evaluation with the math configuration
eval-protocol run --config-name run_math_eval.yaml --config-path examples/math_example/conf

# Override parameters directly from the command line:
eval-protocol run --config-name run_math_eval.yaml --config-path examples/math_example/conf \
  generation.model_name="accounts/fireworks/models/llama-v3p1-405b-instruct" \
  evaluation_params.limit_samples=10
```

**What this command does (typically):**
*   Loads the specified dataset (e.g., GSM8K directly from HuggingFace).
*   Applies any dataset-specific prompts or preprocessing defined in the configuration.
*   Generates model responses (e.g., using the Fireworks API or other configured providers).
*   Evaluates the generated responses using the specified reward function(s).
*   Saves detailed evaluation results to `<config_output_name>.jsonl` (e.g., `math_example_results.jsonl`) in a timestamped output directory (e.g., under `outputs/`).
*   Saves generated prompt/response pairs to `preview_input_output_pairs.jsonl` in the same output directory, suitable for inspection or re-evaluation with `eval-protocol preview`.

**Example: APPS Coding Evaluation**

The `examples/apps_coding_example` shows evaluation on code generation tasks using the `codeparrot/apps` dataset.

```bash
# Run evaluation with the APPS coding configuration
eval-protocol run --config-path examples/apps_coding_example/conf --config-name run_eval

# Example: Limit samples for a quick test
eval-protocol run --config-path examples/apps_coding_example/conf --config-name run_eval evaluation_params.limit_samples=2

# Example: Disable generation to test reward function on cached responses
eval-protocol run --config-path examples/apps_coding_example/conf --config-name run_eval generation.enabled=false
```

These examples showcase how `eval-protocol run` can be adapted for different tasks and datasets through configuration files.

For more details on this command, Hydra configuration, and advanced usage, see the [CLI Overview](docs/cli_reference/cli_overview.mdx) and [Hydra Configuration Guide](docs/developer_guide/hydra_configuration.mdx).

### Fireworks Authentication Setup (Required for Preview/Deploy with Fireworks)

To interact with the Fireworks AI platform for deploying and managing evaluations (including some preview scenarios that might use remote evaluators or if `eval-protocol run` uses a Fireworks-hosted model), Eval Protocol needs your Fireworks AI credentials. You can configure these in two ways:

**A. Environment Variables (Highest Priority)**

Set the following environment variables:

*   `FIREWORKS_API_KEY`: Your Fireworks AI API key. This is required for all interactions with the Fireworks API.
*   `FIREWORKS_ACCOUNT_ID`: Your Fireworks AI Account ID. This is often required for operations like creating or listing evaluators under your account.

```bash
export FIREWORKS_API_KEY="your_fireworks_api_key"
export FIREWORKS_ACCOUNT_ID="your_fireworks_account_id"
```

**B. Configuration File (Lower Priority)**

Alternatively, you can store your credentials in a configuration file located at `~/.fireworks/auth.ini`. If environment variables are not set, Eval Protocol will look for this file.

Create the file with the following format:

```ini
[fireworks]
api_key = YOUR_FIREWORKS_API_KEY
account_id = YOUR_FIREWORKS_ACCOUNT_ID
```

Replace `YOUR_FIREWORKS_API_KEY` and `YOUR_FIREWORKS_ACCOUNT_ID` with your actual credentials.

**Credential Sourcing Order:**

Eval Protocol will prioritize credentials in the following order:
1.  Environment Variables (`FIREWORKS_API_KEY`, `FIREWORKS_ACCOUNT_ID`)
2.  `~/.fireworks/auth.ini` configuration file

Ensure that the `auth.ini` file has appropriate permissions to protect your sensitive credentials.

The `FIREWORKS_API_KEY` is essential for authenticating your requests to the Fireworks AI service. The `FIREWORKS_ACCOUNT_ID` is used to identify your specific account context for operations that are account-specific, such as managing your evaluators. While the API key authenticates *who* you are, the account ID often specifies *where* (under which account) an operation should take place. Some Fireworks API endpoints may require both.

### 4. Evaluating with Sample Data (Preview)

Create a JSONL file with sample conversations to evaluate:

```json
{"messages": [{"role": "user", "content": "Tell me about AI"}, {"role": "assistant", "content": "AI refers to systems designed to mimic human intelligence."}]}
{"messages": [{"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is a subset of AI that focuses on building systems that can learn from data."}]}
```

Preview your evaluation using the CLI:

```bash
eval-protocol preview --metrics-folders "word_count=./path/to/metrics" --samples ./path/to/samples.jsonl
```

For example
```
eval-protocol preview --metrics-folders "word_count=examples/metrics/word_count" --samples development/CODING_DATASET.jsonl
```

### 5. Deploying Your Reward Function

Deploy your reward function to use in training workflows:

```bash
eval-protocol deploy --id my-evaluator --metrics-folders "word_count=./path/to/metrics" --force
```

#### Local Development Server

For local development and testing, you can deploy a reward function as a local server with external tunnel access:

```bash
# Deploy as local server with automatic tunnel (ngrok/serveo)
eval-protocol deploy --id test-local-serve-eval --target local-serve --function-ref examples.row_wise.dummy_example.dummy_rewards.simple_echo_reward --verbose --force
```

**What this does:**
- Starts a local HTTP server on port 8001 serving your reward function
- Creates an external tunnel (using ngrok or serveo.net) to make the server publicly accessible
- Registers the tunnel URL with Fireworks AI for remote evaluation
- Keeps the server running indefinitely in the background

**Key points:**
- The CLI returns to prompt after deployment, but the server continues running in background
- Check running processes: `ps aux | grep -E "(generic_server|ngrok)"`
- Test locally: `curl -X POST http://localhost:8001/evaluate -H "Content-Type: application/json" -d '{"messages": [{"role": "user", "content": "test"}]}'`
- Monitor logs: `tail -f logs/eval-protocol-local/generic_server_*.log`
- Stop server: Kill the background processes manually when done

This is ideal for development, testing webhook integrations, or accessing your reward function from remote services without full cloud deployment.

Or deploy programmatically:

```python
from eval_protocol.evaluation import create_evaluation

evaluator = create_evaluation(
    evaluator_id="my-evaluator",
    metric_folders=["word_count=./path/to/metrics"],
    display_name="My Word Count Evaluator",
    description="Evaluates responses based on word count",
    force=True  # Update if already exists
)
```

## Advanced Usage

### Multiple Metrics

Combine multiple metrics in a single reward function:

```python
from eval_protocol import reward_function
from eval_protocol.models import EvaluateResult, MetricResult, Message # Assuming models are here
from typing import List, Dict, Any, Optional

@reward_function
def combined_reward(
    messages: List[Dict[str, Any]], # Or List[Message]
    original_messages: Optional[List[Dict[str, Any]]] = None, # Or List[Message]
    **kwargs: Any
) -> EvaluateResult:
    """Evaluate with multiple metrics."""
    response = messages[-1].get("content", "")

    # Word count metric
    word_count = len(response.split())
    word_score = min(word_count / 100.0, 1.0)
    word_metric_success = word_count > 10

    # Specificity metric
    specificity_markers = ["specifically", "for example", "such as"]
    marker_count = sum(1 for marker in specificity_markers if marker.lower() in response.lower())
    specificity_score = min(marker_count / 2.0, 1.0)
    specificity_metric_success = marker_count > 0

    # Combined score with weighted components
    final_score = word_score * 0.3 + specificity_score * 0.7

    return EvaluateResult(
        score=final_score,
        reason=f"Combined score based on word count ({word_count}) and specificity markers ({marker_count})",
        metrics={
            "word_count": MetricResult(
                score=word_score,
                success=word_metric_success,
                reason=f"Word count: {word_count}"
            ),
            "specificity": MetricResult(
                score=specificity_score,
                success=specificity_metric_success,
                reason=f"Found {marker_count} specificity markers"
            )
        }
    )
```

### Custom Model Providers

Deploy your reward function with a specific model provider:

```python
# Deploy with a custom provider
my_function.deploy(
    name="my-evaluator-anthropic",
    description="My evaluator using Claude model",
    providers=[
        {
            "providerType": "anthropic",
            "modelId": "claude-3-sonnet-20240229"
        }
    ],
    force=True
)
```

## Dataset Integration

Eval Protocol provides seamless integration with popular datasets through a simplified configuration system:

### Direct HuggingFace Integration

Load datasets directly from HuggingFace Hub without manual preprocessing:

```bash
# Evaluate using GSM8K dataset with math-specific prompts
eval-protocol run --config-name run_math_eval.yaml --config-path examples/math_example/conf
```

### Derived Datasets

Create specialized dataset configurations that reference base datasets and apply transformations:

```yaml
# conf/dataset/gsm8k_math_prompts.yaml
defaults:
  - base_derived_dataset
  - _self_

base_dataset: "gsm8k"
system_prompt: "Solve the following math problem. Show your work clearly. Put the final numerical answer between <answer> and </answer> tags."
output_format: "evaluation_format"
derived_max_samples: 5
```

### Key Benefits

- **No Manual Conversion**: Datasets are converted to evaluation format on-the-fly
- **System Prompt Integration**: Prompts are part of dataset configuration, not evaluation logic
- **Flexible Column Mapping**: Automatically adapts different dataset formats
- **Reusable Configurations**: Base datasets can be extended for different use cases

See the [math example](examples/math_example/) for a complete demonstration of the dataset system.

## Detailed Documentation

For more comprehensive information, including API references, tutorials, and advanced guides, please see our [full documentation](docs/documentation_home.mdx).

## Examples

Check the `examples` directory for complete examples:

- `evaluation_preview_example.py`: How to preview an evaluator.
- `deploy_example.py`: How to deploy a reward function to Fireworks.
- `math_example/`: Demonstrates CLI-based evaluation (`eval-protocol run`) and TRL GRPO training for math problems (GSM8K dataset).
- `apps_coding_example/`: Shows CLI-based evaluation (`eval-protocol run`) for code generation tasks (APPS dataset).
 - `apps_coding_example/`: Shows CLI-based evaluation (`eval-protocol run`) for code generation tasks (APPS dataset).

The OpenEvals project provides a suite of evaluators that can be used directly within Eval Protocol. The helper `eval_protocol.integrations.openeval.adapt` converts any OpenEvals evaluator into a reward function returning an `EvaluateResult`.

```python
from openevals import exact_match
from eval_protocol.integrations.openeval import adapt

exact_match_reward = adapt(exact_match)
result = exact_match_reward(
    messages=[{"role": "assistant", "content": "hello"}],
    ground_truth="hello",
)
print(result.score)
```

The [deepeval](https://github.com/confident-ai/deepeval) project also offers a
variety of metrics. The helper `eval_protocol.integrations.deepeval.adapt_metric`
converts a deepeval metric instance into a reward function returning an
`EvaluateResult`.

```python
from deepeval.metrics import FaithfulnessMetric
from eval_protocol.integrations.deepeval import adapt_metric

faithfulness_reward = adapt_metric(FaithfulnessMetric())
result = faithfulness_reward(
    messages=[{"role": "assistant", "content": "hello"}],
    ground_truth="hello",
)
print(result.score)
```

The GEval metric family uses an LLM-as-a-judge to score outputs based on
custom criteria. You can construct a `GEval` metric and adapt it in the same
way:

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from eval_protocol.integrations.deepeval import adapt_metric

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the answer is factually correct",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
)

correctness_reward = adapt_metric(correctness_metric)
result = correctness_reward(
    messages=[{"role": "user", "content": "Who wrote 1984?"}, {"role": "assistant", "content": "George Orwell"}],
    ground_truth="George Orwell",
)
print(result.score)
```

## Command Line Interface

Eval Protocol includes a CLI for common operations:

```bash
# Show help
eval-protocol --help

# Preview an evaluator
eval-protocol preview --metrics-folders "metric=./path" --samples ./samples.jsonl

# Deploy an evaluator
eval-protocol deploy --id my-evaluator --metrics-folders "metric=./path" --force
```

## Community and Support

*   **GitHub Issues**: For bug reports and feature requests, please use [GitHub Issues](https://github.com/eval-protocol/python-sdk/issues).
*   **GitHub Discussions**: (If enabled) For general questions, ideas, and discussions.
*   Please also review our [Contributing Guidelines](development/CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## Development

### Type Checking

The codebase uses mypy for static type checking. To run type checking:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run mypy
mypy eval_protocol
```

Our CI pipeline enforces type checking, so please ensure your code passes mypy checks before submitting PRs.

### Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Code of Conduct

We are dedicated to providing a welcoming and inclusive experience for everyone. Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Eval Protocol is released under the Apache License 2.0.
