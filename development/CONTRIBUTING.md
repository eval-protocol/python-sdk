# Reward Kit Developer Guide

This comprehensive guide is designed for developers who want to contribute to or modify the Reward Kit codebase. It covers all key aspects of development, from environment setup to testing and contributing new reward functions.

We are committed to fostering an open and welcoming environment. All contributors are expected to adhere to our [Code of Conduct](../../CODE_OF_CONDUCT.md).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/fireworks-ai/eval-protocol.git
cd eval-protocol

# Set up environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"  # Includes development dependencies

# Run tests
uv run pytest

# Type check and lint
make pre-commit
```

## Development Environment

### Setting Up Your Environment

1. **Clone the repository:**

```bash
git clone https://github.com/fireworks-ai/eval-protocol.git
cd eval-protocol
```

2. **Create and activate a virtual environment with uv:**

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
   **Important for LLMs and automated scripts:** After activating the virtual environment, always explicitly use executables from the `.venv/bin/` directory (e.g., `.venv/bin/pip`, `.venv/bin/pytest`, `.venv/bin/python`). This ensures commands run within the isolated environment, even if the shell's `PATH` isn't immediately updated or if context is lost between commands.

3. **Install the package in development mode:**

   Use `uv pip` from the virtual environment:
```bash
uv pip install -e .            # Basic installation
uv pip install -e ".[dev]"     # With development dependencies
```

### Authentication Setup for Development

For development and testing interactions with the Fireworks AI platform, you need to configure your Fireworks AI credentials. Reward Kit supports two methods:

### Local Development Configuration (.env.dev)

For a streamlined local development experience, especially when managing multiple environment variables, Reward Kit utilizes a `.env.dev` file in the root of the project. This file is used to load environment variables automatically when running the application locally.

**Setup:**

1.  **Create the `.env.dev` file:**
    Copy the example environment file to create your local development configuration:
    ```bash
    cp .env.example .env.dev
    ```
2.  **Populate `.env.dev`:**
    Open `.env.dev` and fill in the necessary environment variables, such as `FIREWORKS_API_KEY`, `FIREWORKS_ACCOUNT_ID`, and any other variables required for your development tasks (e.g., `E2B_API_KEY`).

    Example content for `.env.dev`:
    ```
    FIREWORKS_API_KEY="your_dev_fireworks_api_key"
    FIREWORKS_ACCOUNT_ID="abc"
    FIREWORKS_API_BASE="https://api.fireworks.ai"
    E2B_API_KEY="your_e2b_api_key"
    ```

**Important:**
*   The `.env.dev` file should **not** be committed to version control. It is already listed in the `.gitignore` file.
*   Variables set directly in your shell environment will take precedence over those defined in `.env.dev` if `python-dotenv` is configured to not override existing variables (which is the default behavior).

This file simplifies managing development-specific settings without needing to export them in every terminal session.

### Authentication Setup for Development (Continued)

For development and testing interactions with the Fireworks AI platform, you need to configure your Fireworks AI credentials. Reward Kit supports two methods:

**A. Environment Variables (Highest Priority)**

Set the following environment variables. For development, you might use specific development keys or a dedicated development account:

*   `FIREWORKS_API_KEY`: Your Fireworks AI API key.
    *   For development, you might use a specific dev key: `export FIREWORKS_API_KEY="your_dev_fireworks_api_key"`
*   `FIREWORKS_ACCOUNT_ID`: Your Fireworks AI Account ID. This is used to scope operations to your account.
    *   For development against a shared dev environment, this might be a common ID like `pyroworks-dev`: `export FIREWORKS_ACCOUNT_ID="pyroworks-dev"`
*   `FIREWORKS_API_BASE`: (Optional) If you need to target a non-production Fireworks API endpoint.
    *   For development: `export FIREWORKS_API_BASE="https://dev.api.fireworks.ai"`

Example for a typical development setup:
```bash
export FIREWORKS_API_KEY="your_development_api_key"
export FIREWORKS_ACCOUNT_ID="pyroworks-dev" # Or your specific dev account ID
export FIREWORKS_API_BASE="https://dev.api.fireworks.ai" # If targeting dev API
```

**B. Configuration File (Lower Priority)**

If environment variables are not set, Reward Kit will attempt to read credentials from `~/.fireworks/auth.ini`.

Create or ensure the file `~/.fireworks/auth.ini` exists with the following format:
```ini
[fireworks]
api_key = YOUR_FIREWORKS_API_KEY
account_id = YOUR_FIREWORKS_ACCOUNT_ID
```
Replace with your actual development credentials if using this method.

**Credential Sourcing Order:**
Reward Kit prioritizes credentials as follows:
1.  Environment Variables (`FIREWORKS_API_KEY`, `FIREWORKS_ACCOUNT_ID`)
2.  `~/.fireworks/auth.ini` configuration file

**Purpose of Credentials:**
*   `FIREWORKS_API_KEY`: Authenticates your requests to the Fireworks AI service.
*   `FIREWORKS_ACCOUNT_ID`: Identifies your account for operations like managing evaluators. It specifies *where* (under which account) an operation should occur.
*   `FIREWORKS_API_BASE`: Allows targeting different API environments (e.g., development, staging).

**Other Environment Variables:**
*   `E2B_API_KEY`: (Optional) If you are working on or testing features involving E2B code execution:
    `export E2B_API_KEY="your_e2b_api_key"`

## Code Structure

```
eval-protocol/
├── eval_protocol/                 # Main package source code
│   ├── __init__.py             # Package initialization
│   ├── reward_function.py      # Core reward function decorator
│   ├── models.py               # Data models and types
│   ├── typed_interface.py      # Type interfaces for reward functions
│   ├── evaluation.py           # Evaluation pipeline
│   ├── auth.py                 # Authentication utilities
│   ├── cli.py                  # Command line interface
│   ├── rewards/                # Out-of-the-box reward functions
│   │   ├── __init__.py         # Reward functions registry
│   │   ├── code_execution.py   # Code execution rewards
│   │   ├── function_calling.py # Function calling rewards
│   │   ├── json_schema.py      # JSON schema validation
│   │   ├── math.py             # Math evaluation
│   │   ├── format.py           # Format validation
│   │   ├── tag_count.py        # Tag counting
│   │   ├── accuracy.py         # Accuracy evaluation
│   │   ├── language_consistency.py # Language consistency
│   │   ├── reasoning_steps.py  # Reasoning steps evaluation
│   │   ├── length.py           # Response length evaluation
│   │   ├── repetition.py       # Repetition detection
│   │   ├── cpp_code.py         # C/C++ code evaluation
│   │   └── accuracy_length.py  # Combined accuracy and length evaluation
├── examples/                   # Example code and tutorials
│   ├── metrics/                # Example metric implementations
│   ├── samples/                # Sample data for evaluation
│   └── ...                     # Example scripts
├── tests/                      # Unit and integration tests
├── docs/                       # Documentation
└── ...                         # Project configuration files
```

## Reward Function Development

### Creating a New Reward Function

1. Create a new module in `eval_protocol/rewards/` if needed
2. Implement your reward function using the `@reward_function` decorator
3. Update `eval_protocol/rewards/__init__.py` to expose your function
4. Add unit tests in the `tests/` directory

Example structure:

```python
from typing import Dict, List, Any, Union, Optional

from ..typed_interface import reward_function
from ..models import Message, EvaluateResult, MetricResult

@reward_function
def my_reward_function(
    messages: Union[List[Dict[str, Any]], List[Message]],
    ground_truth: Optional[str] = None,
    **kwargs: Any
) -> EvaluateResult:
    """
    Evaluate responses based on custom criteria.

    Args:
        messages: List of conversation messages
        ground_truth: Expected correct answer
        **kwargs: Additional arguments

    Returns:
        EvaluateResult with evaluation score and metrics
    """
    # Your evaluation logic here
    # score = ...
    # reason = "..."
    # metric_score = ...
    # metric_success = ... (e.g., True if a condition is met)
    # metric_reason = "..."
    # For demonstration:
    score = 0.75
    reason = "The response met most criteria."
    metric_score = 0.8
    metric_success = True # Example: metric condition was met
    metric_reason = "Specific aspect evaluated positively."


    return EvaluateResult(
        score=score,
        reason=reason,
        is_score_valid=is_score_valid,
        metrics={
            "metric_name": MetricResult(
                score=metric_score,
                is_score_valid=metric_success,
                reason=metric_reason
            )
        }
    )
```

### Coding Style and Standards

To maintain code quality and consistency, please adhere to the following standards:

- **Formatting**:
    - Use `black` for code formatting. The maximum line length is 88 characters.
    - Use `isort` for organizing imports.
- **Linting**:
    - Adhere to `flake8` guidelines.
- **Type Hinting**:
    - Use type hints for all function parameters, return values, and variables where appropriate.
    - Run `mypy eval_protocol` to check for type errors.
- **Naming Conventions**:
    - `snake_case` for functions, methods, and variables.
    - `PascalCase` for classes and dataclasses.
    - `UPPER_SNAKE_CASE` for constants.
- **Imports**:
    - Group imports in the following order:
        1. Standard library imports
        2. Third-party library imports
        3. Local application/library specific imports (e.g., `from ..module import something`)
    - Separate each group with a blank line.
- **Docstrings**:
    - Write clear and concise docstrings for all public modules, classes, functions, and methods. Follow PEP 257 conventions.
    - For functions, explain the arguments, what the function does, and what it returns.
- **Error Handling**:
    - Use specific exception types rather than generic `Exception`.
    - Provide meaningful error messages.
- **Function Design**:
    - Keep functions and methods short and focused on a single responsibility.
    - Aim for readability and maintainability.
- **Testing**:
    - Write unit tests for all new public functions and significant private logic.
    - Ensure tests cover a variety of cases, including edge cases and expected failures.

### Pre-commit Hooks

To help enforce coding standards and catch issues early, we use pre-commit hooks. These hooks run automatically before each commit to check your code for issues like formatting, linting errors, and type errors.

**Installation and Setup:**

1.  **Install pre-commit**:
    If you installed development dependencies with `uv pip install -e ".[dev]"`, `pre-commit` should already be installed. If not, you can install it via uv:
    ```bash
    uv pip install pre-commit
    ```

2.  **Install the git hooks**:
    Navigate to the root of the repository and run:
    ```bash
    pre-commit install
    ```
    This will set up the pre-commit hooks to run automatically when you `git commit`.

**Usage:**

*   Once installed, pre-commit hooks will run on any changed files before you commit. If any hook fails, the commit will be aborted. You'll need to fix the reported issues and then `git add` the files again before attempting to commit.
*   Some hooks (like `black` and `isort`) may automatically fix issues. If they do, you'll still need to `git add` the modified files.
*   You can also run the hooks manually on all files at any time:
    ```bash
    pre-commit run --all-files
    ```
    This is useful for checking the entire codebase or after pulling new changes.

By using pre-commit hooks, we can ensure a consistent code style and catch many common errors before they even reach the CI pipeline, saving time and effort.

## Testing

### Running Tests

   Use uv to run tests with pytest:
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_evaluation.py

# Run specific test function
uv run pytest tests/test_file.py::test_function

# Run with coverage report
uv run pytest --cov=eval_protocol
```

We can focus on tests/ and examples/ folder for now since there are a lot of other repos

### Writing Tests

Create test files in the `tests/` directory following this pattern:

```python
import unittest
# Import the function you're testing
from eval_protocol.rewards.your_module import your_function

class TestYourFunction(unittest.TestCase):
    """Test your reward function."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        messages = [
            {"role": "user", "content": "Test question"},
            {"role": "assistant", "content": "Test response"}
        ]

        result = your_function(messages=messages)

        # Assert expectations
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
```

### Code Quality Tools

   Use uv to run code quality tools:
```bash
# Type checking
uv run mypy eval_protocol

# Linting
uv run flake8 eval_protocol

# Format code
uv run black eval_protocol
```

## Available Reward Functions

Reward Kit includes these out-of-the-box reward functions:

| Category | Reward Functions |
|----------|-----------------|
| Format | `format_reward` |
| Tag Count | `tag_count_reward` |
| Accuracy | `accuracy_reward` |
| Language | `language_consistency_reward` |
| Reasoning | `reasoning_steps_reward` |
| Length | `length_reward`, `cosine_length_reward` |
| Repetition | `repetition_penalty_reward` |
| Code Execution | `binary_code_reward`, `fractional_code_reward` |
| C/C++ Code | `ioi_cpp_code_reward`, `binary_cpp_code_reward` |
| Combined | `cosine_scaled_accuracy_length_reward` |
| Function Calling | `schema_jaccard_reward`, `llm_judge_reward`, `composite_function_call_reward` |
| JSON Schema | `json_schema_reward` |
| Math | `math_reward` |

## Running Examples

The examples folder contains sample code for using the Reward Kit:

```bash
# Run evaluation preview example
FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY \
FIREWORKS_API_BASE=https://dev.api.fireworks.ai \
uv run python examples/evaluation_preview_example.py

# Run deployment example
FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY \
FIREWORKS_API_BASE=https://dev.api.fireworks.ai \
uv run python examples/deploy_example.py
```


### Running Hydra-based Examples

Several example scripts, particularly those involving local evaluations (`local_eval.py`) and TRL integration (`trl_grpo_integration.py`) within directories like `examples/math_example/`, `examples/math_example_openr1/`, and `examples/tool_calling_example/`, have been refactored to use [Hydra](https://hydra.cc/) for configuration management.

**How to Run:**

1.  **Navigate to the repository root** if you aren't already there.

2.  **Run the script directly using uv:**
    Hydra will automatically pick up the configuration from the `conf` subdirectory relative to the script's location.
    ```bash
    # Example for math_example local_eval.py
    uv run python examples/math_example/local_eval.py

    # Example for math_example trl_grpo_integration.py
    uv run python examples/math_example/trl_grpo_integration.py
    ```

**Configuration:**

*   Configuration files are typically found in a `conf` subdirectory alongside the script (e.g., `examples/math_example/conf/local_eval_config.yaml`).
*   These YAML files define various parameters, including dataset paths, model names, and training arguments.

**Overriding Configuration:**

You can easily override any configuration parameter from the command line:

*   **Dataset Path:**
    ```bash
    uv run python examples/math_example/local_eval.py dataset_file_path=path/to/your/specific_dataset.jsonl
    ```
*   **Model Name (for TRL scripts):**
    ```bash
    uv run python examples/math_example/trl_grpo_integration.py model_name=mistralai/Mistral-7B-Instruct-v0.2
    ```
*   **GRPO Training Arguments (for TRL scripts):**
    Access nested parameters using dot notation.
    ```bash
    uv run python examples/math_example/trl_grpo_integration.py grpo.learning_rate=5e-5 grpo.num_train_epochs=3
    ```
*   **Multiple Overrides:**
    ```bash
    uv run python examples/tool_calling_example/trl_grpo_integration.py dataset_file_path=my_tools_data.jsonl model_name=google/gemma-7b grpo.per_device_train_batch_size=4
    ```

**Output Directory:**

Hydra manages output directories for each run. By default, outputs (logs, saved models, etc.) are saved to a timestamped directory structure like:
`outputs/YYYY-MM-DD/HH-MM-SS/` (relative to where the command is run, typically the repo root).
The exact base output path can also be configured within the YAML files (e.g., `hydra.run.dir`).

Refer to the specific `conf/*.yaml` file for each example to see all available configuration options.

## Command Line Interface


Use the Reward Kit CLI for common operations during development. Use uv to run the CLI commands:

```bash
# Preview an evaluator
uv run reward-kit preview --metrics-folders "word_count=./examples/metrics/word_count" \
--samples ./examples/samples/samples.jsonl

# Deploy an evaluator
uv run reward-kit deploy --id my-test-evaluator \
--metrics-folders "word_count=./examples/metrics/word_count" --force

# Deploy as local development server with tunnel (ideal for development/testing)
uv run reward-kit deploy --id test-local-serve-eval --target local-serve \
--function-ref examples.row_wise.dummy_example.dummy_rewards.simple_echo_reward --verbose --force
```

### Local Development Server

For local development and testing, you can use the `--target local-serve` option to run a reward function server locally with external tunnel access:

```bash
uv run eval-protocol deploy --id test-local-serve-eval --target local-serve \
--function-ref examples.row_wise.dummy_example.dummy_rewards.simple_echo_reward --verbose --force
```

**What this does:**
- Starts a local HTTP server on port 8001 serving your reward function
- Creates an external tunnel (using ngrok or serveo.net) to make the server publicly accessible
- Registers the tunnel URL with Fireworks AI for remote evaluation
- Returns to command prompt but keeps server running in background

**Important Notes:**
- The CLI returns control to you, but the server processes continue running in the background
- Check running processes: `ps aux | grep -E "(generic_server|ngrok)"`
- Test locally: `curl -X POST http://localhost:8001/evaluate -H "Content-Type: application/json" -d '{"messages": [{"role": "user", "content": "test"}]}'`
- Monitor server logs: `tail -f logs/eval-protocol-local/generic_server_*.log`
- Monitor tunnel logs: `tail -f logs/eval-protocol-local/ngrok_*.log`
- Stop when done: Kill the background processes manually

This is perfect for development, webhook testing, or making your reward function accessible to remote services without deploying to cloud infrastructure.

## Debugging Tips

### Authentication Issues

If you encounter authentication issues:

1.  **Check Credential Sources**:
    *   Verify that `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID` are correctly set as environment variables.
    *   If not using environment variables, ensure `~/.fireworks/auth.ini` exists, is correctly formatted, and contains the right `api_key` and `account_id` under the `[fireworks]` section.
    *   Remember the priority: environment variables override the `auth.ini` file.
2.  **Verify API Key Permissions**: Ensure the API key has the necessary permissions for the operations you are attempting.
3.  **Check Account ID**: Confirm that the `FIREWORKS_ACCOUNT_ID` is correct for the environment you are targeting (e.g., `pyroworks-dev` for the dev API, or your personal account ID).
4.  **API Base URL**: If using `FIREWORKS_API_BASE`, ensure it points to the correct API endpoint (e.g., `https://dev.api.fireworks.ai` for development).

You can use the following snippet to check what credentials the Reward Kit is resolving:
```python
from eval_protocol.auth import get_fireworks_api_key, get_fireworks_account_id

api_key = get_fireworks_api_key()
account_id = get_fireworks_account_id()

if api_key:
    print(f"Retrieved API Key (first 4, last 4 chars): {api_key[:4]}...{api_key[-4:]}")
else:
    print("API Key not found.")

if account_id:
    print(f"Retrieved Account ID: {account_id}")
else:
    print("Account ID not found.")
```

### API Debugging

For verbose API logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use the `--verbose` flag with CLI commands (from the venv):

```bash
.venv/bin/eval-protocol --verbose preview --metrics-folders "word_count=./examples/metrics/word_count" \
--samples ./examples/samples/samples.jsonl
```

## Building and Distribution

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
uv build

# Install locally from the built package

uv pip install dist/eval_protocol-*.whl
```

## Contributing Process

We welcome contributions to Reward Kit! Please follow these steps to contribute:

1.  **Find or Create an Issue**:
    *   Look for existing issues on the [GitHub Issues page](https://github.com/fireworks-ai/eval-protocol/issues) that you'd like to work on.
    *   If you have a new feature or bug fix, please create a new issue first to discuss it with the maintainers, unless it's a very minor change.

2.  **Fork and Clone the Repository**:
    *   Fork the repository to your own GitHub account.
    *   Clone your fork locally: `git clone https://github.com/YOUR_USERNAME/eval-protocol.git`
    *   Add the upstream repository: `git remote add upstream https://github.com/fireworks-ai/eval-protocol.git`

3.  **Create a New Branch**:
    *   Create a descriptive branch name for your feature or fix (e.g., `feat/add-new-reward-metric` or `fix/resolve-auth-bug`).
    *   `git checkout -b your-branch-name`

4.  **Make Your Changes**:
    *   Implement your changes, adhering to the [Coding Style and Standards](#coding-style-and-standards).
    *   Ensure your code is well-documented with docstrings.

5.  **Add Tests**:
    *   Write new tests for any new functionality.
    *   Ensure all tests pass by running `.venv/bin/pytest` (after activating the virtual environment).

6.  **Run Code Quality Checks**:

    *   Format your code: `uv run black reward_kit tests`
    *   Check linting: `uv run flake8 reward_kit tests`
    *   Check types: `uv run mypy reward_kit`
    *   Run pre-commit hooks: `pre-commit run --all-files`

7.  **Update Documentation**:
    *   If your changes affect user-facing features or APIs, update the relevant documentation in the `docs/` directory.
    *   Add examples if applicable.

8.  **Commit Your Changes**:
    *   Write clear and concise commit messages. Reference the issue number if applicable (e.g., `feat: Add awesome new metric (closes #123)`).
    *   `git commit -m "Your descriptive commit message"`

9.  **Push to Your Fork**:
    *   `git push origin your-branch-name`

10. **Submit a Pull Request (PR)**:
    *   Open a pull request from your branch to the `main` branch of the `fireworks-ai/eval-protocol` repository.
    *   Provide a clear title and a detailed description of your changes in the PR.
        *   Explain the "what" and "why" of your contribution.
        *   Link to the relevant issue(s) using keywords like `Closes #123` or `Fixes #456`.
    *   Ensure all CI checks pass on your PR.
    *   Be responsive to any feedback or questions from the maintainers during the code review process.

## Documentation

Update the documentation when adding new functionality:

1. Update relevant files in `docs/`
2. Add examples for new reward functions
3. Update the reward functions overview

## Troubleshooting

### Common Issues

1. **API Connection Errors**: Check your internet connection and API base URL
2. **Authentication Failures**: Verify your API key and account ID
3. **Import Errors**: Ensure you're using the correct virtual environment
4. **Deployment Failures**: Check API logs and your account permissions
5. **Type Errors**: Run `mypy eval_protocol` to identify typing issues

For more help, consult the [official documentation](https://github.com/fireworks-ai/eval-protocol) or file an issue on GitHub.
