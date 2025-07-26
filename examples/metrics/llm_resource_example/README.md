# LLM Judge Example

This example demonstrates how to use an LLM as a judge within the reward function decorator's resource management system. It shows the proper separation between environment setup (handled by the framework) and evaluation logic (user code).

## What It Does

1. **Resource Management**: Uses `@reward_function(resources={"llms": [...]})` to let the framework handle LLM deployment
2. **LLM Judge**: Evaluates math problem answers from the GSM8K dataset using an external LLM
3. **Error Separation**: Framework handles deployment errors separately from evaluation errors
4. **Automatic Cleanup**: Resources are automatically cleaned up after evaluation

## Key Features

- **On-demand deployment**: Uses `deployment_type="on-demand"` with automatic `llm.apply()` call
- **Framework-managed setup**: The decorator handles resource initialization, not user code
- **Proper error propagation**: Deployment failures are separate from evaluation failures
- **Resource injection**: LLM client is injected into `kwargs['resources']['llms'][0]`

## Setup

```bash
# Set your Fireworks API key
export FIREWORKS_API_KEY=your_api_key_here
export FIREWORKS_ACCOUNT_ID=pyroworks
```

## Running the Example

```bash
# Run evaluation using the configuration file
FIREWORKS_ACCOUNT_ID=pyroworks eval-protocol run --config-path conf --config-name simple_llm_judge_eval
```

## Configuration

The example includes a Hydra configuration file (`conf/simple_llm_judge_eval.yaml`) that:
- Loads 3 samples from the GSM8K math dataset
- Generates responses to math questions using Llama-3.1-8B-Instruct
- Uses the judge LLM to evaluate the quality of those math responses
- Uses the `evaluate` function from `main.py`
- Outputs results to timestamped directories

## How It Works

1. **Framework Setup**: `@reward_function` decorator calls `llm.apply()` for on-demand deployment
2. **Resource Injection**: Framework injects the deployed LLM into `kwargs['resources']['llms'][0]`
3. **Evaluation**: User code gets the LLM from resources and uses it to judge the math answers
4. **Automatic Cleanup**: Framework handles resource cleanup after evaluation

This demonstrates the proper pattern for using external resources in reward functions - let the framework handle the infrastructure while you focus on the evaluation logic.
