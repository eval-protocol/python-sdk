#!/usr/bin/env python
"""
Example script demonstrating the E2B code execution reward function.

This script shows how to use the E2B code execution reward function
to evaluate code by running it in the E2B cloud sandbox.

Usage:
    python e2b_reward_example.py --api-key YOUR_E2B_API_KEY

You can get an E2B API key from https://e2b.dev/dashboard
"""

import argparse
import os

from reward_protocol.rewards.code_execution import e2b_code_execution_reward


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="E2B code execution reward example")
    parser.add_argument(
        "--api-key",
        help="E2B API key (or set E2B_API_KEY environment variable)",
    )
    args = parser.parse_args()

    # Use API key from arguments or environment variable
    api_key = args.api_key or os.environ.get("E2B_API_KEY")

    if not api_key:
        print(
            "E2B API key is required. Please provide it via --api-key or set the E2B_API_KEY environment variable."
        )
        return

    # Example conversation with a coding task
    messages = [
        {
            "role": "user",
            "content": "Write a Python function to calculate the factorial of a number.",
        },
        {
            "role": "assistant",
            "content": """Here's a Python function to calculate the factorial of a number:

```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# Test the function
print(factorial(5))  # Should output 120
```

This function uses recursion to calculate the factorial. For n = 5, it computes 5 * 4 * 3 * 2 * 1 = 120.""",
        },
    ]

    # Define expected output
    expected_output = "120"

    print("Running code in E2B sandbox...")

    # Evaluate the code using E2B
    result = e2b_code_execution_reward(
        messages=messages,
        expected_output=expected_output,
        language="python",
        api_key=api_key,
        timeout=10,
    )

    # Display results
    print(f"\nScore: {result.score:.2f}")
    print("\nMetrics:")

    for metric_name, metric in result.metrics.items():
        print(f"\n--- {metric_name} ---")
        print(metric.reason)


if __name__ == "__main__":
    main()
