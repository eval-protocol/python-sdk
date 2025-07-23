#!/usr/bin/env python
"""
Example script demonstrating fallback to local execution when E2B is not available.

This script shows how to handle cases where the E2B API key is not available
by falling back to local code execution.

Usage:
    python e2b_fallback_example.py [--api-key YOUR_E2B_API_KEY]
"""

import argparse
import os

from eval_protocol.rewards.code_execution import (
    e2b_code_execution_reward,
    local_code_execution_reward,
)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="E2B fallback example")
    parser.add_argument("--api-key", help="E2B API key (optional)")
    args = parser.parse_args()

    # Example conversation with a coding task
    messages = [
        {
            "role": "user",
            "content": "Write a Python function to check if a number is prime.",
        },
        {
            "role": "assistant",
            "content": """Here's a Python function to check if a number is prime:

```python
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Test the function
print(is_prime(11))  # Should output True
print(is_prime(15))  # Should output False
```

This function implements an efficient algorithm to check if a number is prime. It first handles the base cases and then checks for divisibility using trial division.""",
        },
    ]

    # Define expected output
    expected_output = "True\nFalse"

    # Try to use E2B if API key is provided
    api_key = args.api_key or os.environ.get("E2B_API_KEY")

    if api_key:
        print("API key found. Running code in E2B sandbox...")

        result = e2b_code_execution_reward(
            messages=messages,
            expected_output=expected_output,
            language="python",
            api_key=api_key,
            timeout=10,
        )
    else:
        print("No API key found. Falling back to local execution...")

        result = local_code_execution_reward(
            messages=messages,
            expected_output=expected_output,
            language="python",
            timeout=5,
        )

    # Display results
    print(f"\nScore: {result.score:.2f}")
    print("\nMetrics:")

    for metric_name, metric in result.metrics.items():
        print(f"\n--- {metric_name} ---")
        print(metric.reason)


if __name__ == "__main__":
    main()
