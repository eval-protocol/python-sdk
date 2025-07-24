#!/usr/bin/env python
"""
Example script demonstrating the E2B code execution reward function
with automatic extraction of the expected output.

This script shows how to use the E2B code execution reward function
without explicitly providing an expected output, allowing the function
to extract it automatically from the prompt.

Usage:
    python e2b_auto_extract_example.py --api-key YOUR_E2B_API_KEY

You can get an E2B API key from https://e2b.dev/dashboard
"""

import argparse
import os

from eval_protocol.rewards.code_execution import e2b_code_execution_reward


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="E2B code execution reward with auto-extraction example")
    parser.add_argument(
        "--api-key",
        help="E2B API key (or set E2B_API_KEY environment variable)",
    )
    args = parser.parse_args()

    # Use API key from arguments or environment variable
    api_key = args.api_key or os.environ.get("E2B_API_KEY")

    if not api_key:
        print("E2B API key is required. Please provide it via --api-key or set the E2B_API_KEY environment variable.")
        return

    # Example conversation with a coding task and expected output in the prompt
    messages = [
        {
            "role": "user",
            "content": """Write a Python function to find the sum of all numbers in a list.

Expected output: 15 (for the list [1, 2, 3, 4, 5])""",
        },
        {
            "role": "assistant",
            "content": """Here's a Python function to find the sum of all numbers in a list:

```python
def sum_list(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

# Test the function
result = sum_list([1, 2, 3, 4, 5])
print(result)
```

This function iterates through each number in the list and adds it to a running total.""",
        },
    ]

    print("Running code in E2B sandbox...")

    # Evaluate the code using E2B, letting it extract the expected output
    # ground_truth is None, so the function will attempt auto-extraction from messages.
    result = e2b_code_execution_reward(
        messages=messages,
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
