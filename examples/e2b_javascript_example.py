#!/usr/bin/env python
"""
Example script demonstrating the E2B code execution reward function with JavaScript.

This script shows how to use the E2B code execution reward function
to evaluate JavaScript code by running it in the E2B cloud sandbox.

Usage:
    python e2b_javascript_example.py --api-key YOUR_E2B_API_KEY

You can get an E2B API key from https://e2b.dev/dashboard
"""

import argparse
import os

from eval_protocol.rewards.code_execution import e2b_code_execution_reward


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="E2B JavaScript execution reward example")
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

    # Example conversation with a JavaScript coding task
    messages = [
        {
            "role": "user",
            "content": "Write a JavaScript function to check if a string is a palindrome.",
        },
        {
            "role": "assistant",
            "content": """Here's a JavaScript function to check if a string is a palindrome:

```javascript
function isPalindrome(str) {
    // Remove non-alphanumeric characters and convert to lowercase
    const cleanStr = str.toLowerCase().replace(/[^a-z0-9]/g, '');

    // Compare with its reverse
    const reversedStr = cleanStr.split('').reverse().join('');

    return cleanStr === reversedStr;
}

// Test the function
console.log(isPalindrome("A man, a plan, a canal: Panama"));  // Should output true
console.log(isPalindrome("hello"));  // Should output false
```

This function removes any non-alphanumeric characters and converts the string to lowercase before checking if it reads the same forward and backward.""",
        },
    ]

    # Define expected output
    expected_output = "true\nfalse"

    print("Running JavaScript code in E2B sandbox...")

    # Evaluate the code using E2B
    result = e2b_code_execution_reward(
        messages=messages,
        expected_output=expected_output,
        language="javascript",
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
