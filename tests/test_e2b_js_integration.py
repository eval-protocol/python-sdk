#!/usr/bin/env python
"""
Test script to verify E2B integration works end-to-end with JavaScript.

This script runs a simple JavaScript code example using the E2B code execution
reward function and verifies that it works correctly.
"""

import os

import pytest

from eval_protocol.models import EvaluateResult
from eval_protocol.rewards.code_execution import _HAS_E2B, e2b_code_execution_reward
from tests.conftest import skip_e2b


@skip_e2b
def test_e2b_javascript_integration():
    """Test that E2B integration works correctly for JavaScript code."""
    # Verify API key is available
    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        pytest.skip("E2B_API_KEY environment variable is not set")

    # Simple test case with JavaScript code
    messages = [
        {
            "role": "user",
            "content": "Write a JavaScript function to check if a number is even.",
        },
        {
            "role": "assistant",
            "content": """Here's a JavaScript function to check if a number is even:

```javascript
function isEven(number) {
    return number % 2 === 0;
}

// Test the function
console.log(isEven(4));  // true
console.log(isEven(7));  // false
```

This function returns true if the number is even and false if it's odd.""",
        },
    ]

    expected_output = "true\nfalse"

    # Evaluate the code using E2B
    result = e2b_code_execution_reward(
        messages=messages,
        expected_output=expected_output,
        language="javascript",
        api_key=api_key,
        timeout=15,  # Increase timeout for first-time sandbox creation
    )

    # Verify the result
    assert isinstance(result, EvaluateResult)

    # If we get a sandbox error, consider the test successful if it contains expected errors
    if result.score == 0.0 and "execution_result" in result.metrics:
        error_msg = result.metrics["execution_result"].reason
        if "sandbox timeout" in error_msg or "sandbox was not found" in error_msg or "Invalid API key" in error_msg:
            pytest.skip(f"Skipping due to E2B connection issue: {error_msg}")
        # Also check for error in main result reason
        if result.reason and "Invalid API key" in result.reason:
            pytest.skip("Skipping due to invalid E2B API key")

    # Otherwise, it should be a successful result
    assert result.score == 1.0
    assert "execution_result" in result.metrics
    assert result.metrics["execution_result"].is_score_valid is True


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
