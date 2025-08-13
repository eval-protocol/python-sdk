"""
Tests for code execution reward functions.
"""

import json  # Added import for json.loads

import pytest

from eval_protocol.models import EvaluateResult, Message  # Added for new tests
from eval_protocol.rewards.code_execution import (
    _HAS_E2B,
    compare_outputs,
    e2b_code_execution_reward,
    execute_code_with_e2b,
    execute_javascript_code,
    execute_python_code,
    extract_code_blocks,
    fractional_code_reward,  # Added for new tests
    local_code_execution_reward,
    string_similarity,
)


@pytest.mark.skipif(not _HAS_E2B, reason="E2B not installed")
class TestE2BCodeExecution:
    def test_e2b_reward_function_missing_e2b(self, monkeypatch):
        # Patch _HAS_E2B to False to simulate missing E2B package
        monkeypatch.setattr("eval_protocol.rewards.code_execution._HAS_E2B", False)

        prompt_message_dict = {
            "role": "user",
            "content": "Write a function to add two numbers",
        }
        assistant_message_dict = {
            "role": "assistant",
            "content": """Here's a function to add two numbers:

```python
def add(a, b):
    return a + b

print(add(2, 3))
```

This will output `5`.
""",
        }
        messages_arg = [prompt_message_dict, assistant_message_dict]
        ground_truth_arg = "5"  # This is the expected_output_str

        result = e2b_code_execution_reward(messages=messages_arg, ground_truth=ground_truth_arg, language="python")

        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        assert "E2B package not installed" in result.metrics["error"].reason
        # assert result['score'] == 0.0 # Use attribute access
        # assert "E2B package not installed" in result['metrics']['error']['reason'] # Use attribute access

        # Restore _HAS_E2B to its original value
        monkeypatch.setattr("eval_protocol.rewards.code_execution._HAS_E2B", _HAS_E2B)

    @pytest.mark.skipif(not _HAS_E2B, reason="E2B not installed")
    def test_execute_code_with_e2b_authentication(self, monkeypatch):
        """Test that authentication error is properly handled."""
        # Force E2B_API_KEY to None for this test
        monkeypatch.delenv("E2B_API_KEY", raising=False)

        code = "print('Hello, world!')"
        result = execute_code_with_e2b(code, language="python", api_key=None)

        assert result["success"] is False
        assert "API key is required" in result["error"]

    @pytest.mark.skipif(not _HAS_E2B, reason="E2B not installed")
    def test_e2b_reward_function_no_api_key(self, monkeypatch):
        """Test that missing API key is properly handled in the reward function."""
        # Ensure E2B_API_KEY is not set in environment
        monkeypatch.delenv("E2B_API_KEY", raising=False)

        prompt_message_dict = {
            "role": "user",
            "content": "Write a function to add two numbers",
        }
        assistant_message_dict = {
            "role": "assistant",
            "content": """```python
def add(a, b):
    return a + b

print(add(2, 3))
```""",
        }
        messages_arg = [prompt_message_dict, assistant_message_dict]
        ground_truth_arg = "5"  # This is the expected_output_str

        result = e2b_code_execution_reward(
            messages=messages_arg,
            ground_truth=ground_truth_arg,
            language="python",
            api_key=None,
        )

        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        assert "API key is required" in result.metrics["error"].reason
        # assert result['score'] == 0.0 # Use attribute access
        # assert "API key is required" in result['metrics']['error']['reason'] # Use attribute access


class TestExtractCodeBlocks:
    def test_extract_python_code(self):
        text = """Here's a simple Python function:

```python
def add(a, b):
    return a + b

print(add(2, 3))
```

This will output `5`."""

        code_blocks = extract_code_blocks(text)

        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "python"
        assert "def add(a, b):" in code_blocks[0]["code"]
        assert "print(add(2, 3))" in code_blocks[0]["code"]

    def test_extract_javascript_code(self):
        text = """Here's a simple JavaScript function:

```javascript
function add(a, b) {
    return a + b;
}

console.log(add(2, 3));
```

This will output `5`."""

        code_blocks = extract_code_blocks(text)

        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "javascript"
        assert "function add(a, b) {" in code_blocks[0]["code"]
        assert "console.log(add(2, 3));" in code_blocks[0]["code"]

    def test_extract_multiple_code_blocks(self):
        text = """Here are some code examples:

```python
print("Hello from Python")
```

And another example:

```javascript
console.log("Hello from JavaScript");
```
"""

        code_blocks = extract_code_blocks(text)

        assert len(code_blocks) == 2
        assert code_blocks[0]["language"] == "python"
        assert code_blocks[1]["language"] == "javascript"

    def test_extract_with_language_filter(self):
        text = """Here are some code examples:

```python
print("Hello from Python")
```

And another example:

```javascript
console.log("Hello from JavaScript");
```
"""

        code_blocks = extract_code_blocks(text, language="python")

        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "python"
        assert "Hello from Python" in code_blocks[0]["code"]

    def test_extract_with_no_language_specified(self):
        text = """Here's a code block with no language specified:

```
print("Hello, world!")
```
"""

        code_blocks = extract_code_blocks(text)

        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "unknown"
        assert "Hello, world!" in code_blocks[0]["code"]


class TestExecutePythonCode:
    def test_simple_python_execution(self):
        code = "print('Hello, world!')"
        result = execute_python_code(code)

        assert result["success"] is True
        assert result["output"] == "Hello, world!"
        assert result["error"] is None

    def test_python_execution_with_error(self):
        code = "print(undefined_variable)"
        result = execute_python_code(code)

        assert result["success"] is False
        assert result["output"] is None
        assert "NameError" in result["error"]

    def test_python_execution_with_timeout(self):
        code = "import time; time.sleep(10); print('This should timeout')"
        result = execute_python_code(code, timeout=1)

        assert result["success"] is False
        assert result["output"] is None
        assert "timeout" in result["error"].lower() or "timed out" in result["error"].lower()


# Skip these tests if Node.js is not installed
# Since Node.js is available, we'll let these tests run normally
# @pytest.mark.xfail(reason="Skipping if Node.js not installed")
class TestExecuteJavaScriptCode:
    def test_simple_javascript_execution(self):
        code = "console.log('Hello, world!');"
        result = execute_javascript_code(code)

        assert result["success"] is True
        assert result["output"] == "Hello, world!"
        assert result["error"] is None

    def test_javascript_execution_with_error(self):
        code = "console.log(undefinedVariable);"
        result = execute_javascript_code(code)

        assert result["success"] is False
        assert result["output"] is None
        # Our improved sandbox may return different error messages
        assert "undefined" in result["error"].lower() or "error" in result["error"].lower()

    def test_javascript_execution_with_timeout(self):
        code = "setTimeout(() => { console.log('Done'); }, 10000);"
        result = execute_javascript_code(code, timeout=1)

        assert result["success"] is False
        assert result["output"] is None
        assert "timeout" in result["error"].lower() or "timed out" in result["error"].lower()


class TestCompareOutputs:
    def test_exact_match(self):
        actual = "42"
        expected = "42"
        similarity = compare_outputs(actual, expected)

        assert similarity == 1.0

    def test_whitespace_normalization(self):
        actual = "  Hello,   world!  "
        expected = "Hello, world!"
        similarity = compare_outputs(actual, expected)

        assert similarity == 1.0

    def test_numeric_comparison(self):
        actual = "42.01"
        expected = "42.0"
        similarity = compare_outputs(actual, expected)

        assert similarity > 0.9  # Very close

        actual = "50"
        expected = "42"
        similarity = compare_outputs(actual, expected)

        assert similarity < 0.9  # More different

    def test_multiline_comparison(self):
        actual = "Line 1\nLine 2\nLine 3"
        expected = "Line 1\nLine 2\nLine 3"
        similarity = compare_outputs(actual, expected)

        assert similarity == 1.0

        actual = "Line 1\nLine 2\nLine X"  # One line different
        expected = "Line 1\nLine 2\nLine 3"
        similarity = compare_outputs(actual, expected)

        assert 0.7 < similarity < 1.0  # High but not perfect

    def test_list_comparison(self):
        actual = "[1, 2, 3]"
        expected = "[1, 2, 3]"
        similarity = compare_outputs(actual, expected)

        assert similarity == 1.0

        actual = "[1, 2, 3, 4]"  # Extra item
        expected = "[1, 2, 3]"
        similarity = compare_outputs(actual, expected)

        assert 0.7 < similarity < 1.0  # High but not perfect


class TestLocalCodeExecutionReward:
    def test_python_success_match(self):
        prompt_message_dict = {
            "role": "user",
            "content": "Write a function to add two numbers",
        }
        assistant_message_dict = {
            "role": "assistant",
            "content": """Here's a function to add two numbers:

```python
def add(a, b):
    return a + b

print(add(2, 3))
```

This will output `5`.
""",
        }
        messages_arg = [prompt_message_dict, assistant_message_dict]
        ground_truth_arg = "5"  # This is the expected_output_str

        result = local_code_execution_reward(messages=messages_arg, ground_truth=ground_truth_arg, language="python")

        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        # assert result['score'] == 1.0 # Use attribute access


CODE_WITH_ARG_COLLECTOR = """
def arg_collector(*args, **kwargs):
    return {"args": list(args), "kwargs": kwargs}

def another_func(x):
    return x * 2
"""

# Dummy messages for fractional_code_reward
DUMMY_MESSAGES_FOR_FRACTIONAL_REWARD = [
    Message(role="user", content="Call arg_collector"),
    Message(role="assistant", content=f"```python\n{CODE_WITH_ARG_COLLECTOR}\n```"),
]


class TestFractionalCodeRewardArgParsing:
    @pytest.mark.parametrize(
        "test_input_str, expected_args_list, expected_kwargs_dict",
        [
            # Test case format: (input_string_for_function, expected_args_as_list, expected_kwargs_as_dict)
            ("5", [5], {}),  # Single integer
            ('"hello"', ["hello"], {}),  # Single JSON string
            ("[1, 2, 3]", [[1, 2, 3]], {}),  # JSON array of ints
            (
                "['a', 'b', 'c']",
                [["a", "b", "c"]],
                {},
            ),  # Python list repr with single quotes
            ("1 'foo'", [1, "foo"], {}),  # Space-separated multiple args (int, string)
            (
                '"[1,2,3]"',
                [[1, 2, 3]],
                {},
            ),  # JSON string whose content is list-like (refined)
            ('"5"', [5], {}),  # JSON string whose content is number-like (refined)
            ("", [], {}),  # Empty string for no-arg call
            ("True None", [True, None], {}),  # Boolean and None
            ("{'key': 'value'}", [{"key": "value"}], {}),  # Python dict repr
            (
                '1.0 "[1, \\"a\\"]"',
                [1.0, [1, "a"]],
                {},
            ),  # Float and JSON string with escaped quotes
            (
                "arg1 arg2 arg3",
                ["arg1", "arg2", "arg3"],
                {},
            ),  # Multiple unquoted strings
        ],
    )
    def test_python_function_arg_parsing(self, test_input_str, expected_args_list, expected_kwargs_dict):
        expected_return_val = {
            "args": expected_args_list,
            "kwargs": expected_kwargs_dict,
        }

        test_cases = [{"input": test_input_str, "expected_output": repr(expected_return_val)}]

        # messages_arg combines prompt and assistant's code response
        messages_arg = [
            DUMMY_MESSAGES_FOR_FRACTIONAL_REWARD[0],  # User message
            DUMMY_MESSAGES_FOR_FRACTIONAL_REWARD[1],  # Assistant message with code
        ]
        # test_cases are now passed via the ground_truth parameter
        ground_truth_arg = test_cases

        result = fractional_code_reward(
            messages=messages_arg,
            ground_truth=ground_truth_arg,  # This now takes the test_cases
            language="python",
            # test_cases=test_cases, # Removed, passed via ground_truth
            environment="local",  # Ensure local execution for direct testing of parsing
            function_to_call="arg_collector",
        )

        assert isinstance(result, EvaluateResult), "Result should be an EvaluateResult object"
        assert hasattr(result, "score"), "Result object must contain a 'score' attribute"

        # Detailed assertion for debugging if something fails
        # Attribute access for score
        if result.score != 1.0:
            print(f"Test failed for input: {test_input_str}")
            print(f"Expected return: {repr(expected_return_val)}")
            # Attribute access for metrics
            test_results_metric_data = result.metrics.get("test_results") if result.metrics else None
            if test_results_metric_data:  # MetricResult object
                try:
                    actual_test_run_details_list = json.loads(test_results_metric_data.reason)
                    if (
                        actual_test_run_details_list
                        and isinstance(actual_test_run_details_list, list)
                        and len(actual_test_run_details_list) > 0
                    ):
                        first_test_detail = actual_test_run_details_list[0]
                        print(f"Actual output from execution: {first_test_detail.get('actual_output')}")
                        print(f"Test result details: {first_test_detail.get('details')}")
                    else:
                        print(f"Test results reason content not as expected: {actual_test_run_details_list}")
                except json.JSONDecodeError:
                    # Accessing reason from MetricResult object
                    print(
                        f"Could not parse test_results metric reason (JSONDecodeError): {test_results_metric_data.reason}"
                    )
            else:
                print(f"Full result (object): {result.model_dump()}")  # Dump object for full view

        assert result.score == 1.0, f"Test case for input '{test_input_str}' failed."
        # assert result['score'] == 1.0, f"Test case for input '{test_input_str}' failed (dictionary access)." # Use attribute access

        # Additionally, check the actual output if available in metrics
        test_results_metric = result.metrics.get("test_results") if result.metrics else None
        if test_results_metric:  # MetricResult object
            try:
                # The reason for test_results metric is a JSON string of the list of test results
                actual_test_run_details_list = json.loads(test_results_metric.reason)
                if (
                    actual_test_run_details_list
                    and isinstance(actual_test_run_details_list, list)
                    and len(actual_test_run_details_list) > 0
                ):
                    actual_output_str = actual_test_run_details_list[0].get("actual_output")
                    assert actual_output_str == repr(expected_return_val), (
                        f"Actual output '{actual_output_str}' did not match expected '{repr(expected_return_val)}' for input '{test_input_str}'"
                    )
            except json.JSONDecodeError:  # Catch specifically json.JSONDecodeError
                # Accessing reason from MetricResult object
                print(
                    f"Could not parse test_results metric reason (JSONDecodeError) for input '{test_input_str}': {test_results_metric.reason}"
                )
            except IndexError:
                print(f"test_results metric reason list was empty for input '{test_input_str}'")
        # The 'execution_result' metric might not be present if tests pass but output mismatches,
        # as it's typically for code execution status itself, not output comparison.
        # The primary check is result.score == 1.0 and the actual_output comparison.
        # If result.score == 1.0, it implies successful execution.

    def test_python_success_mismatch(self):
        prompt_message_dict = {
            "role": "user",
            "content": "Write a function to add two numbers",
        }
        assistant_message_dict = {
            "role": "assistant",
            "content": """Here's a function to add two numbers:

```python
def add(a, b):
    return a + b

print(add(1, 3))
```

This will output `4`.
""",
        }
        messages_arg = [prompt_message_dict, assistant_message_dict]
        ground_truth_arg = "5"  # This is the expected_output_str

        result = local_code_execution_reward(messages=messages_arg, ground_truth=ground_truth_arg, language="python")

        assert isinstance(result, EvaluateResult)
        assert result.score < 1.0
        assert "Output similarity:" in result.metrics["output_match"].reason
        # assert result['score'] < 1.0 # Use attribute access
        # assert "Output similarity:" in result['metrics']['output_match']['reason'] # Use attribute access

    def test_code_execution_failure(self):
        prompt_message_dict = {
            "role": "user",
            "content": "Write a function to add two numbers",
        }
        assistant_message_dict = {
            "role": "assistant",
            "content": """Here's a function to add two numbers:

```python
def add(a, b):
    return a + b

print(add(undeclared_variable, 3))
```

This will output `5`.
""",
        }
        messages_arg = [prompt_message_dict, assistant_message_dict]
        ground_truth_arg = "5"  # This is the expected_output_str

        result = local_code_execution_reward(messages=messages_arg, ground_truth=ground_truth_arg, language="python")

        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        assert "failed with error" in result.metrics["execution_result"].reason
        # assert result['score'] == 0.0 # Use attribute access
        # assert "failed with error" in result['metrics']['execution_result']['reason'] # Use attribute access

    def test_extract_expected_output_from_message(self):
        prompt_message_dict = {
            "role": "user",
            "content": "Write a function to add two numbers. Expected output: 5",
        }
        assistant_message_dict = {
            "role": "assistant",
            "content": """Here's a function to add two numbers:

```python
def add(a, b):
    return a + b

print(add(2, 3))
```

This will output `5`.
""",
        }
        messages_arg = [prompt_message_dict, assistant_message_dict]
        ground_truth_arg = "5"  # This is the expected_output_str

        result = local_code_execution_reward(
            messages=messages_arg,
            ground_truth=ground_truth_arg,
            language="python",
        )

        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        # assert result['score'] == 1.0 # Use attribute access
