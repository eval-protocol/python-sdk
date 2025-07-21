import json  # Added import
import logging
from typing import Any, Dict, List, Optional

import pytest

from reward_protocol.models import EvaluateResult, Message, MetricResult
from reward_protocol.rewards.apps_coding_reward import (
    _extract_python_code,
    evaluate_apps_solution,
)

# Configure logger for tests to see debug messages if needed
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__) # Not strictly needed here if only testing functions

# --- Tests for _extract_python_code ---


@pytest.mark.parametrize(
    "response_content, expected_code",
    [
        # Basic Python code block
        (
            "```python\ndef hello():\n    print('Hello')\n```",
            "def hello():\n    print('Hello')",
        ),
        # Basic code block (no language specified)
        (
            "```\ndef world():\n    return 'World'\n```",
            "def world():\n    return 'World'",
        ),
        # Code starting with def, no markdown
        ("def my_func():\n    pass", "def my_func():\n    pass"),
        # Think block before python code block
        (
            "<think>Thinking...</think>\n```python\ndef main():\n    # main function\n    return 0\n```",
            "def main():\n    # main function\n    return 0",
        ),
        # Think block after python code block
        (
            "```python\ndef another():\n    x = 1\n```\n<think>Done.</think>",
            "def another():\n    x = 1",
        ),
        # Think block around python code block
        (
            "<think>Let's code.</think>\n```python\ndef surrounded():\n    y = 2\n```\n<think>Finished.</think>",
            "def surrounded():\n    y = 2",
        ),
        # Think block with no other code
        (
            "<think>Just thinking, no code here.</think>",
            None,
        ),  # Expect None or empty after stripping
        # Think block that becomes empty after removal
        ("<think>Only this.</think>", None),
        # No code, no think block (e.g., refusal)
        (
            "I'm sorry, I cannot fulfill this request.",
            "I'm sorry, I cannot fulfill this request.",
        ),  # Fallback to full content
        # Empty string
        ("", ""),
        # Whitespace only
        ("   \n\t   ", ""),
        # Code with leading/trailing whitespace in markdown
        (
            "  ```python  \n  def spaced_out():\n    val = 10\n  ```  ",
            "def spaced_out():\n    val = 10",
        ),
        # Think block with mixed case
        (
            "<THINK>Mixed case think</THINK>\n```python\ndef case_test():\n    pass\n```",
            "def case_test():\n    pass",
        ),
        # No markers, just code starting with def
        (
            "\n\n# Some comments\ndef simple_def():\n    return True\n# Trailing comment",
            "def simple_def():\n    return True\n# Trailing comment",
        ),
        # No markers, no def, just some text (should return the text)
        ("This is just some text, not code.", "This is just some text, not code."),
        # Think block and then just text, no code markers
        ("<think>Hmm</think>This is text after think.", "This is text after think."),
    ],
)
def test_extract_python_code(response_content: str, expected_code: Optional[str]):
    extracted = _extract_python_code(response_content)
    if expected_code is None:
        assert (
            extracted is None or extracted == ""
        )  # Allow empty string if think block was everything
    else:
        assert extracted == expected_code


# --- Tests for evaluate_apps_solution ---


def test_evaluate_apps_solution_parsable_code():
    messages = [
        Message(role="user", content="prompt"),
        Message(role="assistant", content="```python\ndef foo():\n  return 42\n```"),
    ]
    result = evaluate_apps_solution(
        messages=messages, ground_truth='{"inputs": [], "outputs": []}'
    )
    # With empty inputs/outputs, 0 tests are run by check_correctness.
    # The score becomes 0.0, and reason "Passed 0/0 test cases."
    assert result.score == 0.0
    assert (
        result.reason == "Execution utility returned no results."
    )  # Corrected assertion
    # The "parsability" metric is not set in this path by the current evaluate_apps_solution logic.
    # If it were, it would be 1.0. For now, we remove this assertion or adapt if parsability metric is re-introduced.
    # assert "parsability" not in result.metrics # Or check its specific value if it's expected


def test_evaluate_apps_solution_non_parsable_code():
    messages = [
        Message(role="user", content="prompt"),
        Message(role="assistant", content="def foo():\n  return 42 oops"),
    ]
    result = evaluate_apps_solution(
        messages=messages, ground_truth='{"inputs": [], "outputs": []}'
    )
    assert result.score == 0.0
    assert result.reason and "Execution Error: SyntaxError" in result.reason
    # The "parsability" metric is not set directly for syntax errors caught by check_correctness.
    # Instead, "execution_error_details" will contain the SyntaxError.
    assert "execution_error_details" in result.metrics
    assert result.metrics["execution_error_details"].reason is not None
    details = json.loads(result.metrics["execution_error_details"].reason)
    assert "SyntaxError" in details.get("error", "")


def test_evaluate_apps_solution_empty_code_after_extraction():
    # This simulates if _extract_python_code returns None or empty string
    messages = [
        Message(role="user", content="prompt"),
        Message(role="assistant", content="<think>This is all I have.</think>"),
    ]
    # If code_solution is None, it returns early. ground_truth doesn't matter here.
    result = evaluate_apps_solution(
        messages=messages, ground_truth='{"inputs": [], "outputs": []}'
    )
    assert result.score == 0.0
    assert result.reason == "The provided code solution was empty after extraction."
    assert result.metrics["parsability"].score == 0.0
    assert (
        result.metrics["parsability"].reason == "Empty code solution after extraction."
    )


def test_evaluate_apps_solution_no_messages():
    result = evaluate_apps_solution(messages=[], ground_truth="{}")
    assert result.score == 0.0
    assert result.reason == "No messages provided."
    assert result.metrics["error"].reason == "No messages provided for evaluation."


def test_evaluate_apps_solution_empty_assistant_message_content():
    messages = [
        Message(role="user", content="prompt"),
        Message(role="assistant", content=""),
    ]
    result = evaluate_apps_solution(messages=messages, ground_truth="{}")
    assert result.score == 0.0
    assert result.reason == "The provided code solution was empty after extraction."
    assert result.metrics["parsability"].score == 0.0
    assert (
        result.metrics["parsability"].reason == "Empty code solution after extraction."
    )


def test_evaluate_apps_solution_refusal_message():
    refusal_text = "I'm sorry, but your message seems to be incomplete."
    messages = [
        Message(role="user", content="prompt"),
        Message(role="assistant", content=refusal_text),
    ]
    result = evaluate_apps_solution(
        messages=messages, ground_truth='{"inputs": [], "outputs": []}'
    )
    assert result.score == 0.0
    assert result.reason and "Execution Error: SyntaxError" in result.reason
    # Similar to above, check execution_error_details for the SyntaxError
    assert "execution_error_details" in result.metrics
    assert result.metrics["execution_error_details"].reason is not None
    details = json.loads(result.metrics["execution_error_details"].reason)
    assert "SyntaxError" in details.get("error", "")
