"""
Tests for C/C++ code execution reward functions.
"""

import asyncio
import os
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eval_protocol.models import EvaluateResult, Message, MetricResult  # Changed
from eval_protocol.rewards.cpp_code import (
    PistonClient,
    _ioi_cpp_code_reward_impl,
    add_c_includes,
    add_cpp_includes,
    binary_cpp_code_reward,
    compare_outputs,
    execute_cpp_code,
    extract_code_blocks,
    ioi_cpp_code_reward,
    string_similarity,
)

# RewardOutput import removed, EvaluateResult is already imported

# Example C++ code for testing
SAMPLE_CPP_CODE = """
#include <iostream>
using namespace std;

int main() {
    int a, b;
    cin >> a >> b;
    cout << a + b << endl;
    return 0;
}
"""

SAMPLE_C_CODE = """
#include <stdio.h>

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    printf("%d\\n", a + b);
    return 0;
}
"""


class TestExtractCodeBlocks:
    def test_extract_cpp_code(self):
        text = """Here's a simple C++ function:

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
```

This will output `Hello, World!`."""

        code_blocks = extract_code_blocks(text)

        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "cpp"
        assert "#include <iostream>" in code_blocks[0]["code"]
        assert 'cout << "Hello, World!" << endl;' in code_blocks[0]["code"]

    def test_extract_c_code(self):
        text = """Here's a simple C function:

```c
#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}
```

This will output `Hello, World!`."""

        code_blocks = extract_code_blocks(text, language="c")

        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "c"
        assert "#include <stdio.h>" in code_blocks[0]["code"]
        assert 'printf("Hello, World!\\n");' in code_blocks[0]["code"]

    def test_extract_with_language_filter(self):
        text = """Here are multiple code blocks:

```python
print("Hello, Python!")
```

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, C++!" << endl;
    return 0;
}
```

```c
#include <stdio.h>

int main() {
    printf("Hello, C!\\n");
    return 0;
}
```
"""

        cpp_blocks = extract_code_blocks(text, language="cpp")
        assert len(cpp_blocks) == 1
        assert cpp_blocks[0]["language"] == "cpp"
        assert "Hello, C++" in cpp_blocks[0]["code"]

        c_blocks = extract_code_blocks(text, language="c")
        assert len(c_blocks) == 1
        assert c_blocks[0]["language"] == "c"
        assert "Hello, C!" in c_blocks[0]["code"]


class TestIncludes:
    def test_add_cpp_includes(self):
        code = "int main() { return 0; }"

        enhanced_code = add_cpp_includes(code)

        assert "#include <iostream>" in enhanced_code
        assert "#include <vector>" in enhanced_code
        assert "#include <string>" in enhanced_code
        assert "#include <bits/stdc++.h>" in enhanced_code
        assert "using namespace std;" in enhanced_code
        assert code in enhanced_code

    def test_add_c_includes(self):
        code = "int main() { return 0; }"

        enhanced_code = add_c_includes(code)

        assert "#include <stdio.h>" in enhanced_code
        assert "#include <stdlib.h>" in enhanced_code
        assert "#include <string.h>" in enhanced_code
        assert code in enhanced_code

    def test_dont_duplicate_includes(self):
        code = "#include <iostream>\nusing namespace std;\n\nint main() { return 0; }"

        enhanced_code = add_cpp_includes(code)

        # Check that includes aren't duplicated
        assert "#include <iostream>" in enhanced_code
        assert enhanced_code.count("#include <iostream>") == 1
        assert enhanced_code.count("using namespace std;") == 1


class TestOutputComparison:
    def test_exact_match(self):
        actual = "Hello, World!"
        expected = "Hello, World!"
        similarity = compare_outputs(actual, expected)

        assert similarity == 1.0

    def test_whitespace_normalization(self):
        actual = "Hello,    World!"
        expected = "Hello, World!"
        similarity = compare_outputs(actual, expected)

        assert similarity == 1.0

        actual = "Hello,\nWorld!"
        expected = "Hello, World!"
        similarity = compare_outputs(actual, expected)

        assert similarity <= 1.0

    def test_numeric_comparison(self):
        actual = "42.01"
        expected = "42.0"
        similarity = compare_outputs(actual, expected)

        assert similarity > 0.9  # Very close

        actual = "50"
        expected = "42"
        similarity = compare_outputs(actual, expected)

        assert similarity < 0.9  # Not very close

    def test_multiline_comparison(self):
        actual = "Line 1\nLine 2\nLine 3"
        expected = "Line 1\nLine 2\nLine 3"
        similarity = compare_outputs(actual, expected)

        assert similarity == 1.0

        actual = "Line 1\nLine 2\nLine X"  # One line different
        expected = "Line 1\nLine 2\nLine 3"
        similarity = compare_outputs(actual, expected)

        assert 0.7 < similarity < 1.0  # High but not perfect

    def test_string_similarity(self):
        assert string_similarity("hello", "hello") == 1.0
        assert string_similarity("hello", "hallo") < 1.0
        assert string_similarity("", "") == 1.0
        assert string_similarity("hello", "") == 0.0
        assert string_similarity("", "hello") == 0.0


# Mock response for Piston API
MOCK_PISTON_RUNTIME_RESPONSE = [
    {
        "language": "cpp",
        "version": "11.4.0",
        "aliases": ["c++"],
        "runtime": "gcc",
    },
    {"language": "c", "version": "11.3.0", "aliases": [], "runtime": "gcc"},
]

MOCK_PISTON_EXECUTE_SUCCESS = {
    "language": "cpp",
    "version": "11.4.0",
    "compile": {"stdout": "", "stderr": "", "code": 0, "signal": None},
    "run": {"stdout": "42", "stderr": "", "code": 0, "signal": None},
}

MOCK_PISTON_EXECUTE_COMPILE_ERROR = {
    "language": "cpp",
    "version": "11.4.0",
    "compile": {
        "stdout": "",
        "stderr": "main.cpp:5:10: error: 'cout' was not declared in this scope",
        "code": 1,
        "signal": None,
    },
}

MOCK_PISTON_EXECUTE_RUNTIME_ERROR = {
    "language": "cpp",
    "version": "11.4.0",
    "compile": {"stdout": "", "stderr": "", "code": 0, "signal": None},
    "run": {
        "stdout": "",
        "stderr": "Segmentation fault",
        "code": 139,
        "signal": "SIGSEGV",
    },
}


class TestPistonClient:
    def test_get_runtimes(self):
        # Setup mock for aiohttp get method
        m_session = MagicMock()
        m_response = MagicMock()

        # Configure mock return values
        async def mock_aenter(self):
            return m_response

        m_session.get.return_value.__aenter__ = mock_aenter

        m_response.status = 200

        # Async mock for response.json()
        async def mock_json():
            return MOCK_PISTON_RUNTIME_RESPONSE

        m_response.json = mock_json

        # Setup test client with our mock session
        client = PistonClient(base_endpoint="https://test.endpoint")
        client._session = m_session

        # Run the async method with event loop
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            runtimes = loop.run_until_complete(client.get_runtimes())

            # Check result
            assert runtimes == MOCK_PISTON_RUNTIME_RESPONSE
            assert len(runtimes) == 2
            assert runtimes[0]["language"] == "cpp"
            assert runtimes[1]["language"] == "c"
        finally:
            loop.close()

    def test_execute_success(self):
        # Setup mock for aiohttp post method
        m_session = MagicMock()
        m_response = MagicMock()

        # Configure mock return values
        async def mock_aenter(self):
            return m_response

        m_session.post.return_value.__aenter__ = mock_aenter

        m_response.status = 200

        # Async mock for response.json()
        async def mock_json():
            return MOCK_PISTON_EXECUTE_SUCCESS

        m_response.json = mock_json

        # Setup test client with our mock session
        client = PistonClient(base_endpoint="https://test.endpoint")
        client._session = m_session

        # Run the async method with event loop
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                client.execute(
                    language="cpp",
                    version="11.4.0",
                    files=[{"name": "main.cpp", "content": SAMPLE_CPP_CODE}],
                    stdin="10 15",
                )
            )

            # Check result
            assert result == MOCK_PISTON_EXECUTE_SUCCESS
            assert result["run"]["stdout"] == "42"
            assert result["run"]["code"] == 0
        finally:
            loop.close()

    def test_execute_compile_error(self):
        # Setup mock for aiohttp post method
        m_session = MagicMock()
        m_response = MagicMock()

        # Configure mock return values
        async def mock_aenter(self):
            return m_response

        m_session.post.return_value.__aenter__ = mock_aenter

        m_response.status = 200

        # Async mock for response.json()
        async def mock_json():
            return MOCK_PISTON_EXECUTE_COMPILE_ERROR

        m_response.json = mock_json

        # Setup test client with our mock session
        client = PistonClient(base_endpoint="https://test.endpoint")
        client._session = m_session

        # Run the async method with event loop
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                client.execute(
                    language="cpp",
                    version="11.4.0",
                    files=[
                        {
                            "name": "main.cpp",
                            "content": "int main() { cout << 42; }",
                        }
                    ],
                    stdin="",
                )
            )

            # Check result
            assert result == MOCK_PISTON_EXECUTE_COMPILE_ERROR
            assert result["compile"]["code"] == 1
            assert "error:" in result["compile"]["stderr"]
        finally:
            loop.close()

    def test_execute_runtime_error(self):
        # Setup mock for aiohttp post method
        m_session = MagicMock()
        m_response = MagicMock()

        # Configure mock return values
        async def mock_aenter(self):
            return m_response

        m_session.post.return_value.__aenter__ = mock_aenter

        m_response.status = 200

        # Async mock for response.json()
        async def mock_json():
            return MOCK_PISTON_EXECUTE_RUNTIME_ERROR

        m_response.json = mock_json

        # Setup test client with our mock session
        client = PistonClient(base_endpoint="https://test.endpoint")
        client._session = m_session

        # Run the async method with event loop
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                client.execute(
                    language="cpp",
                    version="11.4.0",
                    files=[
                        {
                            "name": "main.cpp",
                            "content": "int main() { int* p = nullptr; *p = 42; return 0; }",
                        }
                    ],
                    stdin="",
                )
            )

            # Check result
            assert result == MOCK_PISTON_EXECUTE_RUNTIME_ERROR
            assert result["run"]["code"] != 0
            assert result["run"]["signal"] == "SIGSEGV"
        finally:
            loop.close()


class TestExecuteCppCode:
    @patch("eval_protocol.rewards.cpp_code.PistonClient")
    def test_execute_cpp_success(self, MockPistonClient):
        # Setup the mock client
        mock_client = MagicMock()
        mock_client.execute = AsyncMock(return_value=MOCK_PISTON_EXECUTE_SUCCESS)
        MockPistonClient.return_value = mock_client

        # Also mock the close method
        mock_client.close = AsyncMock()

        # Mock get_piston_client
        with patch(
            "eval_protocol.rewards.cpp_code.get_piston_client",
            return_value=mock_client,
        ):
            # Call function with a synchronous wrapper
            async def run_test():
                return await execute_cpp_code(code=SAMPLE_CPP_CODE, stdin="10 15")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_test())

                # Check result
                assert result["success"] is True
                assert result["output"] == "42"
                assert result["error"] is None
            finally:
                loop.close()

    @patch("eval_protocol.rewards.cpp_code.PistonClient")
    def test_execute_cpp_compile_error(self, MockPistonClient):
        # Setup the mock client
        mock_client = MagicMock()
        mock_client.execute = AsyncMock(return_value=MOCK_PISTON_EXECUTE_COMPILE_ERROR)
        MockPistonClient.return_value = mock_client

        # Also mock the close method
        mock_client.close = AsyncMock()

        # Mock get_piston_client
        with patch(
            "eval_protocol.rewards.cpp_code.get_piston_client",
            return_value=mock_client,
        ):
            # Bad code with compilation error
            bad_code = "int main() { cout << 42; }"  # Missing iostream

            # Call function with a synchronous wrapper
            async def run_test():
                return await execute_cpp_code(code=bad_code)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_test())

                # Check result
                assert result["success"] is False
                assert result["output"] is None
                assert "Compilation error" in result["error"]
            finally:
                loop.close()

    @patch("eval_protocol.rewards.cpp_code.PistonClient")
    def test_execute_c_code(self, MockPistonClient):
        # Setup the mock client
        mock_client = MagicMock()
        mock_client.execute = AsyncMock(return_value=MOCK_PISTON_EXECUTE_SUCCESS)
        MockPistonClient.return_value = mock_client

        # Also mock the close method
        mock_client.close = AsyncMock()

        # Mock get_piston_client
        with patch(
            "eval_protocol.rewards.cpp_code.get_piston_client",
            return_value=mock_client,
        ):
            # Call function with a synchronous wrapper
            async def run_test():
                return await execute_cpp_code(code=SAMPLE_C_CODE, language="c", stdin="10 15")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_test())

                # Check result
                assert result["success"] is True
                assert result["output"] == "42"
                assert result["error"] is None
            finally:
                loop.close()


class TestIOICppCodeReward:
    @patch("eval_protocol.rewards.cpp_code.asyncio.get_event_loop")
    @patch("eval_protocol.rewards.cpp_code.execute_cpp_code")
    def test_success_match(self, mock_execute, mock_get_loop):
        # Set up mock event loop
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        # Setup the execution result
        mock_result = {"success": True, "output": "42", "error": None}
        mock_loop.run_until_complete.return_value = mock_result

        # Our execute_cpp_code mock can just return anything since we're bypassing it
        mock_execute.return_value = mock_result

        messages_data = [
            {
                "role": "user",
                "content": "Write a C++ program to add two numbers",
            },
            {
                "role": "assistant",
                "content": f"""Here's a program to add two numbers:

```cpp
{SAMPLE_CPP_CODE}
```

This program reads two integers and outputs their sum.
""",
            },
        ]

        messages_arg = [Message(**messages_data[0]), Message(**messages_data[1])]
        ground_truth_arg = "42"  # This is the expected_output_str

        # Call the function - should use our mocked get_event_loop() and run_until_complete()
        result = _ioi_cpp_code_reward_impl(messages=messages_arg, ground_truth=ground_truth_arg, language="cpp")

        # Check result - execution should have succeeded with perfect match
        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        assert "executed successfully" in result.metrics["execution_result"].reason
        # assert result['score'] == 1.0 # Use attribute access
        # assert (
        #     "executed successfully" in result['metrics']["execution_result"]['reason'] # Use attribute access
        # )

    @patch("eval_protocol.rewards.cpp_code.asyncio.get_event_loop")
    @patch("eval_protocol.rewards.cpp_code.execute_cpp_code")
    def test_success_mismatch(self, mock_execute, mock_get_loop):
        # Set up mock event loop
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        # Setup the execution result
        mock_result = {"success": True, "output": "25", "error": None}
        mock_loop.run_until_complete.return_value = mock_result

        # Our execute_cpp_code mock can just return anything since we're bypassing it
        mock_execute.return_value = mock_result

        messages_data = [
            {
                "role": "user",
                "content": "Write a C++ program to add two numbers",
            },
            {
                "role": "assistant",
                "content": f"""Here's a program to add two numbers:

```cpp
{SAMPLE_CPP_CODE}
```

This program reads two integers and outputs their sum.
""",
            },
        ]

        messages_arg = [Message(**messages_data[0]), Message(**messages_data[1])]
        ground_truth_arg = "42"  # This is the expected_output_str

        # Call the function - should use our mocked get_event_loop() and run_until_complete()
        result = _ioi_cpp_code_reward_impl(messages=messages_arg, ground_truth=ground_truth_arg, language="cpp")

        # Check result - should have partial match
        assert isinstance(result, EvaluateResult)
        assert result.score < 1.0
        assert "Output similarity:" in result.metrics["output_match"].reason
        # assert result['score'] < 1.0 # Use attribute access
        # assert "Output similarity:" in result['metrics']["output_match"]['reason'] # Use attribute access

    @patch("eval_protocol.rewards.cpp_code.asyncio.get_event_loop")
    @patch("eval_protocol.rewards.cpp_code.execute_cpp_code")
    def test_execution_failure(self, mock_execute, mock_get_loop):
        # Set up mock event loop
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        # Setup the execution result
        mock_result = {
            "success": False,
            "output": None,
            "error": "Compilation error",
        }
        mock_loop.run_until_complete.return_value = mock_result

        # Our execute_cpp_code mock can just return anything since we're bypassing it
        mock_execute.return_value = mock_result

        messages_data = [
            {
                "role": "user",
                "content": "Write a C++ program to add two numbers",
            },
            {
                "role": "assistant",
                "content": """Here's a program to add two numbers:

```cpp
int main() {
    cout << "Hello, World!" << endl;  // Missing iostream include
    return 0;
}
```
""",
            },
        ]

        messages_arg = [Message(**messages_data[0]), Message(**messages_data[1])]
        ground_truth_arg = "42"  # This is the expected_output_str

        # Call the function - should use our mocked get_event_loop() and run_until_complete()
        result = _ioi_cpp_code_reward_impl(messages=messages_arg, ground_truth=ground_truth_arg, language="cpp")

        # Check result - execution should have failed
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        assert "failed with error" in result.metrics["execution_result"].reason
        # assert result['score'] == 0.0 # Use attribute access
        # assert "failed with error" in result['metrics']["execution_result"]['reason'] # Use attribute access

    @patch("eval_protocol.rewards.cpp_code.asyncio.get_event_loop")
    @patch("eval_protocol.rewards.cpp_code.run_cpp_test_cases")
    def test_multiple_test_cases(self, mock_run_tests, mock_get_loop):
        # Set up mock event loop
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        # Create test cases result
        from eval_protocol.rewards.cpp_code import TestResult

        test_results = [
            TestResult(test_name="Test 1", score=1.0, status="AC", feedback="Perfect"),
            TestResult(test_name="Test 2", score=1.0, status="AC", feedback="Perfect"),
            TestResult(test_name="Test 3", score=0.5, status="PA", feedback="Partial"),
        ]

        # Setup mock return values
        mock_loop.run_until_complete.return_value = test_results
        mock_run_tests.return_value = test_results

        messages_data = [
            {
                "role": "user",
                "content": "Write a C++ program to add two numbers",
            },
            {
                "role": "assistant",
                "content": f"""Here's a program to add two numbers:

```cpp
{SAMPLE_CPP_CODE}
```

This program reads two integers and outputs their sum.
""",
            },
        ]

        test_cases_data = [
            {"name": "Test 1", "input": "10 15", "expected_output": "25"},
            {"name": "Test 2", "input": "0 0", "expected_output": "0"},
            {"name": "Test 3", "input": "-5 5", "expected_output": "0"},
        ]

        messages_arg = [Message(**messages_data[0]), Message(**messages_data[1])]
        ground_truth_arg = test_cases_data  # This is the test_cases

        # Call the function - should use our mocked get_event_loop() and run_until_complete()
        result = _ioi_cpp_code_reward_impl(messages=messages_arg, ground_truth=ground_truth_arg, language="cpp")

        # Check result
        assert isinstance(result, EvaluateResult)

        # The score in _ioi_cpp_code_reward_impl is calculated based on the ratio of tests
        # that pass the pass_threshold (not an average of scores).
        # With a default pass_threshold of 0.99, only the first two tests would pass,
        # resulting in 2/3 = 0.6666...
        expected_score = 2.0 / 3.0  # 2 out of 3 tests pass the threshold
        assert abs(result.score - expected_score) < 0.001  # Use approximate comparison
        assert "2/3 tests passed" in result.metrics["pass_rate"].reason
        # assert (
        #     abs(result['score'] - expected_score) < 0.001 # Use attribute access
        # )
        # assert "2/3 tests passed" in result['metrics']["pass_rate"]['reason'] # Use attribute access


class TestBinaryCppCodeReward:
    @patch("eval_protocol.rewards.cpp_code._ioi_cpp_code_reward_impl")
    def test_binary_pass(self, mock_reward_impl):
        # Set up mock response
        mock_metrics = {
            "execution_result": MetricResult(score=1.0, reason="Code executed successfully", success=True),
            "output_match": MetricResult(score=1.0, reason="Perfect match", success=True),
        }
        mock_reward_impl.return_value = EvaluateResult(score=1.0, reason="Binary pass", metrics=mock_metrics)

        messages = [
            {
                "role": "user",
                "content": "Write a C++ program to add two numbers",
            },
            {
                "role": "assistant",
                "content": f"""Here's a program to add two numbers:

```cpp
{SAMPLE_CPP_CODE}
```
""",
            },
        ]

        messages_arg = [Message(**messages[0]), Message(**messages[1])]
        ground_truth_arg = "25"  # This is the expected_output_str

        # Call function
        result = binary_cpp_code_reward(messages=messages_arg, ground_truth=ground_truth_arg, language="cpp")

        # Check result
        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        assert "Passed" in result.metrics["binary_result"].reason
        # assert result['score'] == 1.0 # Use attribute access
        # assert "Passed" in result['metrics']['binary_result']['reason'] # Use attribute access

    @patch("eval_protocol.rewards.cpp_code._ioi_cpp_code_reward_impl")
    def test_binary_fail(self, mock_reward_impl):
        # Set up mock response
        mock_metrics = {
            "execution_result": MetricResult(score=1.0, reason="Code executed successfully", success=True),
            "output_match": MetricResult(score=0.8, reason="Close match", success=False),
        }
        mock_reward_impl.return_value = EvaluateResult(
            score=0.8, reason="Binary fail due to partial match", metrics=mock_metrics
        )

        messages = [
            {
                "role": "user",
                "content": "Write a C++ program to add two numbers",
            },
            {
                "role": "assistant",
                "content": f"""Here's a program to add two numbers:

```cpp
{SAMPLE_CPP_CODE}
```
""",
            },
        ]

        messages_arg = [Message(**messages[0]), Message(**messages[1])]
        ground_truth_arg = "25"  # This is the expected_output_str

        # Call function
        result = binary_cpp_code_reward(
            messages=messages_arg,
            ground_truth=ground_truth_arg,
            language="cpp",
            pass_threshold=0.9,  # Set threshold higher than the actual score
        )

        # Check result
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        assert "Failed" in result.metrics["binary_result"].reason
        # assert result['score'] == 0.0 # Use attribute access
        # assert "Failed" in result['metrics']['binary_result']['reason'] # Use attribute access
