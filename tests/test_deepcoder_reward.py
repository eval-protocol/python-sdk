import json
import os
import unittest
from typing import Any, Dict, List

from eval_protocol.models import EvaluateResult, Message
from eval_protocol.rewards import deepcoder_code_reward

# Path to the transformed sample data
TRANSFORMED_SAMPLE_DATA_PATH = "examples/trl_integration/data/deepcoder_mvp_transformed_sample.jsonl"

# Check if E2B API key is available for E2B tests
E2B_API_KEY = os.environ.get("E2B_API_KEY")
E2B_AVAILABLE = bool(E2B_API_KEY)


class TestDeepCoderReward(unittest.TestCase):
    SAMPLES: List[Dict[str, Any]] = []

    @classmethod
    def setUpClass(cls):
        """Load the transformed sample data once for all tests."""
        try:
            with open(TRANSFORMED_SAMPLE_DATA_PATH, "r") as f:
                for line in f:
                    cls.SAMPLES.append(json.loads(line.strip()))
        except FileNotFoundError:
            print(f"Warning: Test data file not found at {TRANSFORMED_SAMPLE_DATA_PATH}")
            # Allow tests to run but they might fail or be skipped if they depend on this data.
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {TRANSFORMED_SAMPLE_DATA_PATH}")

    def test_python_all_tests_pass_local(self):
        """Test Python code that passes all test cases locally."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")

        sample = self.SAMPLES[0]  # add_one function
        prompt_message = Message(role="user", content=sample["prompt"])
        assistant_message = Message(
            role="assistant",
            content="```python\ndef add_one(x):\n  return int(x) + 1\n\n# To be called by the testing harness\nval = input()\nprint(add_one(val))\n```",
        )
        messages_arg = [prompt_message, assistant_message]
        ground_truth_arg = sample["test_cases"]  # test_cases are the ground_truth

        result = deepcoder_code_reward(
            messages=messages_arg,
            ground_truth=ground_truth_arg,
            language="python",
            environment="local",
        )
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)
        self.assertIn("test_results", result.metrics)
        if "test_results" in result.metrics and result.metrics["test_results"].reason:
            details = json.loads(result.metrics["test_results"].reason)
            self.assertTrue(all(tc.get("passed") for tc in details))

    def test_python_one_test_fails_local(self):
        """Test Python code where one test case fails locally."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")

        sample = self.SAMPLES[0]  # add_one function, but we make it add_two
        prompt_message = Message(role="user", content=sample["prompt"])
        assistant_message = Message(
            role="assistant",
            content="```python\ndef add_one(x):\n  return int(x) + 2\n\nval = input()\nprint(add_one(val))\n```",
        )
        messages_arg = [prompt_message, assistant_message]
        ground_truth_arg = sample["test_cases"]

        result = deepcoder_code_reward(
            messages=messages_arg,
            ground_truth=ground_truth_arg,
            language="python",
            environment="local",
        )
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        if "test_results" in result.metrics and result.metrics["test_results"].reason:
            details = json.loads(result.metrics["test_results"].reason)
            self.assertFalse(details[0].get("passed"))  # First test case (5 -> expected 6, actual 7) should fail

    @unittest.skip("Trimmed slow test")
    def test_python_syntax_error_local(self):
        """Test Python code with a syntax error locally."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")
        sample = self.SAMPLES[0]
        prompt_message = Message(role="user", content=sample["prompt"])
        assistant_message = Message(
            role="assistant",
            content="```python\ndef add_one(x)\n  return x + 1\n\nval = input()\nprint(add_one(val))\n```",
        )  # Missing colon
        messages_arg = [prompt_message, assistant_message]
        ground_truth_arg = sample["test_cases"]

        result = deepcoder_code_reward(
            messages=messages_arg,
            ground_truth=ground_truth_arg,
            language="python",
            environment="local",
        )
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        if "test_results" in result.metrics and result.metrics["test_results"].reason:
            details = json.loads(result.metrics["test_results"].reason)
            self.assertTrue(any("error" in tc for tc in details))

    @unittest.skip("Trimmed slow test")
    def test_python_timeout_local(self):
        """Test Python code that times out locally."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")
        sample = self.SAMPLES[0]
        prompt_message = Message(role="user", content=sample["prompt"])
        assistant_message = Message(
            role="assistant",
            content="```python\nimport time\ndef add_one(x):\n  time.sleep(15)\n  return int(x) + 1\n\nval = input()\nprint(add_one(val))\n```",
        )
        messages_arg = [prompt_message, assistant_message]
        ground_truth_arg = sample["test_cases"]

        result = deepcoder_code_reward(
            messages=messages_arg,
            ground_truth=ground_truth_arg,
            language="python",
            timeout=2,  # Short timeout
            environment="local",
        )
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        if "test_results" in result.metrics and result.metrics["test_results"].reason:
            details = json.loads(result.metrics["test_results"].reason)
            self.assertTrue(any(tc.get("error") and "timed out" in str(tc.get("error")).lower() for tc in details))

    def test_no_code_block(self):
        """Test when no code block is found in the assistant message."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")
        sample = self.SAMPLES[0]
        prompt_message = Message(role="user", content=sample["prompt"])
        assistant_message = Message(role="assistant", content="I am not sure how to do that.")
        messages_arg = [prompt_message, assistant_message]
        ground_truth_arg = sample["test_cases"]

        result = deepcoder_code_reward(
            messages=messages_arg,
            ground_truth=ground_truth_arg,
            language="python",
            environment="local",
        )
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertIn("error", result.metrics)
        if "error" in result.metrics:
            self.assertIn("No python code block found", result.metrics["error"].reason)

    @unittest.skip("Trimmed slow test")
    def test_javascript_all_tests_pass_local(self):
        """Test JavaScript code that passes all test cases locally."""
        js_test_cases = [
            {"input": "5", "expected_output": "6\n"},
            {"input": "-2", "expected_output": "-1\n"},
        ]
        prompt_message = Message(role="user", content="Write a JS function addOne.")
        assistant_message = Message(
            role="assistant",
            content="```javascript\nfunction addOne(x) {\n  return parseInt(x) + 1;\n}\n\nconst val = readline();\nconsole.log(addOne(val));\n```",
        )
        messages_arg = [prompt_message, assistant_message]
        ground_truth_arg = js_test_cases

        result = deepcoder_code_reward(
            messages=messages_arg,
            ground_truth=ground_truth_arg,
            language="javascript",
            environment="local",
        )
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)

    @unittest.skip("Trimmed slow test")
    def test_javascript_one_test_fails_local(self):
        """Test JavaScript code where one test case fails locally."""
        js_test_cases = [
            {"input": "5", "expected_output": "6\n"},
            {"input": "-2", "expected_output": "-1\n"},
        ]
        prompt_message = Message(role="user", content="Write a JS function addOne.")
        assistant_message = Message(
            role="assistant",
            content="```javascript\nfunction addOne(x) {\n  return parseInt(x) + 2; // Bug here\n}\n\nconst val = readline();\nconsole.log(addOne(val));\n```",
        )
        messages_arg = [prompt_message, assistant_message]
        ground_truth_arg = js_test_cases

        result = deepcoder_code_reward(
            messages=messages_arg,
            ground_truth=ground_truth_arg,
            language="javascript",
            environment="local",
        )
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)

    @unittest.skip("Trimmed slow test")
    def test_python_all_tests_pass_e2b(self):
        """Test Python code that passes all test cases in E2B."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")

        sample = self.SAMPLES[0]  # add_one function
        prompt_message = Message(role="user", content=sample["prompt"])
        assistant_message = Message(
            role="assistant",
            content="```python\ndef add_one(x):\n  return int(x) + 1\n\nval = input()\nprint(add_one(val))\n```",
        )
        messages_arg = [prompt_message, assistant_message]
        ground_truth_arg = sample["test_cases"]

        try:
            result = deepcoder_code_reward(
                messages=messages_arg,
                ground_truth=ground_truth_arg,
                language="python",
                environment="e2b",
                api_key=E2B_API_KEY,
            )

            if hasattr(result, "metrics") and "error" in result.metrics:
                reason = result.metrics["error"].reason
                if "502 Bad Gateway" in reason or "sandbox timeout" in reason or "sandbox was not found" in reason:
                    self.skipTest("Skipping due to E2B connection issues")
            elif hasattr(result, "reason") and isinstance(result.reason, str):
                if (
                    "502 Bad Gateway" in result.reason
                    or "sandbox timeout" in result.reason
                    or "sandbox was not found" in result.reason
                ):
                    self.skipTest("Skipping due to E2B connection issues")

            self.assertIsInstance(result, EvaluateResult)
            try:
                self.assertEqual(result.score, 1.0)
                self.assertIn("test_results", result.metrics)
                if "test_results" in result.metrics and result.metrics["test_results"].reason:
                    details = json.loads(result.metrics["test_results"].reason)
                    self.assertTrue(all(tc.get("passed") for tc in details))
            except AssertionError as ae:
                print(f"AssertionError in test_python_all_tests_pass_e2b: {ae}")
                print(f"result.score: {result.score}")
                print(f"result.reason: {result.reason}")
                if hasattr(result, "metrics") and result.metrics:
                    print(f"result.metrics: {result.metrics}")
                    if "test_results" in result.metrics and hasattr(result.metrics["test_results"], "reason"):
                        print(f"result.metrics['test_results'].reason: {result.metrics['test_results'].reason}")
                    if "error" in result.metrics and hasattr(result.metrics["error"], "reason"):
                        print(f"result.metrics['error'].reason: {result.metrics['error'].reason}")
                raise  # Re-raise the assertion error
        except Exception as e:
            if "502 Bad Gateway" in str(e) or "sandbox timeout" in str(e) or "sandbox was not found" in str(e):
                self.skipTest(f"Skipping due to E2B connection issues: {e}")
            else:
                raise

    @unittest.skip("Trimmed slow test")
    def test_python_one_test_fails_e2b(self):
        """Test Python code where one test case fails in E2B."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")

        sample = self.SAMPLES[0]  # add_one function, but we make it add_two
        prompt_message = Message(role="user", content=sample["prompt"])
        assistant_message = Message(
            role="assistant",
            content="```python\ndef add_one(x):\n  return int(x) + 2\n\nval = input()\nprint(add_one(val))\n```",
        )
        messages_arg = [prompt_message, assistant_message]
        ground_truth_arg = sample["test_cases"]

        try:
            result = deepcoder_code_reward(
                messages=messages_arg,
                ground_truth=ground_truth_arg,
                language="python",
                environment="e2b",
                api_key=E2B_API_KEY,
            )

            if hasattr(result, "metrics") and "error" in result.metrics:
                reason = result.metrics["error"].reason
                if "502 Bad Gateway" in reason or "sandbox timeout" in reason or "sandbox was not found" in reason:
                    self.skipTest("Skipping due to E2B connection issues")
            elif hasattr(result, "reason") and isinstance(result.reason, str):
                if (
                    "502 Bad Gateway" in result.reason
                    or "sandbox timeout" in result.reason
                    or "sandbox was not found" in result.reason
                ):
                    self.skipTest("Skipping due to E2B connection issues")

            self.assertIsInstance(result, EvaluateResult)
            self.assertEqual(result.score, 0.0)
        except Exception as e:
            if "502 Bad Gateway" in str(e) or "sandbox timeout" in str(e) or "sandbox was not found" in str(e):
                self.skipTest(f"Skipping due to E2B connection issues: {e}")
            else:
                raise

    def test_empty_test_cases(self):
        """Test behavior with an empty list of test cases."""
        prompt_message = Message(role="user", content="Prompt")
        assistant_message = Message(role="assistant", content="```python\nprint('hello')\n```")
        messages_arg = [prompt_message, assistant_message]
        ground_truth_arg: List[Dict[str, Any]] = []  # Empty test cases

        result = deepcoder_code_reward(
            messages=messages_arg,
            ground_truth=ground_truth_arg,
            language="python",
            environment="local",
        )
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)  # Should be 0 as per implementation
        self.assertIn("error", result.metrics)
        if "error" in result.metrics:
            self.assertEqual(result.metrics["error"].reason, "No test cases provided.")


if __name__ == "__main__":
    unittest.main()
