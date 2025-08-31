"""
C/C++ code execution reward functions for evaluating C/C++ code correctness.

This module provides functions to evaluate the correctness of C/C++ code by:
1. Extracting code blocks from messages
2. Executing the code using the Piston execution engine
3. Comparing the output with expected results or running against test cases
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import aiohttp

from ..models import EvaluateResult, Message, MetricResult
from ..reward_function import reward_function


@dataclass
class TestResult:
    """
    Represents the result of a single test case execution.
    """

    test_name: str
    score: float = 0.0
    status: str = "SKIPPED"
    feedback: str = ""
    actual_output: str = ""
    expected_output: str = ""


class PistonError(Exception):
    """Exception raised for errors from the Piston API."""

    pass


class PistonClient:
    """
    A client that communicates with Piston API endpoints for code execution.

    Piston is a general purpose code execution engine:
    https://github.com/engineer-man/piston
    """

    def __init__(
        self,
        base_endpoint: str = "https://emkc.org/api/v2/piston",
        session: Optional[aiohttp.ClientSession] = None,
        timeout: int = 30,
    ):
        self.base_endpoint = base_endpoint
        self._session = session
        self.timeout = timeout

    @property
    def session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(sock_read=self.timeout),
                connector=aiohttp.TCPConnector(
                    limit=10,
                    ttl_dns_cache=300,
                    keepalive_timeout=30,
                ),
            )
        return self._session

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def get_runtimes(self) -> List[Dict[str, Any]]:
        """Get list of supported runtimes."""
        async with self.session.get(f"{self.base_endpoint}/runtimes") as response:
            if response.status != 200:
                raise PistonError(f"Error getting runtimes: {response.status}")
            return await response.json()

    async def execute(
        self,
        language: str,
        version: str,
        files: List[Dict[str, str]],
        stdin: str = "",
        args: List[str] = [],
        compile_timeout: int = 10000,
        run_timeout: int = 3000,
        compile_memory_limit: int = -1,
        run_memory_limit: int = -1,
    ) -> Dict[str, Any]:
        """
        Execute code using the Piston API.

        Args:
            language: Programming language (e.g., "c", "cpp")
            version: Version of the language (e.g., "10.2.0")
            files: List of files to include in execution (each with "name" and "content")
            stdin: Standard input to provide to the program
            args: Command-line arguments to pass to the program
            compile_timeout: Maximum compilation time in milliseconds
            run_timeout: Maximum execution time in milliseconds
            compile_memory_limit: Maximum memory for compilation in bytes (-1 for unlimited)
            run_memory_limit: Maximum memory for execution in bytes (-1 for unlimited)

        Returns:
            Dictionary containing the execution results
        """
        payload = {
            "language": language,
            "version": version,
            "files": files,
            "stdin": stdin,
            "args": args,
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "compile_memory_limit": compile_memory_limit,
            "run_memory_limit": run_memory_limit,
        }

        async with self.session.post(
            f"{self.base_endpoint}/execute",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise PistonError(f"Error executing code: {response.status} - {error_text}")

            result = await response.json()

            if "message" in result:
                raise PistonError(result["message"])

            return result


def get_piston_client(endpoint: Optional[str] = None) -> PistonClient:
    """
    Get a Piston client instance.

    Args:
        endpoint: Optional custom Piston API endpoint

    Returns:
        PistonClient instance
    """
    piston_endpoint = endpoint or os.environ.get("PISTON_ENDPOINT", "https://emkc.org/api/v2/piston")
    assert isinstance(piston_endpoint, str)
    return PistonClient(base_endpoint=piston_endpoint)


def extract_code_blocks(text: str, language: str = "cpp") -> List[Dict[str, str]]:
    """
    Extract code blocks from text.

    Args:
        text: The text to extract code blocks from
        language: Language to filter by (e.g., "cpp", "c")

    Returns:
        List of dictionaries with "code" and "language" keys
    """
    pattern = r"```(\w*)\n([\s\S]*?)\n```"
    matches = re.findall(pattern, text)

    code_blocks = []
    for lang, code in matches:
        lang = lang.lower()

        if language and lang:
            if language == "cpp" and lang not in ["cpp", "c++"]:
                continue
            elif language == "c" and lang != "c":
                continue
            elif language not in ["c", "cpp"] and language != lang:
                continue

        detected_lang = lang if lang else "unknown"
        code_blocks.append({"language": detected_lang, "code": code.strip()})

    return code_blocks


def add_cpp_includes(code: str) -> str:
    """
    Add common C++ includes if they're missing.

    Args:
        code: C++ code

    Returns:
        Code with added includes if necessary
    """
    if not code:
        return code

    includes = []

    if "#include <iostream>" not in code:
        includes.append("#include <iostream>")
    if "#include <vector>" not in code:
        includes.append("#include <vector>")
    if "#include <string>" not in code:
        includes.append("#include <string>")
    if "#include <bits/stdc++.h>" not in code:
        includes.append("#include <bits/stdc++.h>")
    if "using namespace std;" not in code and "std::" not in code:
        includes.append("using namespace std;")

    if includes:
        return "\n".join(includes) + "\n\n" + code

    return code


def add_c_includes(code: str) -> str:
    """
    Add common C includes if they're missing.

    Args:
        code: C code

    Returns:
        Code with added includes if necessary
    """
    if not code:
        return code

    includes = []

    if "#include <stdio.h>" not in code:
        includes.append("#include <stdio.h>")
    if "#include <stdlib.h>" not in code:
        includes.append("#include <stdlib.h>")
    if "#include <string.h>" not in code:
        includes.append("#include <string.h>")

    if includes:
        return "\n".join(includes) + "\n\n" + code

    return code


async def execute_cpp_code(
    code: str,
    stdin: str = "",
    language: str = "cpp",
    version: str = "11.4.0",
    timeout: int = 5000,
    memory_limit: int = 512000000,
    piston_endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute C/C++ code using the Piston API.

    Args:
        code: C/C++ code to execute
        stdin: Standard input to provide to the program
        language: "c" or "cpp"
        version: Version of the compiler to use
        timeout: Maximum execution time in milliseconds
        memory_limit: Maximum memory in bytes
        piston_endpoint: Optional custom Piston API endpoint

    Returns:
        Dictionary with execution results
    """
    if language == "cpp":
        code = add_cpp_includes(code)
    else:
        code = add_c_includes(code)

    client = get_piston_client(piston_endpoint)

    try:
        main_file = {
            "name": "main.cpp" if language == "cpp" else "main.c",
            "content": code,
        }

        result = await client.execute(
            language=language,
            version=version,
            files=[main_file],
            stdin=stdin,
            compile_timeout=timeout,
            run_timeout=timeout,
            run_memory_limit=memory_limit,
        )

        if "compile" in result and result["compile"]["code"] != 0:
            return {
                "success": False,
                "output": None,
                "error": f"Compilation error: {result['compile']['stderr']}",
            }

        if "run" in result:
            if result["run"]["code"] == 0:
                return {
                    "success": True,
                    "output": result["run"]["stdout"],
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "output": (result["run"]["stdout"] if result["run"]["stdout"] else None),
                    "error": f"Runtime error (exit code {result['run']['code']}): {result['run']['stderr']}",
                }

        return {
            "success": False,
            "output": None,
            "error": "Unknown error during execution",
        }

    except PistonError as e:
        return {
            "success": False,
            "output": None,
            "error": f"Piston error: {str(e)}",
        }
    except Exception as e:
        return {"success": False, "output": None, "error": f"Error: {str(e)}"}
    finally:
        loop = asyncio.get_event_loop()
        loop.create_task(client.close())


def compare_outputs(actual: str, expected: str) -> float:
    """
    Compare actual and expected outputs to calculate a similarity score.

    Args:
        actual: Actual output from code execution
        expected: Expected output

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if actual is None:
        actual = ""
    if expected is None:
        expected = ""

    actual_norm = re.sub(r"\s+", " ", actual.strip())
    expected_norm = re.sub(r"\s+", " ", expected.strip())

    if actual_norm == expected_norm:
        return 1.0

    try:
        actual_num = float(actual_norm)
        expected_num = float(expected_norm)

        if expected_num == 0:
            return 1.0 if actual_num == 0 else 0.0

        rel_diff = abs(actual_num - expected_num) / abs(expected_num)

        if rel_diff <= 0.001:
            return 1.0
        elif rel_diff <= 0.01:
            return 0.95
        elif rel_diff <= 0.1:
            return 0.7
        else:
            return max(0.0, 1.0 - min(1.0, rel_diff))
    except (ValueError, TypeError):
        pass

    if "\n" in actual_norm or "\n" in expected_norm:
        actual_lines = actual_norm.split("\n")
        expected_lines = expected_norm.split("\n")

        common_len = min(len(actual_lines), len(expected_lines))
        if common_len == 0:
            return 0.0

        line_similarities = []
        for i in range(common_len):
            if actual_lines[i] == expected_lines[i]:
                line_similarities.append(1.0)
            else:
                line_similarities.append(string_similarity(actual_lines[i], expected_lines[i]))

        total_weight = sum(1 / (i + 1) for i in range(common_len))
        weighted_sum = sum((1 / (i + 1)) * sim for i, sim in enumerate(line_similarities))
        similarity = weighted_sum / total_weight if total_weight > 0 else 0.0

        length_penalty = min(len(actual_lines), len(expected_lines)) / max(len(actual_lines), len(expected_lines))

        return similarity * length_penalty

    return string_similarity(actual_norm, expected_norm)


def string_similarity(s1: str, s2: str) -> float:
    """
    Calculate string similarity.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len if max_len > 0 else 0.0)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance between strings
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if not s2:
        return len(s1)

    previous_row: List[int] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


async def run_cpp_test_cases(
    code: str,
    test_cases: List[Dict[str, Any]],
    language: str = "cpp",
    version: str = "11.4.0",
    timeout: int = 5000,
    memory_limit: int = 512000000,
    piston_endpoint: Optional[str] = None,
) -> List[TestResult]:
    """
    Run C/C++ code against multiple test cases.

    Args:
        code: C/C++ code to execute
        test_cases: List of test cases with "input" and "expected_output" keys
        language: "c" or "cpp"
        version: Version of the compiler to use
        timeout: Maximum execution time in milliseconds
        memory_limit: Maximum memory in bytes
        piston_endpoint: Optional custom Piston API endpoint

    Returns:
        List of TestResult objects
    """
    results = []

    for i, test_case in enumerate(test_cases):
        test_input = test_case.get("input", "")
        expected_output = test_case.get("expected_output", "")
        test_name = test_case.get("name", f"Test {i + 1}")

        execution_result = await execute_cpp_code(
            code=code,
            stdin=test_input,
            language=language,
            version=version,
            timeout=timeout,
            memory_limit=memory_limit,
            piston_endpoint=piston_endpoint,
        )

        test_result = TestResult(test_name=test_name, expected_output=expected_output)

        if execution_result["success"]:
            actual_output = execution_result["output"]
            test_result.actual_output = actual_output
            similarity = compare_outputs(actual_output, expected_output)
            test_result.score = similarity

            if similarity >= 0.99:
                test_result.status = "AC"
            elif similarity > 0:
                test_result.status = "PA"
            else:
                test_result.status = "WA"
            test_result.feedback = f"Similarity: {similarity:.2f}"
        else:
            test_result.status = "CE" if "Compilation error" in execution_result["error"] else "RE"
            test_result.feedback = execution_result["error"]
            test_result.score = 0.0

        results.append(test_result)
        if test_result.score == 0.0:
            break

    return results


@reward_function
def ioi_cpp_code_reward(
    messages: List[Message],
    ground_truth: Union[Optional[str], Optional[List[Dict[str, Any]]]],
    language: str = "cpp",
    version: str = "11.4.0",
    timeout: int = 5000,
    memory_limit: int = 512000000,
    piston_endpoint: Optional[str] = None,
    pass_threshold: float = 0.99,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Wrapper function for the asynchronous implementation to make it compatible with the reward_function decorator.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        return _ioi_cpp_code_reward_impl(
            messages=messages,
            ground_truth=ground_truth,
            language=language,
            version=version,
            timeout=timeout,
            memory_limit=memory_limit,
            piston_endpoint=piston_endpoint,
            pass_threshold=pass_threshold,
            **kwargs,
        )
    finally:
        loop.close()


def _ioi_cpp_code_reward_impl(
    messages: List[Message],
    ground_truth: Union[Optional[str], Optional[List[Dict[str, Any]]]],
    language: str = "cpp",
    version: str = "11.4.0",
    timeout: int = 5000,
    memory_limit: int = 512000000,
    piston_endpoint: Optional[str] = None,
    pass_threshold: float = 0.99,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluate C/C++ code correctness using the Piston execution engine.

    This function evaluates code for competitive programming problems (like IOI)
    by compiling and executing C/C++ code against test cases.

    Args:
        messages: Generated conversation messages
        ground_truth: Expected output string or list of test case dictionaries.
        language: Programming language ("c" or "cpp")
        version: Version of the compiler to use
        timeout: Maximum execution time in milliseconds
        memory_limit: Maximum memory in bytes
        piston_endpoint: Optional custom Piston API endpoint
        pass_threshold: Similarity threshold for considering a test passed
        **kwargs: Additional keyword arguments

    Returns:
        EvaluateResult with score and metrics
    """
    metrics: Dict[str, MetricResult] = {}

    if (
        not messages
        or not isinstance(messages[-1], Message)
        or messages[-1].role != "assistant"
        or messages[-1].content is None
    ):
        return EvaluateResult(
            score=0.0,
            reason="Invalid or missing assistant response in messages.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Last message not a valid assistant response.",
                )
            },
        )

    response_content = messages[-1].content if isinstance(messages[-1].content, str) else ""

    expected_output_str_from_gt: Optional[str] = None
    test_cases_from_gt: Optional[List[Dict[str, Any]]] = None

    if isinstance(ground_truth, str):
        expected_output_str_from_gt = ground_truth
    elif isinstance(ground_truth, list):
        if all(isinstance(item, dict) for item in ground_truth):
            test_cases_from_gt = ground_truth
        else:
            return EvaluateResult(
                score=0.0,
                reason="Invalid ground_truth format: if list, must be list of test case dicts.",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Invalid ground_truth list format.",
                    )
                },
            )
    elif ground_truth is not None:
        return EvaluateResult(
            score=0.0,
            reason="Invalid ground_truth format: expected string, list of test case dicts, or None.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Invalid ground_truth format.",
                )
            },
        )

    code_blocks = extract_code_blocks(response_content, language)

    if not code_blocks:
        return EvaluateResult(
            score=0.0,
            reason=f"No {language} code blocks found in model's response.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason=f"No {language} code blocks found in model's response.",
                )
            },
        )

    code = code_blocks[0]["code"]

    metrics["extracted_code"] = MetricResult(
        score=0.0,
        is_score_valid=True,
        reason=f"Extracted code:\n```{language}\n{code}\n```",
    )

    if expected_output_str_from_gt and not test_cases_from_gt:
        metrics["expected_output"] = MetricResult(
            score=0.0,
            is_score_valid=True,
            reason=f"Expected output:\n{expected_output_str_from_gt}",
        )

    if test_cases_from_gt:
        results = asyncio.get_event_loop().run_until_complete(
            run_cpp_test_cases(
                code=code,
                test_cases=test_cases_from_gt,
                language=language,
                version=version,
                timeout=timeout,
                memory_limit=memory_limit,
                piston_endpoint=piston_endpoint,
            )
        )

        passed = sum(1 for result in results if result.score >= pass_threshold)
        total = len(results)
        overall_score = passed / total if total > 0 else 0.0
        final_reason = f"{passed}/{total} tests passed ({overall_score:.2%})."

        metrics["test_results"] = MetricResult(
            score=overall_score,
            is_score_valid=overall_score >= pass_threshold,
            reason=json.dumps(
                [
                    {
                        "test_name": result.test_name,
                        "status": result.status,
                        "score": result.score,
                        "feedback": result.feedback,
                    }
                    for result in results
                ],
                indent=2,
            ),
        )

        metrics["pass_rate"] = MetricResult(
            score=overall_score,
            is_score_valid=overall_score == 1.0,
            reason=f"{passed}/{total} tests passed ({overall_score:.2%})",
        )

        return EvaluateResult(score=overall_score, reason=final_reason, metrics=metrics)

    elif expected_output_str_from_gt:
        execution_result = asyncio.get_event_loop().run_until_complete(
            execute_cpp_code(
                code=code,
                language=language,
                version=version,
                timeout=timeout,
                memory_limit=memory_limit,
                piston_endpoint=piston_endpoint,
            )
        )

        if execution_result["success"]:
            output = execution_result["output"]
            final_reason = "Code executed successfully."

            metrics["execution_result"] = MetricResult(
                score=1.0,
                is_score_valid=True,
                reason=f"Code executed successfully with output:\n{output}",
            )

            similarity = compare_outputs(output, expected_output_str_from_gt)
            match_reason = (
                f"Output similarity: {similarity:.2f}\n\nExpected:\n{expected_output_str_from_gt}\n\nActual:\n{output}"
            )
            final_reason += f" Output similarity: {similarity:.2f}."

            metrics["output_match"] = MetricResult(
                score=similarity,
                is_score_valid=similarity >= pass_threshold,
                reason=match_reason,
            )

            return EvaluateResult(score=similarity, reason=final_reason, metrics=metrics)
        else:
            error = execution_result["error"]
            final_reason = f"Code execution failed: {error}"

            metrics["execution_result"] = MetricResult(
                score=0.0,
                is_score_valid=False,
                reason=f"Code execution failed with error:\n{error}",
            )

            return EvaluateResult(score=0.0, reason=final_reason, metrics=metrics)

    else:
        execution_result = asyncio.get_event_loop().run_until_complete(
            execute_cpp_code(
                code=code,
                language=language,
                version=version,
                timeout=timeout,
                memory_limit=memory_limit,
                piston_endpoint=piston_endpoint,
            )
        )

        if execution_result["success"]:
            output = execution_result["output"]
            final_reason = "Code executed successfully (no expected output for comparison)."

            metrics["execution_result"] = MetricResult(
                score=1.0,
                is_score_valid=True,
                reason=f"Code executed successfully with output:\n{output}",
            )

            return EvaluateResult(score=1.0, reason=final_reason, metrics=metrics)
        else:
            error = execution_result["error"]
            final_reason = f"Code execution failed: {error}"
            metrics["execution_result"] = MetricResult(
                score=0.0,
                is_score_valid=False,
                reason=f"Code execution failed with error:\n{error}",
            )

            return EvaluateResult(score=0.0, reason=final_reason, metrics=metrics)


@reward_function
def binary_cpp_code_reward(
    messages: List[Message],
    ground_truth: Union[Optional[str], Optional[List[Dict[str, Any]]]],
    language: str = "cpp",
    version: str = "11.4.0",
    timeout: int = 5000,
    memory_limit: int = 512000000,
    piston_endpoint: Optional[str] = None,
    pass_threshold: float = 0.99,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluate C/C++ code correctness and return a binary result (passed/failed).

    This function is a wrapper around ioi_cpp_code_reward that returns 1.0 if the
    score is at or above the pass_threshold, and 0.0 otherwise.

    Args:
        messages: Generated conversation messages
        ground_truth: Expected output string or list of test case dictionaries.
        language: Programming language ("c" or "cpp")
        version: Version of the compiler to use
        timeout: Maximum execution time in milliseconds
        memory_limit: Maximum memory in bytes
        piston_endpoint: Optional custom Piston API endpoint
        pass_threshold: Similarity threshold for considering a test passed
        **kwargs: Additional keyword arguments

    Returns:
        EvaluateResult with binary score (0.0 or 1.0) and metrics
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        reward_output = _ioi_cpp_code_reward_impl(
            messages=messages,
            ground_truth=ground_truth,
            language=language,
            version=version,
            timeout=timeout,
            memory_limit=memory_limit,
            piston_endpoint=piston_endpoint,
            pass_threshold=pass_threshold,
            **kwargs,
        )

        score = reward_output.score
        binary_score = 1.0 if score >= pass_threshold else 0.0
        metrics = dict(reward_output.metrics)
        final_reason = f"Binary score based on threshold {pass_threshold:.2f}. Original score: {score:.2f}."
        metrics["binary_result"] = MetricResult(
            score=binary_score,
            is_score_valid=binary_score == 1.0,
            reason=f"{'Passed' if binary_score > 0 else 'Failed'} (threshold: {pass_threshold:.2f}, actual: {score:.2f})",
        )

        return EvaluateResult(score=binary_score, reason=final_reason, metrics=metrics)
    finally:
        loop.close()
