# mypy: ignore-errors
"""
Code execution reward functions for evaluating code correctness.

This module provides functions to evaluate the correctness of code by:
1. Extracting code blocks from messages
2. Executing the code in a secure environment (local or E2B sandbox)
3. Comparing the output with expected results

Available reward functions:
- local_code_execution_reward: Execute code locally and evaluate correctness
- e2b_code_execution_reward: Execute code in E2B sandbox and evaluate correctness
- fractional_code_reward: Execute code and return exact pass rate
"""

import faulthandler
import json
import multiprocessing
import os
import platform
import re
import resource
import shlex  # Added for robust splitting of arguments
import signal
import subprocess
import sys
import tempfile
import traceback
from io import StringIO
from multiprocessing.managers import DictProxy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Try to import from e2b_code_interpreter first (preferred)
try:
    from e2b_code_interpreter.sync import Sandbox  # type: ignore # Use SyncSandbox

    _HAS_E2B = True
    _E2B_SOURCE = "e2b_code_interpreter"
except ImportError:
    # Fallback to e2b
    try:
        # Assuming 'e2b' package's default Sandbox is synchronous.
        # If 'e2b' also defaults to async, this part might need adjustment too.
        from e2b import Sandbox  # type: ignore

        _HAS_E2B = True
        _E2B_SOURCE = "e2b"
    except ImportError:
        _HAS_E2B = False
        _E2B_SOURCE = ""  # Use empty string instead of None

from ..models import EvaluateResult, Message, MetricResult
from ..reward_function import reward_function


def _target_func_for_execution(result_container, execute_func, args):
    try:
        result = execute_func(*args)
        result_container.update(result)
    except Exception as e:
        error_traceback = traceback.format_exc()
        result_container.update(
            {
                "success": False,
                "output": None,
                "error": f"Execution error: {str(e)}\n{error_traceback}",
            }
        )


def extract_code_blocks(text: str, language: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Extract code blocks from text.

    Args:
        text: The text to extract code blocks from
        language: Optional language to filter by (e.g., "python", "javascript")

    Returns:
        List of dictionaries with "code" and "language" keys
    """
    pattern = r"```(\w*)\n([\s\S]*?)\n```"
    matches = re.findall(pattern, text or "")

    code_blocks = []
    verbose_patterns_removed = []

    # Define patterns for verbose text that might appear inside code blocks
    # These patterns will be removed.
    # Using re.DOTALL to make '.' match newlines.
    verbose_regex_patterns = [
        re.compile(r"<think>.*?</think>", re.DOTALL),
        re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL),
        re.compile(r"Thinking:\s*.*?(?=\n\S)", re.DOTALL),  # Matches "Thinking: ..." until a new non-whitespace line
        re.compile(r"^\s*Here's the Python code.*?\n", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^\s*Okay, here is the code:.*?\n", re.MULTILINE | re.IGNORECASE),
    ]

    for lang, code_content in matches:
        if language and lang and language.lower() != lang.lower():
            continue

        detected_lang = lang.lower() if lang else "unknown"
        original_code_content = code_content
        cleaned_code_content = code_content

        for verbose_pattern in verbose_regex_patterns:
            cleaned_code_content = verbose_pattern.sub("", cleaned_code_content)

        if cleaned_code_content != original_code_content:
            verbose_patterns_removed.append(f"Verbose content removed from '{detected_lang}' block.")

        block_info = {
            "language": detected_lang,
            "code": cleaned_code_content.strip(),
        }
        if verbose_patterns_removed:
            block_info["verbosity_cleaned_reason"] = "; ".join(verbose_patterns_removed)
            verbose_patterns_removed = []

        code_blocks.append(block_info)

    return code_blocks


@reward_function
def local_code_execution_reward(
    messages: List[Message],
    ground_truth: Optional[str] = None,  # This is the new expected_output_str
    language: str = "python",
    timeout: int = 5,
    max_memory_mb: int = 100,  # Specific to local execution
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate code correctness by executing it locally and comparing the output.

    This function executes code in a secure sandbox with memory limits, CPU limits,
    and timeouts to prevent malicious code from harming the system.

    Args:
        messages: List of conversation messages. The last message is assumed to be the
                  assistant's response containing the code.
        ground_truth: Expected output string from code execution. This corresponds to
                      the `expected_output_str` in the previous signature.
        language: Programming language of the code ("python", "javascript", etc.)
        timeout: Maximum execution time in seconds.
        max_memory_mb: Maximum memory usage in megabytes (default: 100).
        **kwargs: Additional keyword arguments.

    Returns:
        EvaluateResult with score and metrics.
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

    # Normalize content to string; Message.content may be str or list of content parts
    last_content = messages[-1].content
    response_content = (
        last_content if isinstance(last_content, str) else "".join([p.text for p in (last_content or [])])
    )
    expected_output_str = ground_truth

    code_blocks = extract_code_blocks(response_content, language)

    if not code_blocks:
        return EvaluateResult(
            score=0.0,
            reason=f"No {language} code blocks found in model's response.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason=f"No {language} code blocks found in model's response.",
                    is_score_valid=False,
                )
            },
        )

    code = code_blocks[0]["code"]

    metrics["extracted_code"] = MetricResult(
        score=0.0,
        reason=f"Extracted code:\n```{language}\n{code}\n```",
        is_score_valid=True,
    )

    if expected_output_str:
        metrics["expected_output"] = MetricResult(
            score=0.0,
            reason=f"Expected output:\n{expected_output_str}",
            is_score_valid=True,
        )

    if language.lower() == "python":
        execution_result = execute_python_code(
            code, timeout
        )  # max_memory_mb is handled inside _execute_python_in_subprocess
    elif language.lower() in ["javascript", "js"]:
        execution_result = execute_javascript_code(code, timeout)
    else:
        metrics["error"] = MetricResult(score=0.0, reason=f"Unsupported language: {language}", is_score_valid=False)
        return EvaluateResult(score=0.0, reason=f"Unsupported language: {language}", metrics=metrics)

    if execution_result["success"]:
        output = execution_result["output"]

        metrics["execution_result"] = MetricResult(
            score=1.0,
            reason=f"Code executed successfully with output:\n{output}",
            is_score_valid=True,
        )

        if expected_output_str:
            similarity = compare_outputs(output, expected_output_str)
            match_reason = (
                f"Output similarity: {similarity:.2f}\n\nExpected:\n{expected_output_str}\n\nActual:\n{output}"
            )

            metrics["output_match"] = MetricResult(
                score=similarity, reason=match_reason, is_score_valid=similarity == 1.0
            )
            final_reason = f"Execution successful. Output similarity: {similarity:.2f}."
            return EvaluateResult(score=similarity, reason=final_reason, metrics=metrics)

        final_reason = "Execution successful. No expected output to compare."
        return EvaluateResult(score=1.0, reason=final_reason, metrics=metrics)
    else:
        error = execution_result["error"]

        metrics["execution_result"] = MetricResult(
            score=0.0,
            reason=f"Code execution failed with error:\n{error}",
            is_score_valid=False,
        )
        final_reason = f"Code execution failed: {error}"
        return EvaluateResult(score=0.0, reason=final_reason, metrics=metrics)


def _process_target_wrapper(execute_func: Callable, args: Tuple, result_container: DictProxy):
    try:
        result = execute_func(*args)
        result_container.update(result)
    except Exception as e:
        error_traceback = traceback.format_exc()
        result_container.update(
            {
                "success": False,
                "output": None,
                "error": f"Execution error: {str(e)}\n{error_traceback}",
            }
        )


def _execute_code_in_process(execute_func: Callable, args: Tuple, timeout: int = 5) -> Dict[str, Any]:
    """
    Execute code in a separate process with timeout and resource limits.

    Args:
        execute_func: Function to execute the code
        args: Arguments to pass to the execute function
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """

    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    process = multiprocessing.Process(target=_process_target_wrapper, args=(execute_func, args, result_dict))
    process.start()
    process.join(timeout=timeout + 0.5)

    if process.is_alive():
        process.terminate()
        process.join(0.5)
        if process.is_alive():
            process.kill()
        return {
            "success": False,
            "output": None,
            "error": f"Timeout: execution timed out after {timeout} seconds",
        }

    if not result_dict:
        return {
            "success": False,
            "output": None,
            "error": "Execution failed without producing any output",
        }

    return dict(result_dict)


def _execute_python_in_subprocess(code: str, timeout: int) -> Dict[str, Any]:
    """
    Inner function to execute Python code in a subprocess.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file_path = temp_file.name

            safe_code = (
                "import sys\n"
                "import os\n"
                "import signal\n"
                "import resource\n"
                "import platform\n\n"
                "def _reliability_guard():\n"
                "    memory_limit = 100 * 1024 * 1024  # 100 MB\n"
                "    if platform.uname().system != 'Darwin':\n"
                "        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))\n"
                "        resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, memory_limit))\n"
                "        resource.setrlimit(resource.RLIMIT_STACK, (memory_limit, memory_limit))\n"
                "    import builtins\n"
                "    builtins.exit = None\n"
                "    builtins.quit = None\n"
                "    os.environ['OMP_NUM_THREADS'] = '1'\n"
                "    os.system = None\n"
                "    os.popen = None\n"
                "    os.execl = None\n"
                "    os.execve = None\n"
                "    os.fork = None\n"
                "    os.remove = None\n"
                "    os.removedirs = None\n"
                "    os.rmdir = None\n"
                "    os.unlink = None\n"
                "    os.access = None\n"
                "\n"
                "_reliability_guard()\n\n" + code
            )

            temp_file.write(safe_code.encode("utf-8"))

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout} seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            process = subprocess.Popen(
                [sys.executable, temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 1)),
            )

            stdout, stderr = process.communicate()
            signal.alarm(0)

            if process.returncode == 0:
                return {
                    "success": True,
                    "output": stdout.strip(),
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "output": None,
                    "error": stderr.strip(),
                }
        except TimeoutError as e:
            return {"success": False, "output": None, "error": str(e)}
        finally:
            signal.alarm(0)
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    except Exception as e:
        error_traceback = traceback.format_exc()
        return {
            "success": False,
            "output": None,
            "error": f"Setup error: {str(e)}\n{error_traceback}",
        }


def execute_python_code(code: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Execute Python code in a secure sandbox.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    return _execute_code_in_process(_execute_python_in_subprocess, args=(code, timeout), timeout=timeout)


def _execute_javascript_in_subprocess(code: str, timeout: int) -> Dict[str, Any]:
    """
    Inner function to execute JavaScript code in a subprocess.

    Args:
        code: JavaScript code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    try:
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            return {
                "success": False,
                "output": None,
                "error": "Node.js is not installed or not found in PATH",
            }

        with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as temp_file:
            temp_file_path = temp_file.name

            safe_code = (
                "// Safety wrapper to prevent dangerous operations\n"
                "process.on('uncaughtException', function(err) {\n"
                "  console.error('Uncaught exception:', err.message);\n"
                "  process.exit(1);\n"
                "});\n\n"
                "process.exit = function() { console.error('exit() is disabled'); };\n"
                "process.kill = function() { console.error('kill() is disabled'); };\n"
                "const fs = require('fs');\n"
                "const originalFsReadFile = fs.readFileSync;\n"
                "const originalFsWriteFile = fs.writeFileSync;\n"
                "fs.readFileSync = function() { console.error('fs.readFileSync() is disabled'); return ''; };\n"
                "fs.writeFileSync = function() { console.error('fs.writeFileSync() is disabled'); };\n"
                "const originalRequire = require;\n"
                "global.require = function(module) {\n"
                "  const safeModules = ['assert', 'buffer', 'crypto', 'events', 'path', 'querystring',\n"
                "                      'string_decoder', 'stream', 'timers', 'url', 'util', 'zlib'];\n"
                "  if (safeModules.includes(module)) {\n"
                "    return originalRequire(module);\n"
                "  } else {\n"
                "    console.error(`Requiring module '${module}' is not allowed for security reasons`);\n"
                "    return {};\n"
                "  }\n"
                "};\n\n"
                "try {\n"
                "  " + code.replace("\n", "\n  ") + "\n"
                "} catch (error) {\n"
                "  console.error('Code execution error:', error.message);\n"
                "  process.exitCode = 1;\n"
                "}\n"
            )

            temp_file.write(safe_code.encode("utf-8"))

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout} seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            process = subprocess.Popen(
                [
                    "node",
                    "--no-warnings",
                    "--max-old-space-size=100",
                    temp_file_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                signal.alarm(0)
                return {
                    "success": False,
                    "output": None,
                    "error": f"JavaScript execution timed out after {timeout} seconds (subprocess.TimeoutExpired). Output: {stdout.strip()}, Error: {stderr.strip()}",
                }

            signal.alarm(0)

            if process.returncode == 0:
                return {
                    "success": True,
                    "output": stdout.strip(),
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "output": None,
                    "error": stderr.strip() or f"JavaScript process exited with code {process.returncode}",
                }
        except TimeoutError as e:
            process.kill()
            _, _ = process.communicate()
            return {
                "success": False,
                "output": None,
                "error": f"JavaScript execution timed out after {timeout} seconds (signal.alarm): {str(e)}",
            }
        finally:
            signal.alarm(0)
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        error_traceback = traceback.format_exc()
        return {
            "success": False,
            "output": None,
            "error": f"Setup error: {str(e)}\n{error_traceback}",
        }


def execute_javascript_code(code: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Execute JavaScript code in a secure sandbox.

    Args:
        code: JavaScript code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    return _execute_code_in_process(_execute_javascript_in_subprocess, args=(code, timeout), timeout=timeout)


def compare_outputs(actual: str, expected: str) -> float:
    """
    Compare actual and expected outputs to calculate a similarity score.

    Args:
        actual: Actual output from code execution
        expected: Expected output

    Returns:
        Similarity score between 0.0 and 1.0
    """
    actual_norm = normalize_output(actual)
    expected_norm = normalize_output(expected)

    if actual_norm == expected_norm:
        return 1.0

    if is_numeric(actual_norm) and is_numeric(expected_norm):
        try:
            actual_num = float(actual_norm)
            expected_num = float(expected_norm)

            if expected_num == 0:
                return 1.0 if actual_num == 0 else 0.0

            rel_diff = abs(actual_num - expected_num) / abs(expected_num)
            if rel_diff <= 0.001:
                return 1.0
            elif rel_diff <= 0.01:
                return 0.9
            elif rel_diff <= 0.1:
                return 0.7
            else:
                return max(0.0, 1.0 - min(1.0, rel_diff))
        except (ValueError, TypeError):
            pass

    if (
        actual_norm.startswith("[")
        and actual_norm.endswith("]")
        and expected_norm.startswith("[")
        and expected_norm.endswith("]")
    ):
        try:
            actual_list = json.loads(actual_norm)
            expected_list = json.loads(expected_norm)

            if not actual_list and not expected_list:
                return 1.0

            if not isinstance(actual_list, list) or not isinstance(expected_list, list):
                raise ValueError("Not a list")

            len_similarity = 1.0 - min(
                1.0,
                abs(len(actual_list) - len(expected_list)) / max(1, max(len(actual_list), len(expected_list))),
            )

            items_similarity = 0.0
            if len(actual_list) > 0 and len(expected_list) > 0:
                total_similarity = 0.0
                for exp_item in expected_list:
                    best_match = 0.0
                    for act_item in actual_list:
                        item_similarity = compare_outputs(str(act_item), str(exp_item))
                        best_match = max(best_match, item_similarity)
                    total_similarity += best_match
                items_similarity = total_similarity / len(expected_list)
            return 0.3 * len_similarity + 0.7 * items_similarity
        except (ValueError, json.JSONDecodeError):
            pass

    if "\n" in actual_norm or "\n" in expected_norm:
        actual_lines = actual_norm.strip().split("\n")
        expected_lines = expected_norm.strip().split("\n")

        if not actual_lines and not expected_lines:
            return 1.0

        len_similarity = 1.0 - min(
            1.0,
            abs(len(actual_lines) - len(expected_lines)) / max(1, max(len(actual_lines), len(expected_lines))),
        )

        lines_similarity = 0.0
        common_len = min(len(actual_lines), len(expected_lines))
        if common_len > 0:
            total_similarity = 0.0
            for i in range(common_len):
                line_similarity = string_similarity(actual_lines[i], expected_lines[i])
                total_similarity += line_similarity
            lines_similarity = total_similarity / common_len
        return 0.3 * len_similarity + 0.7 * lines_similarity

    return string_similarity(actual_norm, expected_norm)


def string_similarity(s1: str, s2: str) -> float:
    """
    Calculate string similarity using character-level comparison.

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

    m, n = len(s1), len(s2)
    lcs_length = longest_common_subsequence_length(s1, s2)

    return lcs_length / max(m, n)


def longest_common_subsequence_length(s1: str, s2: str) -> int:
    """
    Calculate the length of the longest common subsequence.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Length of longest common subsequence
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def normalize_output(output: str) -> str:
    """
    Normalize output for comparison.

    Args:
        output: Output string to normalize

    Returns:
        Normalized output string
    """
    normalized = output.strip()
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def is_numeric(value: str) -> bool:
    """
    Check if a string value represents a numeric value.

    Args:
        value: String value to check

    Returns:
        True if the value is numeric, False otherwise
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def noop(*args: Any, **kwargs: Any) -> Any:
    """A no-operation function that returns None."""
    return None


def execute_code_with_e2b(
    code: str,
    language: str = "python",
    timeout: int = 30,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute code within an E2B sandbox.

    Args:
        code: Code to execute
        language: Programming language of the code ("python", "javascript", etc.)
        timeout: Maximum execution time in seconds
        api_key: Optional E2B API key (if not provided, will use E2B_API_KEY env var)

    Returns:
        Dictionary with execution results
    """
    if not _HAS_E2B:
        return {
            "success": False,
            "output": None,
            "error": "E2B package not installed. Install with: pip install e2b",
        }

    try:
        if api_key is None and os.environ.get("E2B_API_KEY") is None:
            return {
                "success": False,
                "output": None,
                "error": "API key is required for E2B execution. Set it using the api_key parameter or E2B_API_KEY environment variable.",
            }

        with Sandbox(api_key=api_key) as sandbox:
            stdout = []
            stderr = []

            def capture_stdout(output):
                if hasattr(output, "line"):
                    stdout.append(output.line)
                else:
                    stdout.append(str(output))

            def capture_stderr(output):
                if hasattr(output, "line"):
                    stderr.append(output.line)
                else:
                    stderr.append(str(output))

            sandbox.on_exit = lambda *args: None  # type: ignore[method-assign, assignment]

            if language.lower() in ["python", "py"]:
                file_path = "/code/script.py"
                cmd = "python3 /code/script.py"
            elif language.lower() in ["javascript", "js"]:
                file_path = "/code/script.js"
                cmd = "node /code/script.js"
            else:
                return {
                    "success": False,
                    "output": None,
                    "error": f"Unsupported language for E2B: {language}",
                }

            try:
                fs_handler = None
                if _E2B_SOURCE == "e2b_code_interpreter":
                    if hasattr(sandbox, "filesystem"):
                        fs_handler = sandbox.filesystem
                elif _E2B_SOURCE == "e2b":
                    if hasattr(sandbox, "_filesystem"):
                        fs_handler = sandbox._filesystem
                    elif hasattr(sandbox, "filesystem"):
                        fs_handler = sandbox.filesystem

                if not fs_handler:
                    return {
                        "success": False,
                        "output": None,
                        "error": "Could not access E2B sandbox filesystem handler.",
                    }

                try:
                    fs_handler.make_dir("/code")
                except Exception:
                    pass

                fs_handler.write(file_path, code)
            except Exception as e:
                return {
                    "success": False,
                    "output": None,
                    "error": f"Failed to write code to sandbox: {str(e)}",
                }

            try:
                result = sandbox.commands.run(
                    cmd,
                    on_stdout=capture_stdout,
                    on_stderr=capture_stderr,
                    timeout=timeout,
                )

                output = "\n".join(stdout)
                error_output = "\n".join(stderr)

                if result.exit_code == 0:
                    return {"success": True, "output": output, "error": None}
                else:
                    return {
                        "success": False,
                        "output": None,
                        "error": f"Process exited with code {result.exit_code}: {error_output}",
                    }

            except Exception as e:
                return {
                    "success": False,
                    "output": None,
                    "error": f"Execution error: {str(e)}",
                }

    except Exception as e:
        error_traceback = traceback.format_exc()
        return {
            "success": False,
            "output": None,
            "error": f"E2B setup error: {str(e)}\n{error_traceback}",
        }


@reward_function
def e2b_code_execution_reward(
    messages: List[Message],
    ground_truth: Optional[str] = None,
    language: str = "python",
    timeout: int = 30,
    api_key: Optional[str] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate code correctness by executing it in E2B sandbox and comparing the output.

    E2B provides a secure, cloud-based sandbox for executing code safely.

    Args:
        messages: List of conversation messages. The last message is assumed to be the
                  assistant's response containing the code.
        ground_truth: Expected output string from code execution. This corresponds to
                      the `expected_output_str` in the previous signature.
        language: Programming language of the code ("python", "javascript", etc.)
        timeout: Maximum execution time in seconds.
        api_key: Optional E2B API key (if not provided, will use E2B_API_KEY env var).
        **kwargs: Additional keyword arguments.

        Returns:
        EvaluateResult with score and metrics.
    """
    if not _HAS_E2B:
        return EvaluateResult(
            score=0.0,
            reason="E2B package not installed.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason="E2B package not installed. Install with: pip install e2b",
                    is_score_valid=False,
                )
            },
        )

    if api_key is None and os.environ.get("E2B_API_KEY") is None:
        return EvaluateResult(
            score=0.0,
            reason="E2B API key is required.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason="E2B API key is required. Set the E2B_API_KEY environment variable or provide api_key parameter.",
                    is_score_valid=False,
                )
            },
        )

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

    last_content = messages[-1].content
    response_content = (
        last_content if isinstance(last_content, str) else "".join([p.text for p in (last_content or [])])
    )
    expected_output_str = ground_truth

    code_blocks = extract_code_blocks(response_content, language)

    if not code_blocks:
        return EvaluateResult(
            score=0.0,
            reason=f"No {language} code blocks found in model's response.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason=f"No {language} code blocks found in model's response.",
                    is_score_valid=False,
                )
            },
        )

    code = code_blocks[0]["code"]

    metrics["extracted_code"] = MetricResult(
        score=0.0,
        reason=f"Extracted code:\n```{language}\n{code}\n```",
        is_score_valid=True,
    )

    if expected_output_str:
        metrics["expected_output"] = MetricResult(
            score=0.0,
            reason=f"Expected output:\n{expected_output_str}",
            is_score_valid=True,
        )

    execution_result = execute_code_with_e2b(code=code, language=language, timeout=timeout, api_key=api_key)

    if execution_result["success"]:
        output = execution_result["output"]

        metrics["execution_result"] = MetricResult(
            score=1.0,
            reason=f"Code executed successfully in E2B sandbox with output:\n{output}",
            is_score_valid=True,
        )

        if expected_output_str:
            similarity = compare_outputs(output, expected_output_str)
            match_reason = (
                f"Output similarity: {similarity:.2f}\n\nExpected:\n{expected_output_str}\n\nActual:\n{output}"
            )

            metrics["output_match"] = MetricResult(
                score=similarity, reason=match_reason, is_score_valid=similarity == 1.0
            )
            final_reason = f"E2B execution successful. Output similarity: {similarity:.2f}."
            return EvaluateResult(score=similarity, reason=final_reason, metrics=metrics)

        final_reason = "E2B execution successful. No expected output to compare."
        return EvaluateResult(score=1.0, reason=final_reason, metrics=metrics)
    else:
        error = execution_result["error"]

        metrics["execution_result"] = MetricResult(
            score=0.0,
            reason=f"Code execution failed in E2B sandbox with error:\n{error}",
            is_score_valid=False,
        )
        final_reason = f"E2B code execution failed: {error}"
        return EvaluateResult(score=0.0, reason=final_reason, metrics=metrics)


@reward_function
def fractional_code_reward(
    messages: List[Message],
    ground_truth: Union[Optional[str], Optional[List[Dict[str, Any]]]],
    language: str = "python",
    timeout: int = 30,
    environment: str = "local",
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Execute code and return the exact pass rate as a score between 0 and 1.

    Unlike the binary code reward, this function returns the actual score representing
    how closely the code output matches the expected output or how many test cases pass.

    Args:
        messages: List of conversation messages. The last message is assumed to be the
                  assistant's response containing the code.
        ground_truth: Expected output string from code execution, OR a list of test cases.
                      If a string, it's direct output comparison.
                      If a list of dicts, each dict is a test case with "input" and "expected_output".
        language: Programming language of the code ("python", "javascript", etc.).
        timeout: Maximum execution time in seconds.
        environment: Environment to run the code in ("local" or "e2b").
        api_key: Optional E2B API key (if using e2b environment).
        **kwargs: Additional keyword arguments (e.g., function_to_call for _run_test_cases).

    Returns:
        EvaluateResult with score between 0 and 1 representing the exact pass rate.
    """
    metrics_strings: Dict[str, str] = {}

    if (
        not messages
        or not isinstance(messages[-1], Message)
        or messages[-1].role != "assistant"
        or messages[-1].content is None
    ):
        return EvaluateResult(
            score=0.0,
            reason="Invalid or missing assistant response in messages for fractional code reward.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Last message not a valid assistant response.",
                )
            },
        )

    response_content = messages[-1].content

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
                reason="Invalid ground_truth format: expected string or list of test case dicts.",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Invalid ground_truth format.",
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

    # Normalize content to string; Message.content may be str or list of content parts
    _last_content = response_content
    response_content_str = (
        _last_content
        if isinstance(_last_content, str)
        else "".join([getattr(p, "text", "") for p in (_last_content or [])])
    )

    code_blocks = extract_code_blocks(response_content_str, language)

    if not code_blocks:
        return EvaluateResult(
            score=0.0,
            reason=f"No {language} code blocks found in model's response for fractional code reward.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason=f"No {language} code blocks found in model's response.",
                    is_score_valid=False,
                )
            },
        )

    code = code_blocks[0]["code"]

    metrics_strings["extracted_code"] = f"Extracted code:\n```{language}\n{code}\n```"

    if expected_output_str_from_gt and not test_cases_from_gt:
        metrics_strings["expected_output"] = f"Expected output:\n{expected_output_str_from_gt}"

    if test_cases_from_gt:
        return _run_test_cases(
            code=code,
            language=language,
            test_cases=test_cases_from_gt,
            timeout=timeout,
            environment=environment,
            api_key=api_key,
            **kwargs,
        )

    execution_result: Dict[str, Any]
    if environment.lower() == "e2b":
        if not _HAS_E2B:
            return EvaluateResult(
                score=0.0,
                reason="E2B package not installed for fractional code reward.",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        reason="E2B package not installed. Install with: pip install e2b",
                        is_score_valid=False,
                    )
                },
            )
        execution_result = execute_code_with_e2b(code=code, language=language, timeout=timeout, api_key=api_key)
    else:
        if language.lower() == "python":
            execution_result = execute_python_code(code, timeout)
        elif language.lower() in ["javascript", "js"]:
            execution_result = execute_javascript_code(code, timeout)
        else:
            final_metrics_on_error: Dict[str, MetricResult] = {
                k: MetricResult(score=0.0, reason=v, is_score_valid=(k == "extracted_code"))
                for k, v in metrics_strings.items()
            }
            final_metrics_on_error["error"] = MetricResult(
                score=0.0,
                reason=f"Unsupported language: {language}",
                is_score_valid=False,
            )
            return EvaluateResult(
                score=0.0,
                reason=f"Unsupported language for fractional code reward: {language}",
                metrics=final_metrics_on_error,
            )

    metric_results: Dict[str, MetricResult] = {
        k: MetricResult(
            score=0.0,
            reason=v,
            is_score_valid=(
                k == "extracted_code" or (k == "expected_output" and expected_output_str_from_gt is not None)
            ),
        )
        for k, v in metrics_strings.items()
    }

    if execution_result["success"]:
        output = execution_result["output"]
        metric_results["execution_result"] = MetricResult(
            score=1.0,
            reason=f"Code executed successfully with output:\n{output}",
            is_score_valid=True,
        )

        if expected_output_str_from_gt:
            similarity = compare_outputs(output, expected_output_str_from_gt)
            match_reason = (
                f"Output similarity: {similarity:.2f}\n\nExpected:\n{expected_output_str_from_gt}\n\nActual:\n{output}"
            )
            metric_results["output_match"] = MetricResult(
                score=similarity, reason=match_reason, is_score_valid=similarity == 1.0
            )
            final_reason = f"Fractional code execution successful. Output similarity: {similarity:.2f}."
            return EvaluateResult(score=similarity, reason=final_reason, metrics=metric_results)
        else:
            final_reason = "Fractional code execution successful. No expected output string to compare."
            return EvaluateResult(score=1.0, reason=final_reason, metrics=metric_results)
    else:
        error = execution_result["error"]
        metric_results["execution_result"] = MetricResult(
            score=0.0,
            reason=f"Code execution failed with error:\n{error}",
            is_score_valid=False,
        )
        final_reason = f"Fractional code execution failed: {error}"
        return EvaluateResult(score=0.0, reason=final_reason, metrics=metric_results)


def _run_test_cases(
    code: str,
    language: str,
    test_cases: List[Dict[str, Any]],
    timeout: int,
    environment: str,
    api_key: Optional[str] = None,
    function_to_call: Optional[str] = None,
    prompt_for_name_extraction: Optional[str] = None,  # Not used yet, but for future use
    **kwargs: Any,  # Keep kwargs for flexibility, though function_to_call is now explicit
) -> EvaluateResult:  # Changed return type hint to match actual returns
    """
    Run code against multiple test cases and return the fraction of passing tests.
    Can optionally call a specific function if `function_to_call` is provided.

    Args:
        code: The code to execute
        language: Programming language of the code
        test_cases: List of test cases with input and expected output
        timeout: Maximum execution time in seconds
        environment: Environment to run the code in ("local" or "e2b")
        api_key: Optional E2B API key (if using e2b environment)

    Returns:
        EvaluateResult with score representing the fraction of passing tests
    """
    metrics: Dict[str, Any] = {}
    results = []
    passed = 0
    total = len(test_cases)

    if total == 0:
        return EvaluateResult(
            score=0.0,
            reason="No test cases provided",
            metrics={"error": MetricResult(score=0.0, reason="No test cases provided", is_score_valid=False)},
        )

    if language.lower() in ["python", "py"]:
        if function_to_call:

            def prepare_test_code(user_code: str, test_input_str: str, func_name: Optional[str]) -> str:
                import ast
                import json

                def refine_evaluated_value(val: Any) -> Any:
                    if isinstance(val, str):
                        stripped_val = val.strip()
                        if stripped_val.startswith(("[", "{")):
                            try:
                                return json.loads(stripped_val)
                            except json.JSONDecodeError:
                                return val
                        else:
                            try:
                                if "." in stripped_val or "e" in stripped_val.lower() or "E" in stripped_val:
                                    return float(stripped_val)
                                else:
                                    return int(stripped_val)
                            except ValueError:
                                return val
                    return val

                parsed_args = []
                args_str_stripped = test_input_str.strip()

                if not args_str_stripped:
                    pass
                else:
                    parsed_as_single_arg = False
                    try:
                        val_from_json = json.loads(args_str_stripped)
                        parsed_args.append(refine_evaluated_value(val_from_json))
                        parsed_as_single_arg = True
                    except json.JSONDecodeError:
                        try:
                            val_from_ast = ast.literal_eval(args_str_stripped)
                            parsed_args.append(refine_evaluated_value(val_from_ast))
                            parsed_as_single_arg = True
                        except (ValueError, SyntaxError):
                            pass

                    if not parsed_as_single_arg:
                        try:
                            arg_parts = shlex.split(args_str_stripped)
                        except ValueError:
                            arg_parts = [args_str_stripped]

                        for part_str in arg_parts:
                            try:
                                val_from_part_ast = ast.literal_eval(part_str)
                                parsed_args.append(refine_evaluated_value(val_from_part_ast))
                            except (ValueError, SyntaxError):
                                parsed_args.append(refine_evaluated_value(part_str))

                args_repr = ", ".join(map(repr, parsed_args))

                return f"""import sys
import json
import traceback

{user_code}

try:
    result = {func_name}({args_repr})
    print(repr(result))
except Exception as e:
    import traceback
    print(f'Error calling function {func_name}: {{traceback.format_exc()}}', file=sys.stderr)
    import sys
    sys.exit(1)
"""

        else:

            def prepare_test_code(user_code: str, test_input_str: str, func_name: Optional[str]) -> str:
                escaped_test_input = json.dumps(test_input_str)[1:-1].replace("'''", "'\\''\\''\\''")
                return f"""import sys
from io import StringIO

original_stdout = sys.stdout
sys.stdout = captured_stdout = StringIO()
sys.stdin = StringIO('''{escaped_test_input}''')

try:
    exec({repr(user_code)})
except Exception as e:
    import traceback
    print(f'Error executing script: {{traceback.format_exc()}}', file=sys.stderr)
    import sys
    sys.exit(1)

sys.stdout = original_stdout
print(captured_stdout.getvalue(), end='')
"""

    elif language.lower() in ["javascript", "js"]:
        if function_to_call:

            def prepare_test_code(user_code: str, test_input_str: str, func_name: Optional[str]) -> str:
                args_str = test_input_str.strip()
                parsed_args_js = []
                if args_str:
                    for arg in args_str.split():
                        if arg.isdigit() or (arg.startswith("-") and arg[1:].isdigit()):
                            parsed_args_js.append(arg)
                        elif "." in arg and all(
                            c.isdigit() or c == "." or (i == 0 and c == "-") for i, c in enumerate(arg)
                        ):
                            try:
                                float(arg)
                                parsed_args_js.append(arg)
                            except ValueError:
                                parsed_args_js.append(json.dumps(arg))
                        else:
                            parsed_args_js.append(json.dumps(arg))

                args_js_repr = ", ".join(parsed_args_js)
                return f"""{user_code}

try {{
    const result = {func_name}({args_js_repr});
    console.log(JSON.stringify(result));
}} catch (error) {{
    console.error(`Error calling function {func_name}:`, error);
    process.exitCode = 1;
}}
"""

        else:

            def prepare_test_code(user_code: str, test_input_str: str, func_name: Optional[str]) -> str:
                input_lines = test_input_str.strip().split("\n")
                input_setup = "const inputs = " + json.dumps(input_lines) + ";\n"
                input_setup += "let inputIndex = 0;\n"
                input_setup += "const readline = () => inputs[inputIndex++];\n"
                return f"""const originalLog = console.log;
let output = '';
console.log = function(...args) {{
  output += args.map(String).join(' ') + '\\n';
}};

{input_setup}

try {{
    {user_code}
}} catch (error) {{
    console.error('Error executing script:', error);
    process.exitCode = 1;
}}

console.log = originalLog;
process.stdout.write(output);
"""

    else:
        return EvaluateResult(
            score=0.0,
            reason=f"Unsupported language for test cases: {language}",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason=f"Unsupported language for test cases: {language}",
                    is_score_valid=False,
                )
            },
        )

    for i, test_case in enumerate(test_cases):
        test_input = test_case.get("input", "")
        expected = test_case.get("expected_output", "")

        test_code_prepared = prepare_test_code(code, test_input, function_to_call)

        if environment.lower() == "e2b":
            if not _HAS_E2B:
                return EvaluateResult(
                    score=0.0,
                    reason="E2B package not installed for test cases.",
                    metrics={
                        "error": MetricResult(
                            score=0.0,
                            reason="E2B package not installed. Install with: pip install e2b",
                            is_score_valid=False,
                        )
                    },
                )

            execution_result = execute_code_with_e2b(
                code=test_code_prepared,
                language=language,
                timeout=timeout,
                api_key=api_key,
            )
        else:
            if language.lower() in ["python", "py"]:
                execution_result = execute_python_code(test_code_prepared, timeout)
            elif language.lower() in ["javascript", "js"]:
                execution_result = execute_javascript_code(test_code_prepared, timeout)
            else:
                return EvaluateResult(
                    score=0.0,
                    reason=f"Unsupported language for local execution: {language}",
                    metrics={
                        "error": MetricResult(
                            score=0.0,
                            reason=f"Unsupported language for local execution: {language}",
                            is_score_valid=False,
                        )
                    },
                )

        test_result = {
            "test_number": i + 1,
            "input": test_input,
            "expected_output": expected,
            "passed": False,
            "details": "",
        }

        if execution_result["success"]:
            output = execution_result["output"]
            normalized_output = normalize_output(output)
            normalized_expected = normalize_output(expected)

            expected_repr = repr(expected) if function_to_call and language.lower() in ["python", "py"] else None
            normalized_expected_repr = normalize_output(expected_repr) if expected_repr else None

            is_pass = normalized_output == normalized_expected
            if not is_pass and normalized_expected_repr:
                is_pass = normalized_output == normalized_expected_repr

            test_result["passed"] = is_pass
            test_result["actual_output"] = output
            test_result["normalized_actual"] = normalized_output
            test_result["normalized_expected"] = normalized_expected
            test_result["details"] = f"Passed: {is_pass}"

            if test_result["passed"]:
                passed += 1
        else:
            test_result["error"] = execution_result["error"]
            test_result["details"] = f"Error: {execution_result['error']}"

        results.append(test_result)

    score = passed / total if total > 0 else 0.0

    if isinstance(results, list):
        metrics["test_results"] = results
    else:
        metrics["test_results"] = [{"error": "Invalid results format"}]
    metrics["pass_rate"] = f"{passed}/{total} tests passed ({score:.2%})"

    final_metrics: Dict[str, MetricResult] = {}
    for key, value in metrics.items():
        if key == "test_results":
            final_metrics[key] = MetricResult(
                score=score,
                reason=json.dumps(value, indent=2),
                is_score_valid=score == 1.0,
            )
        elif key == "pass_rate":
            final_metrics[key] = MetricResult(
                score=score,
                reason=str(value),
                is_score_valid=score == 1.0,
            )
        elif isinstance(value, MetricResult):
            final_metrics[key] = value
        elif isinstance(value, str):
            final_metrics[key] = MetricResult(score=0.0, reason=value, is_score_valid=False)

    return EvaluateResult(score=score, reason=f"{passed}/{total} tests passed.", metrics=final_metrics)


def reliability_guard(maximum_memory_bytes: Optional[int] = None) -> None:
    """
    Disable various destructive functions and prevent the generated code
    from interfering with the test system.

    This sets resource limits and disables various system calls that could
    be used to interfere with the testing environment.

    Args:
        maximum_memory_bytes: Maximum memory allocation allowed in bytes (optional)

    Warning:
        This function is NOT a security sandbox. Untrusted code should not be
        blindly executed outside of a proper sandbox environment.
    """
    if maximum_memory_bytes is not None:
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
            resource.setrlimit(
                resource.RLIMIT_DATA,
                (maximum_memory_bytes, maximum_memory_bytes),
            )
            resource.setrlimit(
                resource.RLIMIT_STACK,
                (maximum_memory_bytes, maximum_memory_bytes),
            )

    faulthandler.disable()

    import builtins

    builtins.exit = noop  # type: ignore
    builtins.quit = noop  # type: ignore

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = noop  # type: ignore
    os.system = noop  # type: ignore
    os.putenv = noop  # type: ignore
    os.remove = noop  # type: ignore
    os.removedirs = noop  # type: ignore
    os.rmdir = noop  # type: ignore
    os.fchdir = noop  # type: ignore
    os.setuid = noop  # type: ignore
    os.fork = noop  # type: ignore
    os.forkpty = noop  # type: ignore
    os.killpg = noop  # type: ignore
    os.rename = noop  # type: ignore
    os.renames = noop  # type: ignore
    os.truncate = noop  # type: ignore
    os.replace = noop  # type: ignore
    os.unlink = noop  # type: ignore
    os.fchmod = noop  # type: ignore
    os.fchown = noop  # type: ignore
    os.chmod = noop  # type: ignore
    os.chown = noop  # type: ignore
    os.chroot = noop  # type: ignore

    if hasattr(os, "lchflags"):
        os.lchflags = noop  # type: ignore
    if hasattr(os, "lchmod"):
        os.lchmod = noop  # type: ignore
    if hasattr(os, "lchown"):
        os.lchown = noop  # type: ignore

    import shutil

    shutil.rmtree = noop  # type: ignore
    shutil.move = noop  # type: ignore
    shutil.chown = noop  # type: ignore

    class EmptyModule:
        def __getattr__(self, name: str) -> Any:
            return noop

    for mod_name in ["ipdb", "joblib", "psutil", "tkinter"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = EmptyModule()  # type: ignore


class Capturing(list):
    """
    Context manager for capturing stdout output.

    This class captures all output to stdout and stores it in a list,
    allowing for the examination of output from executed code.
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda: None
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout
