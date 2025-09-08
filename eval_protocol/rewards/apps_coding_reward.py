import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional

from eval_protocol.models import EvaluateResult, Message, MetricResult
from eval_protocol.reward_function import reward_function

# Import the new execution utility
from .apps_execution_utils import check_correctness

logger = logging.getLogger(__name__)


# Helper function to extract code from the assistant's response
def _extract_python_code(response_content: str) -> Optional[str]:
    """
    Extracts Python code from a string.
    Tries to find code within ```python ... ``` or ``` ... ``` blocks.
    If not found, tries to find the first 'def ' and takes from there.
    It also attempts to remove <think>...</think> blocks first.
    """
    # Attempt to remove <think>...</think> blocks first
    cleaned_content = re.sub(r"<think>[\s\S]*?</think>", "", response_content, flags=re.IGNORECASE).strip()
    if cleaned_content != response_content.strip():  # Log if <think> block was actually removed
        logger.debug(
            "Removed <think>...</think> block(s). Content after removal (stripped): "
            + repr(cleaned_content[:200])
            + "..."
        )
        if not cleaned_content:  # If stripping results in empty string
            logger.warning("Content became empty after removing <think> block and stripping.")
            return None
    else:  # No <think> block found or removing it resulted in the same stripped string
        cleaned_content = response_content.strip()  # Ensure we work with stripped content if no <think> block

    # Try to find ```python ... ``` in the cleaned content
    match = re.search(r"```python\s*(.*?)\s*```", cleaned_content, re.DOTALL)
    if match:
        logger.debug("Extracted code using ```python ... ``` block.")
        return match.group(1).strip()

    # Try to find ``` ... ``` in the cleaned content
    match = re.search(r"```\s*(.*?)\s*```", cleaned_content, re.DOTALL)
    if match:
        logger.debug("Extracted code using ``` ... ``` block.")
        return match.group(1).strip()

    # Try to find the first 'def ' in the cleaned content
    def_index = cleaned_content.find("def ")
    if def_index != -1:
        logger.debug("Extracted code starting from the first 'def '.")
        return cleaned_content[def_index:].strip()

    # If no specific markers, return the cleaned content stripped.
    # The warning about parsing the entire response if no markers are found is now more accurate.
    if not match and def_index == -1:  # if no ``` or def was found
        # Log if we are falling back to the full (cleaned) content
        logger.warning(
            "No specific code markers (```python, ```, def) found. Attempting to parse content after <think> removal (if any)."
        )
    return cleaned_content  # This is already stripped if <think> was removed, or original stripped content


@reward_function
def evaluate_apps_solution(messages: List[Message], ground_truth: Optional[str], **kwargs) -> EvaluateResult:
    """
    Evaluates a code solution for the APPS dataset.
    Extracts Python code from the last message and checks for basic Python code parsability.
    The ground_truth is expected to be a JSON string containing test cases,
    but it's not used in this initial simplified version.
    """
    if not messages:
        return EvaluateResult(
            score=0.0,
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason="No messages provided for evaluation.",
                    is_score_valid=False,
                )
            },
            reason="No messages provided.",
        )

    raw_solution_content = messages[-1].content if isinstance(messages[-1].content, str) else ""
    code_solution = _extract_python_code(raw_solution_content)

    if not code_solution or not code_solution.strip():
        # Log the raw content if extraction resulted in empty/None
        if raw_solution_content:
            logger.warning(
                f"Code extraction resulted in empty solution. Raw content was: '{raw_solution_content[:200]}...'"
            )
        else:
            logger.warning("Code extraction resulted in empty solution. Raw content was empty.")
        return EvaluateResult(
            score=0.0,
            metrics={
                "parsability": MetricResult(
                    score=0.0,
                    reason="Empty code solution after extraction.",
                    is_score_valid=True,
                ),
                "error": MetricResult(
                    score=0.0,
                    reason="Empty code solution after extraction.",
                    is_score_valid=False,
                ),
            },
            reason="The provided code solution was empty after extraction.",
        )

    logger.debug(f"Extracted code for execution: \n---\n{code_solution[:500]}...\n---")

    # Default score and reason
    score = 0.0
    reason_msg = "Evaluation did not complete successfully."
    metrics: Dict[str, MetricResult] = {}

    in_outs: Optional[Dict[str, Any]] = None
    if isinstance(ground_truth, str):
        # Explicitly assign to a str-typed variable after check for Mypy
        gt_str: str = ground_truth
        logger.debug(f"Raw ground_truth string for sample: {gt_str[:1000]}")
        try:
            in_outs = json.loads(gt_str)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse ground_truth JSON string: {e}. GT (first 200 chars): {(gt_str or '')[:200]}"
            )
            return EvaluateResult(
                score=0.0,
                reason=f"Ground_truth JSONDecodeError: {e}",
                metrics={
                    "error": MetricResult(
                        score=0.0,
                        reason=f"Ground_truth JSONDecodeError: {e}",
                        is_score_valid=False,
                    )
                },
            )
    elif isinstance(ground_truth, dict):
        logger.debug(f"ground_truth is already a dict: {str(ground_truth)[:1000]}")
        in_outs = ground_truth  # It's already parsed (likely by JSONL loader)
    else:
        logger.error(
            f"ground_truth is neither a string nor a dict. Type: {type(ground_truth)}. Value (first 200 chars): {str(ground_truth)[:200]}"
        )
        return EvaluateResult(
            score=0.0,
            reason="Invalid ground_truth type.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason=f"Invalid ground_truth type: {type(ground_truth)}",
                    is_score_valid=False,
                )
            },
        )

    if not isinstance(in_outs, dict) or "inputs" not in in_outs or "outputs" not in in_outs:
        logger.error(
            f"Parsed ground_truth is not in the expected format (dict with 'inputs' and 'outputs'). Parsed: {str(in_outs)[:200]}"
        )
        return EvaluateResult(
            score=0.0,
            reason="Invalid ground_truth structure after parsing.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    reason="Invalid ground_truth structure after parsing.",
                    is_score_valid=False,
                )
            },
        )

    # Log the parsed in_outs and specifically check for fn_name
    fn_name_from_gt = in_outs.get("fn_name")
    if not fn_name_from_gt:
        logger.warning("fn_name not found in ground_truth dict, will rely on system prompt for main() or full script.")
        # fn_name_from_gt will remain None, forcing testing_util to use standard_input path.
    logger.info(
        f"Using fn_name from ground_truth (if present): {fn_name_from_gt}. Parsed in_outs (first 500 chars of dump): {json.dumps(in_outs)[:500]}"
    )

    # Default timeout for check_correctness, can be made configurable via kwargs if needed
    timeout = kwargs.get("execution_timeout", 10)
    debug_execution = True  # For now, enable debug prints from check_correctness/run_test

    # Construct the wrapper script
    # Standard imports often used in competitive programming / APPS
    standard_imports = """
import traceback, sys, json, ast, collections, copy, datetime, functools, heapq, io, itertools, math, operator, random, re, string, statistics, typing
sys.setrecursionlimit(6*10**5)
"""
    # Wrapper to call the user's function (fn_name_from_gt) and handle I/O
    # This wrapper will be executed by testing_util's standard_input path.
    # It expects testing_util to provide the actual test case input via sys.stdin.
    # It will print the function's result to sys.stdout, which testing_util will capture.

    # Determine how arguments should be passed based on fn_name_from_gt
    # If 'main', assume it handles its own stdin. Otherwise, parse stdin as args.
    # The testing_util.py's standard_input path provides the *entire* input for one test case as a single string to stdin.

    # If fn_name_from_gt is 'main', the model's code should contain 'def main():' which reads stdin.
    # If fn_name_from_gt is specific, the model's code is 'def specific_name(...):'.
    # The wrapper needs to call this specific_name.
    # The APPS 'inputs' are usually strings, where each string is the *entire* stdin for one run of the target function.
    # Or, for call-based, 'inputs' is a list of lists of arguments.
    # Since we are forcing standard_input path for testing_util by setting fn_name=None in in_outs_for_check,
    # testing_util will provide the content of in_outs["inputs"][test_case_idx] to stdin.

    # The generated code_solution might be a full script or just a function.
    # If it's just a function, the wrapper needs to call it.
    # If it's a full script with if __name__ == "__main__":, that will be handled by testing_util's stdio path.

    # Let's assume the new system prompt encourages `def main(): ...`
    # The `testing_util.py` standard input path wraps the solution in `def code(): ... solution ...`
    # and then calls `code()`. If `solution` is `def main(): ...`, then `code()` just defines `main`.
    # We need `main()` to be called.
    # So, the `code_solution` itself should end with `if __name__ == "__main__": main()` or just `main()`.
    # The system prompt now asks for `main()`. Let's assume the model provides it and might call it.

    # Forcing testing_util to use its standard_input path by ensuring fn_name is None in the dict passed to it.
    # The actual function name logic is now handled by the system prompt guiding the model.
    # The `in_outs` dict passed to check_correctness will have its 'fn_name' key removed or set to None
    # to ensure testing_util.py uses its standard input execution path.
    # The `generation` argument to check_correctness will be the `code_solution`.
    # `testing_util.py` will wrap this `code_solution` in `def code(): ...` and call `code()`.
    # If `code_solution` is `def main(): ... ; main()`, it should work.
    # If `code_solution` is just `def main(): ...`, it won't work.
    # The new system prompt is: "Structure your solution within a main() function. ... main() should handle it. ... main() should print..."
    # This implies the model should provide a callable main that does everything.

    # Let's simplify: assume the model provides a runnable script (e.g. with main() called at the end, or top-level code)
    # due to the new system prompt. We will rely on testing_util's standard_input path.
    # We need to ensure `fn_name` is NOT in `in_outs` when calling `check_correctness`.

    in_outs_for_check = in_outs.copy()  # Use a copy to modify for check_correctness
    if "fn_name" in in_outs_for_check:
        # Remove fn_name to force testing_util's standard_input path,
        # as our system prompt now asks for a main() that handles IO.
        # The generated code itself should be a runnable script.
        del in_outs_for_check["fn_name"]
        logger.info("Removed 'fn_name' from in_outs for check_correctness to use standard_input path.")

    final_code_to_execute = code_solution  # The model's full response (after extraction)

    try:
        results_list, exec_metadata_list = check_correctness(
            in_outs=in_outs_for_check,  # This now has no 'fn_name'
            generation=final_code_to_execute,
            timeout=timeout,
            debug=debug_execution,
        )

        # Process results_list
        if not results_list:  # Should not happen if check_correctness returns properly
            reason_msg = "Execution utility returned no results."
            logger.error(reason_msg)
            metrics["execution_error"] = MetricResult(score=0.0, reason=reason_msg, is_score_valid=False)
        else:
            # Check for error codes (-1 for runtime/timeout, -2 for compilation error)
            # These error codes are per test case as per testing_util.py's results.append()
            # However, check_correctness's _temp_run appends a list, so results_list is a list of lists.
            # The outer list from check_correctness usually has one item: the list of results from run_test.

            actual_results = results_list  # results_list from check_correctness is already the list of actual outcomes

            num_tests = len(actual_results)
            if num_tests == 0:  # Should ideally not happen if in_outs['inputs'] is non-empty
                reason_msg = "No test cases were effectively run or reported by execution utility."
                logger.warning(reason_msg)
                # Score remains 0.0
            else:
                passed_count = sum(1 for res in actual_results if res is True)
                score = float(passed_count) / num_tests
                reason_msg = f"Passed {passed_count}/{num_tests} test cases."
                logger.info(f"Execution result: {reason_msg}")

            metrics["pass_rate"] = MetricResult(score=score, reason=f"{passed_count}/{num_tests}")
            metrics["raw_results"] = MetricResult(
                score=0.0, reason=json.dumps(actual_results), is_score_valid=False
            )  # Store raw results

        # Process metadata
        # exec_metadata_list is a list of dicts. If it's a single dict (e.g. compilation error), wrap it.
        # The check_correctness in apps_execution_utils.py should return a list of metadata dicts.
        if exec_metadata_list:
            # If there's a single metadata entry that contains a significant error (like compilation)
            # it might apply to the whole attempt.
            # For now, just log it or add to metrics.
            # The original prime_code's compute_score returns a list of metadata.
            # We'll store it as a JSON string for simplicity in metrics.
            # If only one metadata dict, it might be a global error (e.g. compilation)
            if len(exec_metadata_list) == 1 and exec_metadata_list[0].get("error"):
                reason_msg += f" Execution Error: {exec_metadata_list[0]['error']}"
                metrics["execution_error_details"] = MetricResult(
                    score=0.0,
                    reason=json.dumps(exec_metadata_list[0]),
                    is_score_valid=False,
                )
            elif exec_metadata_list:  # It's not a global error, but there's metadata (e.g., for Wrong Answer)
                metrics["execution_metadata"] = MetricResult(
                    score=0.0,
                    reason=json.dumps(exec_metadata_list),
                    is_score_valid=False,
                )
                # If it's a "Wrong Answer" and score is 0, enhance the reason_msg
                if score == 0.0 and exec_metadata_list[0].get("error_message") == "Wrong Answer":
                    first_fail_meta = exec_metadata_list[0]
                    reason_msg += (
                        f". First fail details: Inputs: {first_fail_meta.get('inputs', 'N/A')}, "
                        f"Expected: {first_fail_meta.get('expected', 'N/A')}, "
                        f"Got: {first_fail_meta.get('output', 'N/A')}"
                    )

        # If score is 0 and there was an error in metadata, reflect it in reason_msg
        # This condition might be redundant now due to the above, or could be a fallback.
        if score == 0.0 and metrics.get("execution_error_details") and "Execution Error" not in reason_msg:
            pass  # reason_msg might already be updated by global error or Wrong Answer details.

    except Exception as e:
        score = 0.0  # Ensure score is 0 on any unexpected error in this block
        reason_msg = f"Error during code execution or result processing: {type(e).__name__}: {e}"
        logger.error(reason_msg, exc_info=True)
        metrics["evaluation_error"] = MetricResult(score=0.0, reason=reason_msg, is_score_valid=False)

    return EvaluateResult(score=score, metrics=metrics, reason=reason_msg)
