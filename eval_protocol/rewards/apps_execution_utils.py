# Copyright 2024 PRIME team and/or its affiliates (Adapted for reward-kit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import multiprocessing
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

# Adapted import to point to our local apps_testing_util.py
from .apps_testing_util import run_test

# Note: The original file had a compute_score function.
# We are primarily interested in check_correctness and its helper _temp_run.
# The main reward function in apps_coding_reward.py will call check_correctness.


def _temp_run(
    sample: Dict[str, Any],
    generation: str,
    debug: bool,
    result_list: list,
    metadata_list: list,
    timeout: int,
):
    """
    Helper function to run a single test in a separate process context.
    Manages stdout/stderr redirection and captures results/metadata.
    """
    # Redirect stdout/stderr to prevent interference and capture output if needed by run_test
    # Note: run_test itself might also capture stdout for standard_input type problems.
    # This top-level redirection ensures the process itself doesn't pollute console.
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    # Temporarily disable stdout/stderr redirection to see debug prints from run_test
    # sys.stdout = open(os.devnull, "w")
    # sys.stderr = open(os.devnull, "w")
    print("[_temp_run] Executing run_test for sample. Debug prints from run_test should be visible.")

    try:
        res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
        result_list.append(res)
        metadata_list.append(metadata)
    except Exception:
        # This catch-all is for unexpected errors within _temp_run itself or run_test if it raises
        # instead of returning error codes in `res`.
        # run_test is designed to return error codes like -1 or -2 in `res` for failures.
        num_inputs = len(sample.get("inputs", []))
        tb_str = traceback.format_exc()
        print(f"[_temp_run] Exception caught: {tb_str}")
        result_list.append([-1] * num_inputs if num_inputs > 0 else [-1])  # Mark all as error
        metadata_list.append({"error": "Exception in _temp_run/run_test", "traceback": tb_str})
    finally:
        # Restore stdout/stderr
        # if sys.stdout is not original_stdout and hasattr(sys.stdout, 'close'): # Check if it was replaced and closable
        #     sys.stdout.close()
        # if sys.stderr is not original_stderr and hasattr(sys.stderr, 'close'): # Check if it was replaced and closable
        #     sys.stderr.close()
        sys.stdout = original_stdout  # Always restore
        sys.stderr = original_stderr  # Always restore


def check_correctness(
    in_outs: Optional[dict], generation: str, timeout: int = 10, debug: bool = True
) -> tuple[List[Any], List[Dict[str, Any]]]:
    """
    Checks correctness of code generation with a global timeout using multiprocessing.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`.
    Args:
        in_outs: Dictionary with "inputs" and "outputs" lists, and optionally "fn_name".
        generation: The code string to test.
        timeout: Timeout in seconds for each test case execution within run_test,
                 and also for the overall process.
        debug: Debug flag passed to run_test.

    Returns:
        A tuple containing:
        - A list of results (e.g., booleans for pass/fail, or error codes).
        - A list of metadata dictionaries corresponding to each result.
    """
    if not in_outs or "inputs" not in in_outs or not isinstance(in_outs["inputs"], list):
        # Handle cases where in_outs might be None or malformed early
        return [-1], [{"error": "Invalid or missing in_outs structure"}]

    manager = multiprocessing.Manager()
    result_proxy = manager.list()  # Using proxy list for multiprocessing
    metadata_proxy = manager.list()  # Using proxy list for multiprocessing

    process = multiprocessing.Process(
        target=_temp_run,
        args=(in_outs, generation, debug, result_proxy, metadata_proxy, timeout),
    )
    process.start()
    process.join(timeout=timeout + 1)  # Join with a slightly longer timeout for the process itself

    if process.is_alive():
        process.kill()  # Force kill if still alive after timeout
        # process.terminate() # Alternative, more graceful termination

    # Convert proxy lists to regular lists for return
    # Ensure that if the process was killed, we have some default error state.
    if not result_proxy:  # If result_proxy is empty (e.g., process killed before appending)
        num_inputs = len(in_outs.get("inputs", []))
        final_results = [-1] * num_inputs if num_inputs > 0 else [-1]  # Mark all as error
        final_metadata = (
            [
                {
                    "error": "Global timeout or process killed prematurely",
                    "details": "No results returned from subprocess.",
                }
            ]
            * num_inputs
            if num_inputs > 0
            else [{"error": "Global timeout"}]
        )
        if debug:
            print("Global timeout or process killed before results could be appended.")
    else:
        final_results = list(result_proxy)[0]  # Expecting run_test to return a list of results for the inputs
        final_metadata = list(metadata_proxy)  # This should be a list of dicts

    # Ensure metadata_list has a corresponding entry for each input if it was a single dict from run_test
    if (
        isinstance(final_metadata, list)
        and len(final_metadata) == 1
        and isinstance(final_metadata[0], dict)
        and len(in_outs.get("inputs", [])) > 1
    ):
        # If run_test returned a single metadata dict for multiple inputs (e.g. on compilation error)
        # or if _temp_run appended a single error dict.
        # We might want to duplicate this error metadata for all inputs if results indicate multiple failures.
        # However, the original logic in prime_code's compute_score seems to handle metadata as a list already.
        # For now, assume metadata_proxy structure is as intended by _temp_run.
        pass

    return final_results, final_metadata
