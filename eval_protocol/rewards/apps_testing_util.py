# Copyright 2024 PRIME team and/or its affiliates
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

# Borrowed from: https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/utils.py
# Adapted for reward-kit: Removed pyext.RuntimeModule, other minor adjustments may be needed.

import ast
import faulthandler
import importlib.util  # Added for dynamic module loading
import json
import platform
import re  # Added for re.search

# to run the solution files we're using a timing based approach
import signal
import sys
import textwrap  # Added for dedenting model output
import traceback

# used for debugging to time steps
from datetime import datetime
from enum import Enum

# for capturing the stdout
from io import StringIO

# used for testing the code that reads from input
from unittest.mock import mock_open, patch

import numpy as np

# from pyext import RuntimeModule # Removed this problematic import


def truncatefn(s, length=300):
    assert isinstance(s, str)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda: None  # Changed lambda x: 1 to lambda: None
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def only_int_check(val):
    return isinstance(val, int)


def string_int_check(val):
    return isinstance(val, str) and val.isdigit()


def combined_int_check(val):
    return only_int_check(val) or string_int_check(val)


def clean_traceback(error_traceback):
    file_start = error_traceback.find('File "<string>"')
    if file_start == -1:  # Check if "<string>" is not found, common if exec is used directly
        file_start = error_traceback.find('File "<dynamic_module>"')  # Fallback for our dynamic module name

    if file_start != -1:
        error_traceback = "Traceback (most recent call last):\n  " + error_traceback[file_start:]
    return error_traceback


def _load_module_from_string(module_name, code_string):
    """Loads a Python module from a string using importlib."""
    spec = importlib.util.spec_from_loader(module_name, loader=None, origin="<generated_code>")
    if spec is None:
        raise ImportError(f"Could not create spec for dynamic module '{module_name}'")

    module = importlib.util.module_from_spec(spec)

    # Execute the code in the new module's namespace
    # Ensure that the module is usable by adding it to sys.modules temporarily if needed,
    # or by ensuring its __dict__ is correctly populated.
    try:
        exec(code_string, module.__dict__)
        # sys.modules[module_name] = module # Optional: if other parts of the code expect it in sys.modules
    except Exception:
        raise
    return module


def run_test(in_outs, test=None, debug=False, timeout=15):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    # Disable functionalities that can make destructive changes to the test.
    reliability_guard()

    if debug:
        print(f"start = {datetime.now().time()}")

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if test is None:
        raise AssertionError("should not happen: test code is none")
    elif test is not None:
        results = []
        # Standard library imports prepended to the solution
        sol = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"  # noqa: E501
        if debug:
            print(f"loading test code = {datetime.now().time()}")

        if which_type == CODE_TYPE.call_based:
            sol += test
            if debug:
                print(f"sol = {sol}")
            signal.alarm(timeout)  # This is Unix-specific
            try:
                # Replace RuntimeModule.from_string
                tmp_sol = _load_module_from_string("tmp_sol_call_based", sol)
                tmp = tmp_sol if "class Solution" not in test else tmp_sol.Solution()
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                error_traceback = traceback.format_exc()
                if debug:
                    print(f"type 0 compilation error = {e}")
                results.append(-2)
                return results, {
                    "error": repr(e),
                    "traceback": clean_traceback(error_traceback),
                }
            signal.alarm(0)

        elif which_type == CODE_TYPE.standard_input:
            try:
                astree = ast.parse(test)
                last_block = astree.body[-1]
                if isinstance(last_block, ast.If):
                    condition = last_block.test
                    if ast.unparse(condition).strip() == "__name__ == '__main__'":
                        # Build modules for unparse to avoid passing lists to ast.unparse
                        prefix_module = ast.Module(body=astree.body[:-1], type_ignores=[])
                        body_module = ast.Module(body=last_block.body, type_ignores=[])
                        test = ast.unparse(prefix_module) + "\n" + ast.unparse(body_module)
            except Exception:
                pass

            # `test` is the user's generated code string at this point.
            # Preprocessing for `if __name__ == "__main__"` is already done.

            # Dedent the entire model-generated code block first
            try:
                dedented_test_code = textwrap.dedent(test)
            except Exception as e:
                # In case dedent fails (e.g. on empty or malformed string), use original
                if debug:
                    print(f"Warning: textwrap.dedent failed on model code: {e}. Using original code.")
                dedented_test_code = test

            # Check if 'def main(' is in the dedented code and if 'main()' call is missing.
            main_defined = "def main(" in dedented_test_code
            main_called_at_toplevel = re.search(r"^\s*main\s*\(\s*\)", dedented_test_code, re.MULTILINE) is not None
            # Also consider if it's guarded by if __name__ == "__main__": which was removed by AST.
            # If the AST modification removed an if __name__ block that called main,
            # the original `test` string would be different from the AST-unparsed one.
            # This is complex to track perfectly here.
            # For now, a simpler heuristic: if `def main` is there, and no obvious `main()` call.

            user_code_lines = dedented_test_code.split("\n")

            # Imports from user code should be top-level in the module `sol`
            # Other lines form the body of `def code():`
            code_body_lines = []

            for line in user_code_lines:
                stripped_line = line.strip()
                if stripped_line.startswith("from ") or stripped_line.startswith("import "):
                    sol += stripped_line + "\n"  # Add stripped import directly to sol module scope
                else:
                    # Add original line from (potentially dedented) user code to be tab-indented
                    code_body_lines.append("\t" + line)

            if main_defined and not main_called_at_toplevel:
                # If system prompt asks for main(), and model provides def main() but no call, add it.
                # This assumes main() takes no arguments if called this way.
                # This is appended to be *inside* the `def code():` wrapper.
                code_body_lines.append("\tmain()")
                if debug:
                    print("Appended main() call as it was defined but not found called at top level.")

            # Construct the `def code():` wrapper string
            code_wrapper_str = "stdin = sys.stdin\nstdout = sys.stdout\ndef code():\n"
            code_wrapper_str += "\n".join(code_body_lines)

            sol += code_wrapper_str  # Add the "def code(): ..." to sol

            if debug:
                print(f"Constructed sol for standard_input: {sol}")
            method_name = "code"  # We will call the code() function
            signal.alarm(timeout)  # Unix-specific
            try:
                # Replace RuntimeModule.from_string
                tmp_sol = _load_module_from_string("tmp_sol_std_input", sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                error_traceback = traceback.format_exc()
                if debug:
                    print(f"type 1 compilation error = {e}")
                results.append(-2)
                return results, {
                    "error": repr(e),
                    "traceback": clean_traceback(error_traceback),
                }
            signal.alarm(0)

        if debug:
            print(f"get method = {datetime.now().time()}")

        try:
            # Ensure attribute name is a string for getattr
            method_name_str = str(method_name)
            method = getattr(tmp, method_name_str)
        except AttributeError:  # More specific exception
            signal.alarm(0)
            error_traceback = traceback.format_exc()
            # error_info = sys.exc_info() # sys.exc_info() is less clear than repr(e)
            results.append(-2)
            return results, {
                "error": f"AttributeError: Method '{method_name}' not found in dynamically loaded module.",
                "traceback": clean_traceback(error_traceback),
            }
        except Exception as e:  # Catch other potential errors during getattr
            signal.alarm(0)
            error_traceback = traceback.format_exc()
            results.append(-2)
            return results, {
                "error": repr(e),
                "traceback": clean_traceback(error_traceback),
            }

        for index, inputs_str in enumerate(in_outs["inputs"]):  # Renamed inputs to inputs_str
            raw_inputs = inputs_str
            raw_outputs = in_outs["outputs"][index]

            current_inputs = []  # Variable to hold processed inputs for the current test case

            if which_type == CODE_TYPE.call_based:
                # Assuming inputs_str is a string where each line is a separate JSON object for an argument
                current_inputs = [json.loads(line) for line in inputs_str.split("\n") if line.strip()]
                # Ensure in_outs["outputs"][index] is loaded if it's a string
                if isinstance(in_outs["outputs"][index], str):
                    in_outs["outputs"][index] = json.loads(in_outs["outputs"][index])

                truncate_line_size = 300 // (raw_inputs.count("\n") + 1) if raw_inputs.count("\n") > 0 else 300
                raw_inputs_truncated = "\n".join(
                    [truncatefn(line, truncate_line_size) for line in raw_inputs.strip().split("\n")]
                )
                raw_outputs_truncated = (
                    truncatefn(json.dumps(in_outs["outputs"][index]), 200)
                    if not isinstance(in_outs["outputs"][index], str)
                    else truncatefn(in_outs["outputs"][index], 200)
                )

            else:  # standard_input
                current_inputs = inputs_str  # For standard input, inputs might be a single string block
                raw_inputs_truncated = truncatefn(raw_inputs)
                raw_outputs_truncated = truncatefn(in_outs["outputs"][index], 200)

            # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
            # This part seems specific and might need careful handling if inputs are not always lists of dicts
            try:
                if which_type == CODE_TYPE.call_based and current_inputs and isinstance(current_inputs[0], dict):
                    current_inputs = [
                        {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in current_inputs[0].items()}
                    ]
            except Exception:
                pass  # Ignore if conversion fails, proceed with original

            # Similar conversion for outputs
            try:
                if isinstance(in_outs["outputs"][index], dict):
                    in_outs["outputs"][index] = {
                        int(k) if isinstance(k, str) and k.isdigit() else k: v
                        for k, v in in_outs["outputs"][index].items()
                    }
                elif (
                    isinstance(in_outs["outputs"][index], list)
                    and in_outs["outputs"][index]
                    and isinstance(in_outs["outputs"][index][0], dict)
                ):
                    in_outs["outputs"][index][0] = {
                        int(k) if isinstance(k, str) and k.isdigit() else k: v
                        for k, v in in_outs["outputs"][index][0].items()
                    }
            except Exception:
                pass

            if debug:
                print(
                    f"time: {datetime.now().time()} testing index = {index}  inputs = {current_inputs}, type = {which_type}"
                )

            if which_type == CODE_TYPE.call_based:
                signal.alarm(timeout)  # Unix-specific
                faulthandler.enable()
                try:
                    output = method(*current_inputs)

                    # For comparison, ensure output format matches expected (e.g. list vs tuple)
                    # ground truth sequences are not tuples
                    if isinstance(output, tuple):
                        output = list(output)

                    # Comparison logic
                    tmp_result = output == in_outs["outputs"][index]
                    # Handle cases where expected output might be a list containing the actual output
                    if isinstance(in_outs["outputs"][index], list) and len(in_outs["outputs"][index]) == 1:
                        tmp_result = tmp_result or (output == in_outs["outputs"][index][0])

                    # Further comparison for list of tuples vs list of lists
                    try:
                        if (
                            isinstance(output, list)
                            and output
                            and isinstance(output[0], tuple)
                            and isinstance(in_outs["outputs"][index], list)
                            and in_outs["outputs"][index]
                            and isinstance(in_outs["outputs"][index][0], list)
                        ):
                            output_list_of_lists = [list(x) for x in output]
                            tmp_result = tmp_result or (output_list_of_lists == in_outs["outputs"][index])
                            if isinstance(in_outs["outputs"][index][0], list):  # If expected is list of lists
                                tmp_result = tmp_result or (output_list_of_lists == in_outs["outputs"][index][0])

                    except Exception:
                        pass

                    results.append(tmp_result)

                    if tmp_result is not True:
                        return results, {
                            "output": truncatefn(json.dumps(output), 200),
                            "expected": raw_outputs_truncated,
                            "inputs": raw_inputs_truncated,
                            "error_message": "Wrong Answer",
                        }
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    error_traceback = traceback.format_exc()
                    faulthandler.disable()
                    if debug:
                        print(f"Call-based runtime error or time limit exceeded error = {e}")
                    results.append(-1)  # Indicate error
                    return results, {
                        "error": repr(e),
                        "traceback": clean_traceback(error_traceback),
                    }
                faulthandler.disable()
                signal.alarm(0)

            elif which_type == CODE_TYPE.standard_input:
                faulthandler.enable()

                # Ensure inputs_str is a single string for StringIO
                processed_inputs_str = inputs_str
                if isinstance(inputs_str, list):
                    processed_inputs_str = "\n".join(inputs_str)

                # Ensure ground_truth is a string for comparison
                ground_truth_str = in_outs["outputs"][index]
                if isinstance(ground_truth_str, list):
                    ground_truth_str = "\n".join(ground_truth_str)

                signal.alarm(timeout)  # Unix-specific
                captured_output_str = ""
                try:
                    with Capturing() as output_lines:
                        call_method(method, processed_inputs_str)
                    captured_output_str = "".join(
                        output_lines
                    ).rstrip()  # rstrip to remove trailing newline often added
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    error_traceback = traceback.format_exc()
                    faulthandler.disable()
                    results.append(-1)  # Indicate error
                    return results, {
                        "error": repr(e),
                        "traceback": clean_traceback(error_traceback),
                    }
                faulthandler.disable()
                signal.alarm(0)

                # Comparison for standard input
                # Normalize by splitting lines and stripping whitespace from each line
                output_for_compare = [line.strip() for line in captured_output_str.splitlines()]
                expected_for_compare = [line.strip() for line in ground_truth_str.splitlines()]

                tmp_result = output_for_compare == expected_for_compare

                # Additional float comparison if direct string match fails
                if not tmp_result:
                    try:
                        if len(output_for_compare) == len(expected_for_compare):
                            output_float = [float(x) for x in output_for_compare]
                            gt_float = [float(x) for x in expected_for_compare]
                            if np.allclose(output_float, gt_float):
                                tmp_result = True
                    except (ValueError, TypeError):
                        pass  # Not all are numbers, stick to string comparison

                results.append(tmp_result)
                if tmp_result is not True:
                    return results, {
                        "output": truncatefn(captured_output_str, 200),
                        "expected": raw_outputs_truncated,
                        "inputs": raw_inputs_truncated,
                        "error_message": "Wrong Answer",
                    }
        # If all test cases for this sample passed
    return results, {}


def custom_compare_(output, ground_truth):
    # This function seems to be part of an older comparison logic,
    # more direct comparisons are now in run_test.
    # Keeping it for now in case it's referenced, but likely can be simplified/removed.
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False


def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def call_method(method, inputs_str_for_mock):  # Renamed inputs to avoid conflict
    # inputs_str_for_mock is the single string containing all inputs for stdin

    inputs_line_iterator = iter(inputs_str_for_mock.split("\n"))

    @patch("builtins.open", mock_open(read_data=inputs_str_for_mock))
    @patch("sys.stdin", StringIO(inputs_str_for_mock))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator) + "\n")  # Add newline as readline expects
    @patch(
        "sys.stdin.readlines",
        lambda *args: [line + "\n" for line in inputs_str_for_mock.split("\n")],
    )
    @patch("sys.stdin.read", lambda *args: inputs_str_for_mock)
    def _inner_call_method(_method_to_call):  # Renamed _method to avoid conflict
        try:
            return _method_to_call()
        except SystemExit:  # Allow SystemExit to pass through, e.g. if code calls exit()
            pass
        finally:
            pass

    return _inner_call_method(method)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource  # Moved import here as it's Unix-specific for some parts

        # Check if resource module has RLIMIT_AS, etc. (for cross-platform safety)
        if hasattr(resource, "RLIMIT_AS"):
            resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        if hasattr(resource, "RLIMIT_DATA"):
            resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin" and hasattr(resource, "RLIMIT_STACK"):  # RLIMIT_STACK not on macOS
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()  # This is fine

    # It's generally safer to avoid modifying builtins directly if possible.
    # For a library, this can have wide-ranging effects.
    # Consider if this level of modification is truly necessary for eval_protocol's use case
    # or if the multiprocessing wrapper in utils.py provides sufficient isolation.
    # Note: The original implementation had many builtins and os/shutil functions commented out.
    # These have been removed for clarity, as the preferred method of sandboxing
    # would be via process isolation (e.g. multiprocessing or a dedicated sandbox env).
    # Modifying builtins directly in a library function can have unintended side effects.

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    # Disabling os functions: Be cautious, as this makes the execution environment very restrictive.
    # This might be too aggressive if the generated code legitimately needs some safe os interactions.
    # The multiprocessing wrapper in utils.py already provides process isolation.

    # Example of functions that were previously considered for disabling:
    # os.kill, os.system, os.remove, os.fork, etc.
    # shutil.rmtree, shutil.move
    # subprocess.Popen
    # Modifying __builtins__ or sys.modules entries.

    # For eval_protocol, rely on higher-level sandboxing if untrusted code execution is a concern.
    # The memory limits via `resource` are a good first step for resource exhaustion.
    import shutil  # Keep import if other shutil functions are used, or remove if not.
    import subprocess  # Keep import if other subprocess functions are used, or remove if not.
    import sys  # Keep import for sys.stdout, sys.stdin manipulations.
