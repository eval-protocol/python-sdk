"""Normalizes various coding dataset formats from SandboxFusion.

This script converts datasets into a unified OpenAI-compatible JSONL format.
"""

import ast
import hashlib
import json
import os
from typing import (  # Removed Callable, Union. Added List, TypedDict
    Any,
    Dict,
    List,
    Optional,
    TypedDict,
)

from transformers import AutoTokenizer  # For Repobench-P

# Define the root path to the SandboxFusion sample datasets
# (Relative to the eval-protocol project root)
SANDBOX_SAMPLES_DIR = "./SandboxFusion/sandbox/tests/datasets/samples/"

# List of Python-specific dataset .jsonl files
PYTHON_SPECIFIC_JSONL_FILES = [
    "code_eval_shadow_humaneval_python.jsonl",
    "code_eval_mbpp.jsonl",
    "code_eval_mhpp.jsonl",
    "code_eval_ncb_python_en.jsonl",
    "code_eval_ncb_python_zh.jsonl",
    "code_eval_repobench_c_python_sampled.jsonl",  # RepoBench-C, Python subset
    "code_eval_repobench_p_python_sampled.jsonl",  # RepoBench-P, Python subset
    "code_eval_cruxeval.jsonl",  # Python by default
    "code_eval_cruxeval_x.jsonl",  # Multilingual, filter for Python
    "code_eval_aider_benchmark_v1.jsonl",
    "code_eval_bigcodebench.jsonl",
    "code_eval_EvoEval.jsonl",
]

# List of multilingual dataset .jsonl files that need filtering for Python
MULTILINGUAL_JSONL_FILES_TO_FILTER = [
    "code_eval_mbxp_v1_en.jsonl",
    "code_eval_humanevalds_v1_en.jsonl",
    "code_eval_humanevalds_v2_en.jsonl",
    "code_eval_mbxp_v2_en.jsonl",
]

ALL_SOURCE_JSONL_FILES = PYTHON_SPECIFIC_JSONL_FILES + MULTILINGUAL_JSONL_FILES_TO_FILTER

# Output file path
OUTPUT_JSONL_FILE = "./development/CODING_DATASET.jsonl"

# --- Helper for Repobench-P ---
# Global tokenizer instance for Repobench-P to avoid reloading it for each problem
# Note: This assumes "assets/tokenizer/gpt2" is accessible relative to the execution path.
try:
    repobench_p_tokenizer = AutoTokenizer.from_pretrained("gpt2")
except OSError:
    print("Warning: Could not load gpt2 tokenizer for Repobench-P. Falling back to basic split for token counting.")
    repobench_p_tokenizer = None


def count_tokens_for_repobench_p(text: str) -> int:
    """Count tokens for Repobench-P, using the gpt2 tokenizer or a fallback."""
    if repobench_p_tokenizer:
        return len(repobench_p_tokenizer.encode(text))
    return len(text.split())  # Basic fallback


def decode_tokens_for_repobench_p(tokens: list) -> str:
    """Decode tokens for Repobench-P, using the gpt2 tokenizer or a fallback."""
    if repobench_p_tokenizer:
        return repobench_p_tokenizer.decode(tokens)
    return " ".join(map(str, tokens))  # Basic fallback


def comment_repobench_p_snippet(code: str, language: str):
    """Comment out a code snippet based on its language for Repobench-P."""
    if language == "python":
        return "\n".join([f"# {line}" for line in code.split("\n")])
    # Add other languages if necessary, though we focus on Python
    return code


# --- End Helper for Repobench-P ---


class ContextInfo(TypedDict):
    text: str
    tokens: int
    original_index: int


def extract_python_docstring(code_string: str) -> Optional[str]:
    """Extract docstring from first func/class in Python code."""
    try:
        tree = ast.parse(code_string.strip())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                return docstring.strip() if docstring else None
        return None
    except SyntaxError:
        return None


def format_aider_prompt(problem_json: dict) -> str:
    """Format the prompt for Aider benchmark style problems."""
    question = problem_json.get("content", "")
    return (
        f"{question}\n\nPlease generate the code in the following format:\n```python\n# Your code response here\n```"
    )


def format_mbpp_prompt(problem_json: dict) -> str:
    """Format the prompt for MBPP and MBXP style problems."""
    question = problem_json.get("content", "")
    test_list = problem_json.get("test_list", [])  # MBPP specific
    if not test_list and isinstance(problem_json.get("labels"), dict):  # For MBXP
        test_list = problem_json["labels"].get("test_list", [])

    tests_string = "\n".join(test_list)
    return (
        f"You are an expert Python programmer, and here is your task: {question} "
        f"Your code should pass these tests:\n\n{tests_string}"
    )


def format_repobench_p_prompt(problem_json: dict, lang: str = "python") -> str:
    """Format the prompt for Repobench-P style problems.

    This is a simplified port of Repobench-P's _generate_single_prompt.
    """
    # This is complex and may need the actual tokenizer assets to be perfectly replicated.
    # Using gpt2 tokenizer as a stand-in. Max prompt length default from parser.
    max_prompt_length_tokens = 8000
    # Max prompt tokens
    current_file_max_tokens = 1600  # Max tokens for current file code

    code_context = problem_json.get("code", "")
    file_path = problem_json.get("file_path", "unknown_file.py")
    import_statement = problem_json.get("import_statement", "")

    # Prepare current file's code
    code_snippet = "\n".join(code_context.split("\n")[-60:])  # Last 60 lines
    if lang == "python":
        code_snippet = f"# Path: {file_path}\n{import_statement}\n{code_snippet}"
    # (Add other lang handling if needed)

    code_tokens = count_tokens_for_repobench_p(code_snippet)
    if code_tokens > current_file_max_tokens:
        # This truncation needs to be done carefully with actual tokens
        # For simplicity, we're using a rough character-based trim if tokenizer failed.
        if repobench_p_tokenizer:
            encoded_tokens = repobench_p_tokenizer.encode(code_snippet)[-current_file_max_tokens:]
            code_snippet = decode_tokens_for_repobench_p(encoded_tokens)
        else:  # Fallback if tokenizer is not available
            code_snippet = code_snippet[-int(current_file_max_tokens * 4) :]  # Approx char length

    current_prompt_tokens = count_tokens_for_repobench_p(code_snippet)
    final_prompt_parts: List[str] = [code_snippet]  # Current code is the last part

    # Prepare context snippets
    contexts_info: List[ContextInfo] = []
    raw_contexts = problem_json.get("context", [])
    if isinstance(raw_contexts, list):
        for i, ctx_item in enumerate(raw_contexts):
            if not isinstance(ctx_item, dict):
                continue
            snippet_path = ctx_item.get("path", "unknown_context_file.py")
            snippet_content = ctx_item.get("snippet", "")

            commented_snippet = comment_repobench_p_snippet(snippet_content, lang)

            if lang == "python":
                formatted_snippet = f"# Path: {snippet_path}\n{commented_snippet}\n"
            # (Add other lang handling)
            else:
                formatted_snippet = f"// Path: {snippet_path}\n{commented_snippet}\n"

            contexts_info.append(
                {
                    "text": formatted_snippet,
                    "tokens": count_tokens_for_repobench_p(formatted_snippet),
                    "original_index": i,
                }
            )

    # Add gold snippet first if specified and exists
    gold_snippet_idx = problem_json.get("gold_snippet_index", -1)
    if isinstance(gold_snippet_idx, int) and 0 <= gold_snippet_idx < len(contexts_info):
        gold_snippet_info = next((c for c in contexts_info if c["original_index"] == gold_snippet_idx), None)
        if gold_snippet_info and (current_prompt_tokens + gold_snippet_info["tokens"] <= max_prompt_length_tokens):
            final_prompt_parts.insert(0, gold_snippet_info["text"])  # Prepend
            current_prompt_tokens += gold_snippet_info["tokens"]
            contexts_info = [
                c for c in contexts_info if c["original_index"] != gold_snippet_idx
            ]  # Remove from further processing

    # Add other contexts sorted by md5 hash, until token limit
    contexts_info.sort(key=lambda x: hashlib.md5(str(x["text"]).encode("utf8")).hexdigest())

    for ctx_info in contexts_info:
        if current_prompt_tokens + ctx_info["tokens"] <= max_prompt_length_tokens:
            final_prompt_parts.insert(0, ctx_info["text"])  # Prepend
            current_prompt_tokens += ctx_info["tokens"]
        else:
            break  # Token limit reached

    return "".join(reversed(final_prompt_parts))  # They were prepended, so reverse to get correct order


def format_cruxeval_output_prompt(problem_json: dict) -> str:
    """Format the prompt for CruxEval output prediction tasks."""
    # Using 'direct output prompt' style from cruxeval.py
    code = problem_json.get("code", "")
    test_input = problem_json.get("input", "")  # This is the input to f()

    # Ensure test_input is represented as a string literal if it's not already
    # The problem_json['input'] should already be in the correct string representation.

    return (
        "You are given a Python function and an assertion containing an input to "
        "the function. Complete the assertion with a literal (no unsimplified "
        "expressions, no function calls) containing the output when executing the "
        "provided code on the given input, even if the function is incorrect or "
        "incomplete. Do NOT output any extra information. Provide the full assertion "
        "with the correct output in [ANSWER] and [/ANSWER] tags, "
        "following the examples.\n\n"
        "[PYTHON]\n"
        "def f(n):\n"
        "    return n\n"
        "assert f(17) == ??\n"
        "[/PYTHON]\n"
        "[ANSWER]\n"
        "assert f(17) == 17\n"
        "[/ANSWER]\n\n"
        "[PYTHON]\n"
        "def f(s):\n"
        '    return s + "a"\n'
        'assert f("x9j") == ??\n'
        "[/PYTHON]\n"
        "[ANSWER]\n"
        'assert f("x9j") == "x9ja"\n'
        "[/ANSWER]\n\n"
        f"[PYTHON]\n{code}\nassert f({{test_input}}) == ??\n[/PYTHON]\n[ANSWER]\n"
    )


def format_cruxeval_output_assistant(problem_json: dict) -> str:
    """Format the assistant's response for CruxEval output prediction tasks."""
    test_input = problem_json.get("input", "")
    expected_output = problem_json.get("output", "")  # This is the value f() returns
    # Ensure expected_output is represented as a string literal
    # The problem_json['output'] should already be in the correct string representation.
    return f"assert f({test_input}) == {expected_output}"


def normalize_problem_to_openai_format(
    problem_json: dict, filename: str, is_multilingual_file: bool
) -> Optional[Dict[str, Any]]:
    """Normalize a problem from various dataset formats to the OpenAI messages format.

    Handles dataset-specific prompt engineering and language filtering.
    """
    problem_id_str = str(problem_json.get("id", "N/A"))
    try:
        # Robust key finding from the original script
        user_content_keys = [
            "content",
            "prompt",
            "problem",
            "text",
            "code",
        ]  # Added "code" as a fallback
        assistant_content_keys = [
            "canonical_solution",
            "solution",
            "code",
            "completion",
            "next_line",
            "output",
        ]  # Added "next_line", "output"

        raw_user_content = None
        primary_user_key_was_wrong_type = False
        for key_idx, key in enumerate(user_content_keys):
            if key in problem_json:
                if isinstance(problem_json[key], str):
                    raw_user_content = problem_json[key]
                    break
                elif key_idx == 0 and key == user_content_keys[0]:  # Only log if primary 'content' is wrong type
                    primary_user_key_was_wrong_type = True

        raw_assistant_content = None
        for key_idx, key in enumerate(assistant_content_keys):
            if key in problem_json:
                if isinstance(
                    problem_json[key], (str, int, float, bool, list, dict)
                ):  # Allow more types for raw_assistant
                    raw_assistant_content = problem_json[key]
                    if isinstance(
                        raw_assistant_content, (int, float, bool, list, dict)
                    ):  # Convert non-strings for now
                        raw_assistant_content = str(raw_assistant_content)
                    break
                elif key_idx == 0 and key == assistant_content_keys[0]:
                    pass

        if raw_user_content is None:
            if primary_user_key_was_wrong_type:
                print(
                    f"Warning: Skipping ID {problem_id_str} in {filename} - "
                    f"primary user key '{user_content_keys[0]}' not a string."
                )
            else:
                print(
                    f"Warning: Skipping ID {problem_id_str} in {filename} - "
                    f"missing user content (keys: {user_content_keys})."
                )
            return None

        labels_data = problem_json.get("labels")
        labels = {}
        if isinstance(labels_data, str):
            try:
                labels = json.loads(labels_data)
            except json.JSONDecodeError:
                print(f"Warning: Skipping ID {problem_id_str} in {filename} - malformed JSON in labels.")
                return None
        elif isinstance(labels_data, dict):
            labels = labels_data

        programming_language = labels.get("programming_language", "python" if "python" in filename else None)
        if not programming_language and "cruxeval_x" in filename and isinstance(problem_json.get("id"), str):
            lang_part = problem_json["id"].split("_")[0]
            if lang_part in ["python", "py"]:
                programming_language = "python"

        if is_multilingual_file or "cruxeval_x" in filename:
            if programming_language != "python":
                return None

        final_user_content = raw_user_content
        final_assistant_content = str(raw_assistant_content) if raw_assistant_content is not None else ""

        if "aider_benchmark" in filename:
            final_user_content = format_aider_prompt(problem_json)
        elif "mbpp" in filename and "mbxp" not in filename:
            final_user_content = format_mbpp_prompt(problem_json)
            test_setup_code = labels.get("test_setup_code", "")
            if test_setup_code and isinstance(test_setup_code, str) and test_setup_code not in final_assistant_content:
                final_assistant_content = test_setup_code.strip() + "\n\n" + final_assistant_content
        elif "mhpp" in filename:
            original_content_for_mhpp = problem_json.get("content", "")
            first_line_of_test = ""
            if problem_json.get("test") and isinstance(problem_json["test"], str):
                first_line_of_test = problem_json["test"].split("\n")[0]
            prompt_stub = original_content_for_mhpp
            if '"""' in prompt_stub:
                prompt_stub = prompt_stub[: prompt_stub.rfind('"""')]
            final_user_content = f'{prompt_stub}\n    e.g. {first_line_of_test} """'
            if not ("def " in final_assistant_content.strip() or "class " in final_assistant_content.strip()):
                if original_content_for_mhpp.rstrip().endswith(":"):
                    final_assistant_content = original_content_for_mhpp.rstrip() + "\n" + final_assistant_content
                elif original_content_for_mhpp.endswith("\n"):
                    final_assistant_content = original_content_for_mhpp + final_assistant_content
                else:
                    final_assistant_content = original_content_for_mhpp + "\n" + final_assistant_content
        elif "ncb_python" in filename:
            final_user_content = problem_json.get("content", raw_user_content)
            final_assistant_content = problem_json.get("canonical_solution", raw_assistant_content)
        elif "repobench_c" in filename:
            final_user_content = problem_json.get("prompt", raw_user_content)
            final_assistant_content = problem_json.get("next_line", raw_assistant_content)
        elif "repobench_p" in filename:
            final_user_content = format_repobench_p_prompt(problem_json, lang="python")
            final_assistant_content = problem_json.get("next_line", raw_assistant_content)
        elif "cruxeval" in filename:
            final_user_content = format_cruxeval_output_prompt(problem_json)
            final_assistant_content = format_cruxeval_output_assistant(problem_json)
        elif (
            "humaneval" in filename
            or "evoeval" in filename
            or "bigcodebench" in filename
            or (
                is_multilingual_file
                and ("humanevalds" in filename or labels.get("task_id", "").startswith("humanevalds"))
            )
        ):
            extracted_docstring = extract_python_docstring(raw_user_content)
            if extracted_docstring:
                final_user_content = extracted_docstring
                if not ("def " in final_assistant_content.strip() or "class " in final_assistant_content.strip()):
                    if raw_user_content.rstrip().endswith(":"):
                        final_assistant_content = raw_user_content.rstrip() + "\n" + final_assistant_content
                    elif raw_user_content.endswith("\n"):
                        final_assistant_content = raw_user_content + final_assistant_content
                    else:
                        final_assistant_content = raw_user_content + "\n" + final_assistant_content
            else:
                final_user_content = raw_user_content
        elif is_multilingual_file and ("mbxp" in filename or labels.get("task_id", "").startswith("mbxp")):
            final_user_content = format_mbpp_prompt(problem_json)
            test_setup_code = labels.get("test_setup_code", "")
            if test_setup_code and isinstance(test_setup_code, str) and test_setup_code not in final_assistant_content:
                final_assistant_content = test_setup_code.strip() + "\n\n" + final_assistant_content
        else:
            extracted_docstring = extract_python_docstring(raw_user_content)
            if extracted_docstring:
                final_user_content = extracted_docstring
                if not ("def " in final_assistant_content.strip() or "class " in final_assistant_content.strip()):
                    if raw_user_content.rstrip().endswith(":"):
                        final_assistant_content = raw_user_content.rstrip() + "\n" + final_assistant_content
                    elif raw_user_content.endswith("\n"):
                        final_assistant_content = raw_user_content + final_assistant_content
                    else:
                        final_assistant_content = raw_user_content + "\n" + final_assistant_content

        if not isinstance(final_user_content, str) or not isinstance(final_assistant_content, str):
            print(
                f"Warning: Skipping ID {problem_id_str} in {filename} - "
                f"invalid content types (user: {type(final_user_content)}, "
                f"assistant: {type(final_assistant_content)})."
            )
            return None
        if not final_user_content.strip() or not final_assistant_content.strip():
            print(f"Warning: Skipping ID {problem_id_str} in {filename} - empty processed content.")
            return None
        if final_assistant_content.strip() == "import sys; sys.exit(0)":
            print(f"Warning: Skipping ID {problem_id_str} in {filename} - placeholder solution.")
            return None

        return {
            "messages": [
                {"role": "user", "content": final_user_content.strip()},
                {"role": "assistant", "content": final_assistant_content.strip()},
            ]
        }
    except Exception as e:
        print(f"Warning: Skipping ID {problem_id_str} in {filename} - error ({type(e).__name__}: {e}).")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Process SandboxFusion datasets and normalize them to OpenAI JSONL format.

    This function iterates through specified dataset files, normalizes each problem
    to an OpenAI-compatible format, and writes the results to an output JSONL file.
    It handles language filtering for multilingual datasets and logs errors or
    skipped problems.
    """
    output_dir = os.path.dirname(OUTPUT_JSONL_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_count = 0
    skipped_count = 0
    file_error_count = 0

    print(f"Starting dataset normalization. Output will be written to {OUTPUT_JSONL_FILE}")

    with open(OUTPUT_JSONL_FILE, "w", encoding="utf-8") as outfile:
        for filename_idx, filename in enumerate(ALL_SOURCE_JSONL_FILES):
            filepath = os.path.join(SANDBOX_SAMPLES_DIR, filename)
            is_multilingual = filename in MULTILINGUAL_JSONL_FILES_TO_FILTER

            if not os.path.exists(filepath):
                print(f"Warning: File not found, skipping: {filepath}")
                file_error_count += 1
                continue

            print(f"Processing file {filename_idx + 1}/{len(ALL_SOURCE_JSONL_FILES)}: {filename}...")
            lines_in_file = 0
            processed_in_file = 0
            skipped_in_file = 0
            try:
                with open(filepath, "r", encoding="utf-8") as infile:
                    for line_number, line in enumerate(infile, 1):
                        lines_in_file += 1
                        stripped_line = line.strip()
                        if not stripped_line:
                            continue
                        try:
                            problem_data = json.loads(stripped_line)
                        except json.JSONDecodeError:
                            print(f"Warning: Malformed JSON on line {line_number} in {filepath}. Skipping line.")
                            skipped_in_file += 1
                            continue

                        normalized_problem = normalize_problem_to_openai_format(
                            problem_data, filename, is_multilingual
                        )
                        if normalized_problem:
                            outfile.write(json.dumps(normalized_problem) + "\n")
                            processed_in_file += 1
                        else:
                            skipped_in_file += 1
                print(
                    f"Finished {filename}. Lines: {lines_in_file}, "
                    f"Processed: {processed_in_file}, Skipped: {skipped_in_file}"
                )
                processed_count += processed_in_file
                skipped_count += skipped_in_file
            except Exception as e:
                print(f"Error processing file {filepath}: {type(e).__name__}: {e}. Skipping rest of file.")
                import traceback

                traceback.print_exc()
                file_error_count += 1

    print("\nDataset normalization complete.")
    print(f"Total problems processed and written: {processed_count}")
    print(f"Total problems/lines skipped: {skipped_count}")
    print(f"Total files with errors or not found: {file_error_count}")


if __name__ == "__main__":
    main()
