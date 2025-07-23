import json
from typing import Any, Dict, List, Optional


def prepare_deepcoder_sample_for_trl(raw_sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms a raw DeepCoder-style sample into a format suitable for TRL training
    with the deepcoder_code_reward function.

    This involves extracting the user prompt, parsing test cases, and appending
    necessary instructions to the prompt based on the presence of a target_function.

    Args:
        raw_sample: A dictionary representing a single raw sample, typically from
                    a JSONL file like 'simulated_deepcoder_raw_sample.jsonl'.
                    Expected keys: "prompt" (list of chat messages),
                                   "reward_model": {"ground_truth": "[...test_cases...]"},
                                   "target_function": (optional) string.

    Returns:
        A dictionary containing:
        - 'prompt': The processed prompt string for the LLM.
        - 'test_cases': A list of parsed test case dictionaries.
        - 'target_function': The target function name, if provided.
    """
    prompt_content = ""
    if isinstance(raw_sample.get("prompt"), list) and len(raw_sample["prompt"]) > 0:
        for msg in raw_sample["prompt"]:
            if msg.get("role") == "user" and msg.get("content"):
                prompt_content = msg["content"]
                break
    if not prompt_content:
        prompt_content = str(raw_sample.get("prompt", ""))

    target_function = raw_sample.get("target_function")

    # Append instructions based on target_function
    # These instructions guide the LLM to produce output compatible with deepcoder_code_reward
    if target_function:
        instruction = (
            f"\n\nIMPORTANT: You are to write a Python function named '{target_function}'. "
            "Generate ONLY the complete function definition for this function. "
            "Do not include any example usage, print statements outside the function, "
            "or any code that reads from stdin or writes to stdout, unless the problem "
            "description explicitly requires the function itself to perform such I/O."
        )
    else:
        # This case might be less common for deepcoder_code_reward if it expects a function name,
        # but providing a fallback instruction.
        instruction = (
            "\n\nIMPORTANT: Your code should be a complete Python script or function. "
            "If the problem implies standard input/output, structure your code to read from "
            "stdin and print to stdout. Only print the final result."
        )

    final_prompt = prompt_content + instruction

    test_cases_str = raw_sample.get("reward_model", {}).get("ground_truth", "[]")
    try:
        test_cases = json.loads(test_cases_str)
    except json.JSONDecodeError:
        # If ground_truth is already parsed (e.g. if input is already somewhat processed)
        if isinstance(test_cases_str, list):
            test_cases = test_cases_str
        else:
            test_cases = []

    return {
        "prompt": final_prompt,
        "test_cases": test_cases,
        "target_function": target_function,  # Pass through for reward_kwargs_map
    }
