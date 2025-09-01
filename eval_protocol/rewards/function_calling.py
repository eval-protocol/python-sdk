import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Import OpenAI at module level for mocking in tests
try:
    import openai
    from openai import OpenAI
except ImportError:
    # Type to mock in tests
    OpenAI = None  # type: ignore

import copy
from collections import Counter

from ..models import EvaluateResult, Message, MetricResult
from ..typed_interface import reward_function


def match_function_call(
    messages: List[Dict[str, Any]],
    function_name: str,
    parsed_arguments: Dict[str, Any],
    expected_call_schema: Dict[str, Any],
    argument_match_strictness: str = "exact",
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate how well a function call matches an expected schema.

    Args:
        messages: The conversation messages (for context, not directly used for call parts).
        function_name: The parsed function name.
        parsed_arguments: The parsed arguments from the function call.
        expected_call_schema: The expected schema for the function call.
        argument_match_strictness: How strict to be with argument matching:
            - "exact": All arguments must match exactly
            - "partial": Only check provided arguments, ignore missing ones
            - "flexible": Allow extra arguments and type mismatches with penalty

    Returns:
        EvaluateResult with score and metrics
    """
    metrics = {}

    # 1. Function name match
    expected_name = expected_call_schema.get("name", "")
    name_match = function_name == expected_name
    name_score = 1.0 if name_match else 0.0
    name_reason = f"Function name {'matches' if name_match else 'does not match'}: expected '{expected_name}', got '{function_name}'"
    metrics["function_name_match"] = MetricResult(score=name_score, reason=name_reason, is_score_valid=name_match)

    # 2. Arguments match
    expected_args = expected_call_schema.get("arguments", {})
    arg_score = 0.0
    arg_details = []

    missing_args = []
    extra_args = []
    type_mismatches = []
    perfect_matches = []

    for arg_name, arg_schema in expected_args.items():
        expected_type = arg_schema.get("type", "any")

        if arg_name not in parsed_arguments:
            missing_args.append(arg_name)
            arg_details.append(f"Missing argument: {arg_name}")
        else:
            arg_value = parsed_arguments[arg_name]
            type_matched = True
            if expected_type == "string" and not isinstance(arg_value, str):
                type_mismatches.append(arg_name)
                arg_details.append(f"Type mismatch for {arg_name}: expected string, got {type(arg_value).__name__}")
                type_matched = False
            elif expected_type == "number" and not isinstance(arg_value, (int, float)):
                type_mismatches.append(arg_name)
                arg_details.append(f"Type mismatch for {arg_name}: expected number, got {type(arg_value).__name__}")
                type_matched = False
            elif expected_type == "boolean" and not isinstance(arg_value, bool):
                type_mismatches.append(arg_name)
                arg_details.append(f"Type mismatch for {arg_name}: expected boolean, got {type(arg_value).__name__}")
                type_matched = False
            elif expected_type == "array" and not isinstance(arg_value, list):
                type_mismatches.append(arg_name)
                arg_details.append(f"Type mismatch for {arg_name}: expected array, got {type(arg_value).__name__}")
                type_matched = False
            elif expected_type == "object" and not isinstance(arg_value, dict):
                type_mismatches.append(arg_name)
                arg_details.append(f"Type mismatch for {arg_name}: expected object, got {type(arg_value).__name__}")
                type_matched = False

            if type_matched:
                perfect_matches.append(arg_name)
                arg_details.append(f"Argument {arg_name} matches expected type {expected_type}")

    for arg_name in parsed_arguments:
        if arg_name not in expected_args:
            extra_args.append(arg_name)
            arg_details.append(f"Unexpected argument: {arg_name}")

    if argument_match_strictness == "exact":
        if missing_args or extra_args or type_mismatches:
            arg_score = 0.0
        else:
            arg_score = 1.0
    elif argument_match_strictness == "partial":
        if extra_args or type_mismatches:
            arg_score = 0.0
        else:
            total_provided = len(parsed_arguments)
            if total_provided == 0:
                arg_score = 0.0
            else:
                correct_args = len(perfect_matches)
                arg_score = correct_args / total_provided
    elif argument_match_strictness == "permissive" or argument_match_strictness == "flexible":
        if missing_args or type_mismatches:
            arg_score = 0.0
        else:
            arg_score = 1.0
    else:
        raise ValueError(f"Invalid argument_match_strictness: {argument_match_strictness}")

    arg_reason = "\n".join(arg_details)
    metrics["arguments_match"] = MetricResult(
        score=arg_score,
        reason=arg_reason,
        is_score_valid=arg_score == 1.0 if len(expected_args) > 0 else True,
    )

    # 3. Calculate final score
    final_score = (name_score + arg_score) / 2.0
    final_reason = f"Overall score based on name match ({name_score:.2f}) and argument match ({arg_score:.2f})."

    return EvaluateResult(score=final_score, reason=final_reason, metrics=metrics)


def calculate_jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    Calculate Jaccard similarity between two sets.

    Jaccard similarity is defined as the size of the intersection divided by the size of the union.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Jaccard similarity score between 0.0 and 1.0
    """
    if not set1 and not set2:
        return 1.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union


def extract_schema_properties(schema: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """
    Extract properties from a JSON schema as a set of (name, type) tuples.

    Args:
        schema: JSON schema object

    Returns:
        Set of (property_name, property_type) tuples
    """
    properties = set()

    def process_properties(schema_obj: Dict[str, Any], prefix: str = ""):
        if not isinstance(schema_obj, dict):
            return

        props = schema_obj.get("properties", {})
        for prop_name, prop_schema in props.items():
            prop_path = f"{prefix}.{prop_name}" if prefix else prop_name
            prop_type = prop_schema.get("type", "any")
            properties.add((prop_path, prop_type))
            if prop_type == "object":
                process_properties(prop_schema, prop_path)

        pattern_props = schema_obj.get("patternProperties", {})
        for pattern, pattern_schema in pattern_props.items():
            prop_path = f"{prefix}[{pattern}]" if prefix else f"[{pattern}]"
            prop_type = pattern_schema.get("type", "any")
            properties.add((prop_path, prop_type))
            if prop_type == "object":
                process_properties(pattern_schema, prop_path)

        items = schema_obj.get("items", {})
        if items and isinstance(items, dict):
            prop_path = f"{prefix}[]" if prefix else "[]"
            prop_type = items.get("type", "any")
            properties.add((prop_path, prop_type))
            if prop_type == "object":
                process_properties(items, prop_path)

    process_properties(schema)
    return properties


def normalize_schema(schema: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Normalize schema to a standard dictionary format.

    Args:
        schema: JSON schema as dictionary or string

    Returns:
        Normalized schema dictionary
    """
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            return {}

    if not isinstance(schema, dict):
        return {}

    return schema


# New Exact Tool Match Reward Function and Helpers
# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV


def maybe_deserialize_tool_call_arguments(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Deserializes the 'arguments' field (if it's a JSON string) within each tool call's 'function' object.
    Input tool_calls are expected to be in OpenAI format:
    [{'id': ..., 'type': 'function', 'function': {'name': ..., 'arguments': 'JSON_STRING_ARGS'}}, ...]
    """
    processed_tool_calls = []
    if not tool_calls:
        return []

    for tc_openai_format in tool_calls:
        if not isinstance(tc_openai_format, dict) or "function" not in tc_openai_format:
            continue

        function_details = tc_openai_format.get("function", {})
        if not isinstance(function_details, dict) or "arguments" not in function_details:
            continue

        arguments_val = function_details["arguments"]
        deserialized_args = arguments_val

        if isinstance(arguments_val, str):
            if not arguments_val.strip():
                deserialized_args = {}
            else:
                try:
                    deserialized_args = json.loads(arguments_val)
                except json.JSONDecodeError:
                    # If arguments string is not valid JSON, keep it as a string.
                    # This matches behavior of some models that might return non-JSON arguments.
                    pass

        new_tc = copy.deepcopy(tc_openai_format)
        new_tc["function"]["arguments"] = deserialized_args
        processed_tool_calls.append(new_tc)

    return processed_tool_calls


def parse_tool_calls(completion: str) -> list:
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", completion, re.DOTALL)
    row_tool_calls = []
    for match in matches:
        try:
            tool_call_str = match.strip()
            row_tool_calls.append(json.loads(tool_call_str))
        except Exception:
            continue
    return row_tool_calls


def compare_tool_calls(generated_tool_calls: list, gt_tool_calls: list) -> bool:
    if len(generated_tool_calls) != len(gt_tool_calls):
        return False

    generated_tool_calls_serialized = [json.dumps(item, sort_keys=True) for item in generated_tool_calls]
    gt_tool_calls_serialized = [json.dumps(item, sort_keys=True) for item in gt_tool_calls]

    return generated_tool_calls_serialized == gt_tool_calls_serialized


def eval_tool_call(generation: dict, ground_truth: dict) -> bool:
    if ground_truth is None or "tool_calls" not in ground_truth:
        expected_gt_tool_calls = []
    else:
        expected_gt_tool_calls = ground_truth["tool_calls"]

    deserialized_gt_openai_tool_calls = maybe_deserialize_tool_call_arguments(expected_gt_tool_calls or [])
    ground_truth_simple_format = [tc["function"] for tc in deserialized_gt_openai_tool_calls if "function" in tc]

    generated_simple_format = []
    raw_generated_tool_calls = generation.get("tool_calls")

    if raw_generated_tool_calls:
        processed_gen_tool_calls_openai_format = []
        for tc in raw_generated_tool_calls:
            if hasattr(tc, "model_dump"):
                processed_gen_tool_calls_openai_format.append(tc.model_dump())
            elif isinstance(tc, dict):
                processed_gen_tool_calls_openai_format.append(tc)

        deserialized_gen_openai_tool_calls = maybe_deserialize_tool_call_arguments(
            processed_gen_tool_calls_openai_format
        )
        generated_simple_format = [tc["function"] for tc in deserialized_gen_openai_tool_calls if "function" in tc]
    elif generation.get("content") and "<tool_call>" in generation["content"]:
        parsed_tool_calls_from_content_str = parse_tool_calls(generation["content"])
        temp_openai_formatted_list = []
        for item in parsed_tool_calls_from_content_str:
            if isinstance(item, dict) and "function" in item and "type" in item:
                temp_openai_formatted_list.append(item)
            elif isinstance(item, dict) and "name" in item and "arguments" in item:
                temp_openai_formatted_list.append(
                    {
                        "id": f"parsed_call_{len(temp_openai_formatted_list)}",
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": (
                                json.dumps(item["arguments"])
                                if isinstance(item["arguments"], dict)
                                else item["arguments"]
                            ),
                        },
                    }
                )

        if temp_openai_formatted_list:
            deserialized_calls_from_content = maybe_deserialize_tool_call_arguments(temp_openai_formatted_list)
            generated_simple_format = [tc["function"] for tc in deserialized_calls_from_content if "function" in tc]

    return compare_tool_calls(generated_simple_format, ground_truth_simple_format)


@reward_function
def exact_tool_match_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> EvaluateResult:
    if not messages:
        return EvaluateResult(score=0.0, reason="No messages provided for evaluation.", metrics={})

    generation_message_obj = messages[-1]
    generation_dict: Dict[str, Any]

    if isinstance(generation_message_obj, Message):
        generation_dict = {
            "role": generation_message_obj.role,
            "content": generation_message_obj.content,
        }
        if generation_message_obj.tool_calls:
            generation_dict["tool_calls"] = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in generation_message_obj.tool_calls
            ]
    elif isinstance(generation_message_obj, dict):
        generation_dict = generation_message_obj
    else:
        return EvaluateResult(
            score=0.0,
            reason=f"Unexpected type for generation message: {type(generation_message_obj)}",
            metrics={},
        )

    if ground_truth is None:
        has_generation_tool_calls = False
        if generation_dict.get("tool_calls"):
            has_generation_tool_calls = True
        elif "<tool_call>" in generation_dict.get("content", ""):
            if parse_tool_calls(generation_dict.get("content", "")):
                has_generation_tool_calls = True

        score = 1.0 if not has_generation_tool_calls else 0.0
        reason = (
            "Ground truth not provided. Score based on absence (1.0) or presence (0.0) of tool calls in generation."
        )
        return EvaluateResult(score=score, reason=reason, metrics={})

    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            # Cast to string before slicing to satisfy type checker if ground_truth is of unknown type
            gt_preview = str(ground_truth)
            return EvaluateResult(
                score=0.0,
                reason=f"Ground truth was a string but failed to parse as JSON: {gt_preview[:100]}...",
                metrics={},
            )

    if not isinstance(ground_truth, dict):
        return EvaluateResult(
            score=0.0,
            reason=f"Ground truth is not a dictionary (even after attempting parse): {type(ground_truth)}",
            metrics={},
        )

    score = float(eval_tool_call(generation_dict, ground_truth))
    reason = f"Exact tool match evaluation score: {score}"
    return EvaluateResult(score=score, reason=reason, metrics={})


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# End of New Exact Tool Match Reward Function and Helpers


@reward_function
def schema_jaccard_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Dict[str, Any]] = None,
    function_call: Optional[Dict[str, Any]] = None,
    expected_schema: Optional[Union[Dict[str, Any], str]] = None,
    **kwargs,
) -> EvaluateResult:
    """
    DEPRECATED: This function is deprecated and will be removed in a future version.
    Please use `exact_tool_match_reward` for evaluating tool calls.

    NOTE: This function now delegates to exact_tool_match_reward.
    Original Jaccard similarity logic for function call schemas is bypassed.
    The helper functions for Jaccard similarity are kept in this file as they
    are used by eval_protocol.rewards.json_schema.py.

    Args:
        messages: List of conversation messages.
        ground_truth: Expected assistant response as a dictionary.
        function_call: Kept for signature compatibility.
        expected_schema: Kept for signature compatibility.
        **kwargs: Additional keyword arguments.

    Returns:
        EvaluateResult from exact_tool_match_reward.
    """
    warnings.warn(
        "`schema_jaccard_reward` is deprecated and will be removed in a future version. "
        "Please use `exact_tool_match_reward`.",
        DeprecationWarning,
        stacklevel=2,
    )
    return exact_tool_match_reward(messages=messages, ground_truth=ground_truth, **kwargs)


@reward_function
def llm_judge_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Dict[str, Any]] = None,
    function_call: Optional[Dict[str, Any]] = None,
    expected_schema: Optional[Union[Dict[str, Any], str]] = None,
    expected_behavior: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    **kwargs,
) -> EvaluateResult:
    """
    DEPRECATED: This function is deprecated and will be removed in a future version.
    Please use `exact_tool_match_reward` for evaluating tool calls.

    NOTE: This function now delegates to exact_tool_match_reward.
    Original LLM judge logic is bypassed.

    Args:
        messages: List of conversation messages.
        ground_truth: Expected assistant response as a dictionary.
        function_call: Kept for signature compatibility.
        expected_schema: Kept for signature compatibility.
        expected_behavior: Kept for signature compatibility.
        openai_api_key: Kept for signature compatibility.
        model: Kept for signature compatibility.
        temperature: Kept for signature compatibility.
        **kwargs: Additional keyword arguments.

    Returns:
        EvaluateResult from exact_tool_match_reward.
    """
    warnings.warn(
        "`llm_judge_reward` is deprecated and will be removed in a future version. "
        "Please use `exact_tool_match_reward`.",
        DeprecationWarning,
        stacklevel=2,
    )
    return exact_tool_match_reward(messages=messages, ground_truth=ground_truth, **kwargs)


@reward_function
def composite_function_call_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Dict[str, Any]] = None,
    function_call: Optional[Dict[str, Any]] = None,
    expected_schema: Optional[Union[Dict[str, Any], str]] = None,
    expected_behavior: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    weights: Optional[Dict[str, float]] = None,
    **kwargs,
) -> EvaluateResult:
    """
    DEPRECATED: This function is deprecated and will be removed in a future version.
    Please use `exact_tool_match_reward` for evaluating tool calls.

    This reward function now delegates to exact_tool_match_reward
    for an exact match evaluation of tool calls.
    The model's response (containing the function call) is assumed to be `messages[-1]`.

    Args:
        messages: List of conversation messages, where `messages[-1]` is the model's response.
        ground_truth: Expected assistant response as a dictionary, typically containing 'tool_calls'.
                      This is passed directly to exact_tool_match_reward.
        function_call: Kept for signature compatibility.
        expected_schema: Kept for signature compatibility.
        expected_behavior: Kept for signature compatibility.
        openai_api_key: Kept for signature compatibility.
        llm_model: Kept for signature compatibility.
        weights: Kept for signature compatibility.
        **kwargs: Additional keyword arguments passed to exact_tool_match_reward.

    Returns:
        EvaluateResult with score and metrics from exact_tool_match_reward.
    """
    warnings.warn(
        "`composite_function_call_reward` is deprecated and will be removed in a future version. "
        "Please use `exact_tool_match_reward`.",
        DeprecationWarning,
        stacklevel=2,
    )
    return exact_tool_match_reward(messages=messages, ground_truth=ground_truth, **kwargs)


# JSON schema reward functions have been moved to json_schema.py module
