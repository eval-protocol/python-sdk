import ast
import inspect
import json
import logging  # Import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from eval_protocol.agent.resources.bfcl_sim_api_resource import BFCLSimAPIResource
from eval_protocol.models import EvaluateResult, Message, MetricResult
from eval_protocol.typed_interface import reward_function

# Get logger for bfcl_reward
logger = logging.getLogger("bfcl_reward")
logger.setLevel(logging.DEBUG)  # Ensure debug logs are processed


# Helper functions adapted from BfclRubric (copied here)
def _parse_function_call(func_call_str: str):
    """
    Parses a function call string into a JSON-like dictionary.

    :param func_call_str: String representation of a function call.
    :return: JSON-like dictionary with function name and arguments.
    """
    try:
        tree = ast.parse(func_call_str, mode="eval")
        if not isinstance(tree.body, ast.Call):
            raise ValueError("Input is not a valid function call.")
        func_name = tree.body.func.id if isinstance(tree.body.func, ast.Name) else None
        if not func_name:
            raise ValueError("Could not determine function name.")
        args_dict = {}
        for kw in tree.body.keywords:
            args_dict[kw.arg] = ast.literal_eval(kw.value)
        for i, arg in enumerate(tree.body.args):
            args_dict[f"pos_arg_{i}"] = ast.literal_eval(arg)  # Standardized positional arg key

        json_obj = {"name": func_name, "args": args_dict}
        return json_obj
    except Exception:
        raise ValueError(f"Error parsing function call string: {func_call_str}")


def _are_function_calls_equivalent(call1: Dict[str, Any], call2: Dict[str, Any]) -> bool:
    """
    Compares two parsed function call dictionaries for semantic equivalence.
    Special handling for 'sort' command arguments.
    """
    if not isinstance(call1, dict) or not isinstance(call2, dict):
        logger.warning(f"Invalid input to _are_function_calls_equivalent: call1={call1}, call2={call2}")
        return False

    name1 = call1.get("name")
    name2 = call2.get("name")

    if name1 != name2:
        return False

    args1 = call1.get("args", {})
    args2 = call2.get("args", {})

    if not isinstance(args1, dict) or not isinstance(args2, dict):
        logger.warning(f"Invalid args in _are_function_calls_equivalent: args1={args1}, args2={args2}")
        return False  # Should be dicts

    if name1 == "sort":
        val1 = args1.get("pos_arg_0", args1.get("file_name"))
        val2 = args2.get("pos_arg_0", args2.get("file_name"))

        # Check if both extracted values are not None and are equal
        # And that both arg dicts have only one relevant key for the sort value
        # (to avoid matching if one has extra unrecognized args beyond the file name)

        # Condition 1: Both args have exactly one key
        cond1_holds = len(args1) == 1 and len(args2) == 1
        # Condition 2: The values associated with the (potentially different) keys are the same
        cond2_holds = val1 is not None and val1 == val2

        if cond1_holds and cond2_holds:
            return True
        # Fallback to direct equality if the specific sort logic doesn't cleanly apply
        # (e.g. if one has pos_arg_0 and the other has file_name AND other args, or different number of args)
        return args1 == args2
    else:
        # For all other functions, use direct argument dictionary equality
        return args1 == args2


def _is_subsequence_unordered(list1: List[Dict[str, Any]], list2: List[Dict[str, Any]]) -> tuple[bool, list]:
    """
    Checks if all elements of list1 are present in list2, using _are_function_calls_equivalent for comparison.
    Also returns the elements of list1 that are not present in list2.
    """
    if not list1:  # If list1 is empty, it's always a subsequence
        return True, []
    # If list1 is not empty but list2 is, list1 cannot be a subsequence.
    # This also handles the case where list2 becomes empty during processing.
    if not list2 and list1:
        return False, list1[:]

    list2_copy = list2[:]  # Make a copy to modify
    missing_elements = []

    for item1 in list1:
        found_match_in_list2 = False
        for i, item2 in enumerate(list2_copy):
            if _are_function_calls_equivalent(item1, item2):
                list2_copy.pop(i)  # Remove the matched element from list2_copy
                found_match_in_list2 = True
                break  # Move to the next item in list1
        if not found_match_in_list2:
            missing_elements.append(item1)

    is_subsequence = len(missing_elements) == 0
    return is_subsequence, missing_elements


def compare_comparable_states(model_state: Dict[str, Any], gt_state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Compares two comparable state dictionaries.
    Returns True if they match, False otherwise, and a dictionary of differences.
    """
    if set(model_state.keys()) != set(gt_state.keys()):
        return False, {
            "keys_mismatch": {
                "model_keys": list(model_state.keys()),
                "gt_keys": list(gt_state.keys()),
            }
        }

    all_match = True
    differences = {}
    for class_name in model_state.keys():
        model_instance_state = model_state[class_name]
        gt_instance_state = gt_state[class_name]

        if model_instance_state != gt_instance_state:
            all_match = False
            diffs = {
                k: (model_instance_state.get(k), gt_instance_state.get(k))
                for k in set(model_instance_state) | set(gt_instance_state)
                if model_instance_state.get(k) != gt_instance_state.get(k)
            }
            differences[class_name] = diffs

    return all_match, differences


@reward_function
def bfcl_reward(
    messages: List[Message],  # Full conversation, assistant responses are at the end
    ground_truth: Dict[str, Any],  # Contains 'function_calls' and 'comparable_state'
    state: Dict[str, Any],  # Runtime state (BFCLSimAPIResource, successful_func_calls)
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluates agent performance on BFCL tasks based on state, function calls, and format.
    """
    ground_truth_function_calls: Optional[List[List[str]]] = ground_truth.get("function_calls")
    ground_truth_comparable_state: Optional[Dict[str, Any]] = ground_truth.get("comparable_state")

    # Log ground truth data received
    logger.debug(f"Ground truth function calls from input: {ground_truth_function_calls}")
    logger.debug(f"Ground truth comparable state from input: {ground_truth_comparable_state}")

    if ground_truth_function_calls is None or ground_truth_comparable_state is None:
        return EvaluateResult(
            score=0.0,
            reason="Ground truth 'function_calls' or 'comparable_state' not found in ground_truth dict.",
            metrics={},
        )

    # Access the BFCLSimAPIResource instance from the state
    bfcl_resource: Optional[BFCLSimAPIResource] = state.get("resource")

    if not isinstance(bfcl_resource, BFCLSimAPIResource):
        return EvaluateResult(
            score=0.0,
            reason="BFCLSimAPIResource instance not found in state.",
            metrics={},
        )

    # --- State Matches Check ---
    model_comparable_state = bfcl_resource.get_comparable_state()
    state_match, state_diffs = compare_comparable_states(model_comparable_state, ground_truth_comparable_state)

    state_match_score = 0.5 if state_match else 0.0

    # --- Function Call Matches Check ---
    # model_successful_func_calls is List[List[Dict[str, Any]]], one inner list per user turn's accumulated calls
    model_successful_func_calls_per_turn = state.get("successful_func_calls", [])

    num_func_matches_for_score = 0  # Number of user turns where model's calls matched GT's calls for that turn
    func_match_score = 0.0

    num_gt_turns_with_calls = len(ground_truth_function_calls) if ground_truth_function_calls else 0
    num_model_turns_with_actual_calls = len(model_successful_func_calls_per_turn)

    # Iterate over GT turns to see if the model matched them
    # This handles cases where model makes fewer turns with calls than GT expects.
    if num_gt_turns_with_calls > 0:
        for i in range(num_gt_turns_with_calls):
            gt_calls_str_for_this_turn = ground_truth_function_calls[i]  # List[str]

            model_calls_for_this_turn = []  # List[Dict]
            if i < num_model_turns_with_actual_calls:
                model_calls_for_this_turn = model_successful_func_calls_per_turn[i]

            try:
                gt_calls_for_this_turn = [_parse_function_call(call_str) for call_str in gt_calls_str_for_this_turn]
                logger.debug(f"GT calls for turn {i}: {json.dumps(gt_calls_for_this_turn)}")
                logger.debug(f"Model calls for turn {i}: {json.dumps(model_calls_for_this_turn)}")

                is_match_for_turn, missing_gt_calls = _is_subsequence_unordered(
                    gt_calls_for_this_turn, model_calls_for_this_turn
                )
                if is_match_for_turn:
                    num_func_matches_for_score += 1
                    logger.debug(f"Turn {i} matched.")
                else:
                    logger.debug(
                        f"Turn {i} did NOT match. Missing GT calls in model's calls: {json.dumps(missing_gt_calls)}"
                    )
            except Exception as e:
                logger.error(f"Error comparing function calls for GT turn index {i}: {e}")

        if (
            num_func_matches_for_score == num_gt_turns_with_calls
            and num_model_turns_with_actual_calls == num_gt_turns_with_calls
        ):
            func_match_score = 0.5
        elif (
            num_func_matches_for_score == num_gt_turns_with_calls
            and num_model_turns_with_actual_calls != num_gt_turns_with_calls
        ):
            func_match_score = 0.0
        else:
            func_match_score = 0.0

    elif num_gt_turns_with_calls == 0:
        if num_model_turns_with_actual_calls == 0:
            func_match_score = 0.5
        else:
            func_match_score = 0.0

    reason_num_total_gt_turns_with_calls = (
        num_gt_turns_with_calls if num_gt_turns_with_calls > 0 else "0 (no GT calls expected)"
    )

    # --- Format Check (on model's response messages from the `messages` list) ---
    format_score = 0.2
    valid_assistant_messages = 0
    total_assistant_messages = 0
    assistant_message_found = False

    # Iterate over all messages to find assistant responses
    # The actual model response messages are part of the `messages` list.
    # Typically, these would be the last few messages if it's a multi-turn interaction,
    # or messages[-1] if it's a single assistant response.
    # For simplicity in format check, we scan all assistant messages in the provided `messages`.
    for msg in messages:
        if isinstance(msg, Message) and msg.role == "assistant":
            assistant_message_found = True
            total_assistant_messages += 1
            # Check for any content or any tool_call
            content_str = msg.content if isinstance(msg.content, str) else ""
            if (content_str and content_str.strip()) or msg.tool_calls:
                valid_assistant_messages += 1

    if not assistant_message_found:
        format_score = 0.0
    elif total_assistant_messages > 0 and valid_assistant_messages == 0:
        # Assistant messages were found, but none had content or tool_calls
        format_score = 0.0
    # If valid_assistant_messages > 0, format_score remains 0.2 (or could be scaled)

    # --- Combine Scores ---
    base_score = state_match_score + func_match_score

    if base_score >= 1.0:
        final_score = base_score + format_score
        reason = "State and function calls matched ground truth."
        if format_score == 0.2:
            reason += " Format was also correct."
        else:
            reason += " Format was incorrect."
    else:
        final_score = 0.0
        reason = "State or function calls did not perfectly match ground truth."
        if state_match_score < 0.5:
            reason += " State match failed."
            if state_diffs:
                reason += f" Differences: {json.dumps(state_diffs)}"
        if func_match_score < 0.5:  # Check against 0.5 as perfect score for this component
            reason += f" Function call match failed ({num_func_matches_for_score}/{reason_num_total_gt_turns_with_calls} GT turns with calls matched)."

    # Add metrics
    metrics = {}
    metrics["state_match"] = MetricResult(
        score=state_match_score,
        is_score_valid=state_match_score == 0.5,
        reason=f"State match: {state_match}" + (f", Differences: {json.dumps(state_diffs)}" if state_diffs else ""),
    )
    metrics["function_call_match"] = MetricResult(
        score=func_match_score,
        is_score_valid=func_match_score == 0.5,  # Success if it gets the full 0.5 for this part
        reason=f"{num_func_matches_for_score}/{reason_num_total_gt_turns_with_calls} GT turns with calls matched by model. Model made calls in {num_model_turns_with_actual_calls} turn(s).",
    )
    metrics["format_check"] = MetricResult(
        score=format_score,
        is_score_valid=format_score == 0.2,  # Success if it gets the full 0.2 for format
        reason=f"{valid_assistant_messages}/{total_assistant_messages} assistant messages had correct format.",
    )

    return EvaluateResult(score=final_score, reason=reason, metrics=metrics)
