"""IFEval partial credit reward function.

Score = (number of constraints satisfied) / (total constraints)
"""

import ast
import json
from typing import Any

# Import both instruction registries
# Try relative import first (when used as package), fall back to direct import
try:
    from .ifeval_registry import INSTRUCTION_DICT as IFEVAL_INSTRUCTION_DICT
    from .ifbench_registry import INSTRUCTION_DICT as IFBENCH_INSTRUCTION_DICT
except ImportError:
    from ifeval_registry import INSTRUCTION_DICT as IFEVAL_INSTRUCTION_DICT
    from ifbench_registry import INSTRUCTION_DICT as IFBENCH_INSTRUCTION_DICT

# Combine both registries: IFEval (54) + IFBench OOD (58)
INSTRUCTION_DICT = {}
INSTRUCTION_DICT.update(IFEVAL_INSTRUCTION_DICT)
INSTRUCTION_DICT.update(IFBENCH_INSTRUCTION_DICT)


def ifeval_partial_credit_reward(
    response: str,
    ground_truth: dict | str | list,
    strip_thinking: bool = True,
) -> float:
    """
    Calculate IFEval partial credit score for a response.

    Args:
        response: The model's response text.
        ground_truth: Constraint specification. Can be:
            - A dict with 'instruction_id' and 'kwargs' keys
            - A list containing such a dict
            - A JSON string encoding of the above
        strip_thinking: If True, strip <think>...</think> tags from response.

    Returns:
        Float score in [0, 1] representing fraction of constraints satisfied.

    Example:
        ground_truth = {
            "instruction_id": ["keywords:existence", "length_constraints:number_words"],
            "kwargs": [{"keywords": ["hello"]}, {"num_words": 100, "relation": "at least"}]
        }
        score = ifeval_partial_credit_reward(response, ground_truth)
    """
    if not response:
        return 0.0

    # Strip thinking tags if present
    if strip_thinking and "</think>" in response:
        response = response.split("</think>")[-1].strip()

    # Parse ground_truth
    if isinstance(ground_truth, str):
        try:
            constraint_dict = json.loads(ground_truth)
        except json.JSONDecodeError:
            constraint_dict = ast.literal_eval(ground_truth)
    else:
        constraint_dict = ground_truth

    # Handle list wrapper
    if isinstance(constraint_dict, list):
        constraint_dict = constraint_dict[0]

    # Get instruction IDs and kwargs
    instruction_keys = constraint_dict["instruction_id"]
    args_list = constraint_dict["kwargs"]

    # Check each constraint and assign partial credit
    num_satisfied = 0
    num_total = len(instruction_keys)

    for instruction_key, args in zip(instruction_keys, args_list):
        if args is None:
            args = {}
        args = {k: v for k, v in args.items() if v is not None}

        if instruction_key not in INSTRUCTION_DICT:
            # Unknown constraint, skip but count as not satisfied
            continue

        instruction_cls = INSTRUCTION_DICT[instruction_key]
        instruction_instance = instruction_cls(instruction_key)
        instruction_instance.build_description(**args)

        try:
            if response.strip() and instruction_instance.check_following(response):
                num_satisfied += 1
        except (IndexError, AttributeError, ZeroDivisionError, ValueError):
            # Library has bugs with empty/malformed/short responses
            # Treat as constraint not satisfied
            pass

    # Partial credit: fraction of constraints satisfied
    return num_satisfied / num_total if num_total > 0 else 0.0
