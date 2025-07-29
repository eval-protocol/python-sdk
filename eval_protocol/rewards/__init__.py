"""
Out-of-the-box reward functions for common use cases.
"""

# Import specific reward functions
from . import (
    accuracy,
    accuracy_length,
    bfcl_reward,
    code_execution,
    cpp_code,
    deepcoder_reward,
    format,
    function_calling,
    json_schema,
    language_consistency,
    lean_prover,
    length,
    list_comparison_math_reward,
    math,
    multiple_choice_math_reward,
    reasoning_steps,
    repetition,
    tag_count,
)
from .accuracy_length import cosine_scaled_accuracy_length_reward

# Import function separately to avoid name conflict with the module
from .bfcl_reward import bfcl_reward as bfcl_reward_function

# Directly import specific reward functions for easy access
from .code_execution import fractional_code_reward
from .cpp_code import binary_cpp_code_reward, ioi_cpp_code_reward
from .deepcoder_reward import deepcoder_code_reward

# To make individual functions directly importable from eval_protocol.rewards
# e.g., from eval_protocol.rewards import composite_function_call_reward
from .function_calling import (
    composite_function_call_reward,
    exact_tool_match_reward,
    llm_judge_reward,
    schema_jaccard_reward,
)
from .lean_prover import (
    deepseek_huggingface_prover_benchmark,
    deepseek_prover_v2_reward,
    lean_prover_reward,
)

# Import these with aliases to avoid name conflicts
from .list_comparison_math_reward import list_comparison_math_reward as list_comparison_math_reward_function
from .multiple_choice_math_reward import multiple_choice_math_reward as multiple_choice_math_reward_function

__all__ = [
    # Modules
    "function_calling",
    "json_schema",
    "math",
    "code_execution",
    "format",
    "tag_count",
    "accuracy",
    "language_consistency",
    "reasoning_steps",
    "length",
    "repetition",
    "cpp_code",
    "accuracy_length",
    "lean_prover",
    "deepcoder_reward",
    "multiple_choice_math_reward",
    "list_comparison_math_reward",
    "bfcl_reward",
    # Specific functions for direct import
    "fractional_code_reward",
    "deepcoder_code_reward",
    "multiple_choice_math_reward_function",
    "list_comparison_math_reward_function",
    "ioi_cpp_code_reward",
    "binary_cpp_code_reward",
    "cosine_scaled_accuracy_length_reward",
    "lean_prover_reward",
    "deepseek_prover_v2_reward",
    "deepseek_huggingface_prover_benchmark",
    "bfcl_reward_function",
    "composite_function_call_reward",
    "exact_tool_match_reward",
    "llm_judge_reward",
    "schema_jaccard_reward",
]
