from typing import Any, Optional, Tuple

import dspy
from dspy.clients.lm import LM
from dspy.primitives import Example, Prediction
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback
from dspy.teleprompt.gepa.gepa import GEPAFeedbackMetric

from eval_protocol.pytest.types import TestFunction
from eval_protocol.models import EvaluationRow, EPParameters, Message


# =============================================================================
# Reflection LM configurations for GEPA
# =============================================================================

# Reflection LM configs use LiteLLM format: "provider/model_name"
# API keys should be set via environment variables:
#   - OPENAI_API_KEY for OpenAI models
#   - FIREWORKS_API_KEY for Fireworks models
#   - ANTHROPIC_API_KEY for Anthropic models

REFLECTION_LM_CONFIGS = {
    # OpenAI models
    "gpt-5": {
        "model": "openai/gpt-5",
        "temperature": 1.0,
        "max_tokens": 32000,
    },
    "gpt-4o": {
        "model": "openai/gpt-4o",
        "temperature": 1.0,
        "max_tokens": 16000,
    },
    # Anthropic models
    "claude-sonnet": {
        "model": "anthropic/claude-sonnet-4-20250514",
        "temperature": 1.0,
        "max_tokens": 16000,
    },
    # Fireworks models
    "kimi-k2": {
        "model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct-0905",
        "temperature": 0.6,
        "max_tokens": 131000,
    },
    "llama-4-maverick": {
        "model": "fireworks_ai/accounts/fireworks/models/llama4-maverick-instruct-basic",
        "temperature": 1.0,
        "max_tokens": 65536,
    },
    "deepseek-r1": {
        "model": "fireworks_ai/accounts/fireworks/models/deepseek-r1",
        "temperature": 1.0,
        "max_tokens": 65536,
    },
    "qwen3-235b": {
        "model": "fireworks_ai/accounts/fireworks/models/qwen3-235b-a22b",
        "temperature": 1.0,
        "max_tokens": 65536,
    },
}


def build_reflection_lm(reflection_lm_name: str) -> LM:
    """
    Build a DSPy LM for GEPA's reflection step.

    Args:
        reflection_lm_name: One of the predefined configs ("gpt-5", "gpt-4o",
                           "claude-sonnet", "kimi-k2-instruct-0905")
                           OR a raw LiteLLM model string (e.g., "openai/gpt-4o")

    Returns:
        A dspy.LM configured for reflection.

    Note: API keys must be set via environment variables:
        - OPENAI_API_KEY for OpenAI models
        - FIREWORKS_API_KEY for Fireworks models
        - ANTHROPIC_API_KEY for Anthropic models
    """
    if reflection_lm_name in REFLECTION_LM_CONFIGS:
        config = REFLECTION_LM_CONFIGS[reflection_lm_name]
        return dspy.LM(
            model=config["model"],
            temperature=config.get("temperature"),
            max_tokens=config.get("max_tokens"),
        )
    else:
        # Assume it's a raw LiteLLM model string
        return dspy.LM(model=reflection_lm_name)


def gold_and_pred_to_row(
    gold: Example,
    pred: Prediction,
    input_field: str = "problem",
    output_field: str = "answer",
) -> EvaluationRow:
    """
    Convert a GEPA (gold, pred) pair into an EvaluationRow for an EP `@evaluation_test`.

    Args:
        gold: The ground-truth example
        pred: The model's prediction
        input_field: Name of the input field in the DSPy signature
        output_field: Name of the output field in the DSPy signature

    Note: ground_truth is preserved in its original type (list, dict, str, etc.)
    to support structured comparisons like SQL result matching.
    """
    gt = gold.get(output_field, None)
    # Preserve original type - don't convert to string!
    # This is important for SQL evaluators that expect list[dict] results
    ground_truth = gt

    content = pred.get(output_field, "")

    return EvaluationRow(
        messages=[
            Message(role="assistant", content=str(content))
        ],  # TODO: for some evals, you might need system / user message too.
        ground_truth=ground_truth,
    )


def row_to_prediction(row: EvaluationRow) -> ScoreWithFeedback:
    """
    Convert an EvaluationRow into a GEPA-compatible ScoreWithFeedback
    (implemented as a dspy.Prediction subclass in dspy.teleprompt.gepa).
    """
    if row.evaluation_result is None:
        return dspy.Prediction(
            score=0.0,
            feedback="No evaluation_result was produced by the evaluation_test.",
        )

    score = float(row.evaluation_result.score or 0.0)
    feedback = row.evaluation_result.reason or f"This trajectory got a score of {score}."
    return dspy.Prediction(score=score, feedback=feedback)


def ep_test_to_gepa_metric(
    test_fn: TestFunction,
    input_field: str = "problem",
    output_field: str = "answer",
) -> GEPAFeedbackMetric:
    """
    Adapter: convert an EP-style `test_fn(row: EvaluationRow) -> EvaluationRow` into
    a GEPAFeedbackMetric-compatible callable.

    Args:
        test_fn: The EP evaluation test function
        input_field: Name of the input field in the DSPy signature (default: "problem")
        output_field: Name of the output field in the DSPy signature (default: "answer")

    The resulting metric:
    - Constructs an EvaluationRow from (gold, pred) using the configured field names.
    - Applies the EP test_fn to populate `row.evaluation_result`.
    - Returns a dspy.Prediction(score, feedback) derived from that result.

    Note: The @evaluation_test decorator wraps functions as async, so we need to
    handle both sync and async test functions.
    """
    import asyncio
    import inspect

    def metric(
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[DSPyTrace] = None,
    ) -> ScoreWithFeedback:
        row = gold_and_pred_to_row(gold, pred, input_field, output_field)

        # Call the test function - handle both sync and async
        result = test_fn(row)  # pyright: ignore

        # If it's a coroutine, run it synchronously
        if inspect.iscoroutine(result):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                # Already in an async context - create a new loop in a thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, result)
                    evaluated_row: EvaluationRow = future.result()
            else:
                # No running loop - safe to use asyncio.run
                evaluated_row = asyncio.run(result)
        else:
            evaluated_row = result  # pyright: ignore[reportAssignmentType]

        # TODO: this is problematic. for groupwise, we will have to extend this to handle list[EvaluationRow]

        score_result = row_to_prediction(evaluated_row)
        return score_result

    return metric


# =============================================================================
# DSPy Program Creation (maps SingleTurnRolloutProcessor → DSPy Module)
# =============================================================================

from typing import Callable, Type
from enum import Enum


class DSPyModuleType(Enum):
    """Available DSPy module types for single-turn rollouts."""

    PREDICT = "predict"  # Simple input → output
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Adds reasoning before output (good for math)
    PROGRAM_OF_THOUGHT = "program_of_thought"  # Generates code to solve problems


# Type alias for custom module factory
DSPyModuleFactory = Callable[[dspy.Signature], dspy.Module]


def create_signature(
    input_field: str = "problem",
    output_field: str = "answer",
    instructions: str | None = None,
    input_desc: str | None = None,
    output_desc: str | None = None,
) -> dspy.Signature:
    """
    Create a DSPy Signature for single-turn tasks.

    Args:
        input_field: Name of the input field (default: "problem")
        output_field: Name of the output field (default: "answer")
        instructions: System prompt / instructions (what GEPA optimizes!)
        input_desc: Description for the input field
        output_desc: Description for the output field

    Returns:
        A dspy.Signature configured for the task.
    """
    # Build signature string
    signature_str = f"{input_field} -> {output_field}"

    # Create base signature
    if instructions:
        sig = dspy.Signature(signature_str, instructions=instructions)
    else:
        sig = dspy.Signature(signature_str)

    # Add field descriptions if provided
    if input_desc:
        sig = sig.with_updated_fields(input_field, desc=input_desc)
    if output_desc:
        sig = sig.with_updated_fields(output_field, desc=output_desc)

    return sig


def create_single_turn_program(
    system_prompt: str | None = None,
    input_field: str = "problem",
    output_field: str = "answer",
    module_type: DSPyModuleType | str = DSPyModuleType.CHAIN_OF_THOUGHT,
    input_desc: str | None = None,
    output_desc: str | None = None,
    module_factory: DSPyModuleFactory | None = None,
) -> dspy.Module:
    """
    Create a DSPy program that mirrors SingleTurnRolloutProcessor.

    This is the general mapping:
    - SingleTurnRolloutProcessor: system message + user message → LLM → assistant response
    - DSPy Module: instructions + input field → LLM → output field

    GEPA optimizes the `instructions` (system prompt equivalent)!

    Args:
        system_prompt: The system prompt (becomes signature.instructions).
        input_field: Name of the input field (default: "problem")
        output_field: Name of the output field (default: "answer")
        module_type: Which DSPy module to use:
            - PREDICT: Simple input → output
            - CHAIN_OF_THOUGHT: Adds reasoning before output (default, good for complex tasks)
            - PROGRAM_OF_THOUGHT: Generates code to solve problems
        input_desc: Optional description for the input field
        output_desc: Optional description for the output field
        module_factory: Custom factory function to create the module.
                       If provided, overrides module_type.
                       Signature: (dspy.Signature) -> dspy.Module

    Returns:
        A DSPy module ready for GEPA optimization.

    Examples:
        # Default: ChainOfThought for math
        program = create_single_turn_program(system_prompt="Solve step by step")

        # Simple classification
        program = create_single_turn_program(
            input_field="text",
            output_field="label",
            module_type=DSPyModuleType.PREDICT
        )

        # Custom module
        program = create_single_turn_program(
            system_prompt="...",
            module_factory=lambda sig: MyCustomModule(sig)
        )
    """
    # Create the signature
    sig = create_signature(
        input_field=input_field,
        output_field=output_field,
        instructions=system_prompt,
        input_desc=input_desc,
        output_desc=output_desc,
    )

    # Use custom factory if provided
    if module_factory is not None:
        return module_factory(sig)

    # Convert string to enum if needed
    if isinstance(module_type, str):
        module_type = DSPyModuleType(module_type)

    # Create the appropriate module type
    if module_type == DSPyModuleType.PREDICT:
        program = dspy.Predict(sig)
    elif module_type == DSPyModuleType.CHAIN_OF_THOUGHT:
        program = dspy.ChainOfThought(sig)
    elif module_type == DSPyModuleType.PROGRAM_OF_THOUGHT:
        program = dspy.ProgramOfThought(sig)
    else:
        raise ValueError(f"Unknown module type: {module_type}")

    return program


def configure_dspy_lm(ep_params: EPParameters) -> None:
    """
    Configure DSPy to use the same LLM as the EP evaluation.

    Extracts model info from ep_params.completion_params and configures dspy.

    DSPy uses LiteLLM under the hood, so:
    - Model format: "provider/model_name" (e.g., "openai/gpt-4o", "fireworks_ai/...")
    - API keys: Set via environment variables (OPENAI_API_KEY, FIREWORKS_API_KEY, etc.)
    """
    raw_params = ep_params.completion_params

    # Handle completion_params being a list (for sweeps) - use the first one
    if isinstance(raw_params, list):
        completion_params = (raw_params[0] if raw_params else None) or {}
    else:
        completion_params = raw_params or {}

    # Extract model name (expected to already be in LiteLLM format)
    model = completion_params.get("model", "openai/gpt-4")

    # Extract optional parameters
    temperature = completion_params.get("temperature")  # None = use provider default
    max_tokens = completion_params.get("max_tokens")  # None = use provider default

    # Build kwargs - only include non-None values
    lm_kwargs: dict[str, Any] = {"model": model}
    if temperature is not None:
        lm_kwargs["temperature"] = temperature
    if max_tokens is not None:
        lm_kwargs["max_tokens"] = max_tokens

    # Pass through any extra kwargs from completion_params that DSPy/LiteLLM supports
    passthrough_keys = ["num_retries", "cache"]
    for key in passthrough_keys:
        if key in completion_params:
            lm_kwargs[key] = completion_params[key]

    lm = dspy.LM(**lm_kwargs)
    dspy.configure(lm=lm)


# =============================================================================
# Dataset Conversion (EvaluationRow → DSPy Example)
# =============================================================================


def extract_system_prompt_from_rows(rows: list[EvaluationRow]) -> str | None:
    """
    Extract the system prompt from a list of EvaluationRows.

    Assumes all rows have the same system prompt (common in benchmarks).
    Returns the first system message content found, or None.
    """
    for row in rows:
        system_msg = row.get_system_message()
        if system_msg and system_msg.content:
            content = system_msg.content
            return str(content) if content else None
    return None


def extract_user_content(row: EvaluationRow) -> str:
    """Extract the user message content from an EvaluationRow."""
    user_msg = row.get_first_user_message()
    if user_msg and user_msg.content:
        return str(user_msg.content)
    return ""


def evaluation_row_to_dspy_example(
    row: EvaluationRow,
    input_field: str = "problem",
    output_field: str = "answer",
) -> Example:
    """
    Convert an EvaluationRow to a DSPy Example.

    Maps:
    - User message content → input_field (e.g., "problem")
    - ground_truth → output_field (e.g., "answer")

    Note: ground_truth is preserved in its original type to support
    structured comparisons (e.g., SQL result matching with list[dict]).
    """
    # Extract user message as input
    input_content = extract_user_content(row)

    # Ground truth is the expected output - preserve original type!
    # Don't convert to string - this breaks SQL evaluators that expect list[dict]
    output_content = row.ground_truth if row.ground_truth is not None else ""

    return dspy.Example(
        **{
            input_field: input_content,
            output_field: output_content,
        }
    ).with_inputs(input_field)


def evaluation_rows_to_dspy_examples(
    rows: list[EvaluationRow],
    input_field: str = "problem",
    output_field: str = "answer",
) -> list[Example]:
    """Convert a list of EvaluationRows to DSPy Examples."""
    return [evaluation_row_to_dspy_example(row, input_field, output_field) for row in rows]


def train_val_test_split(
    rows: list[EvaluationRow],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[list[EvaluationRow], list[EvaluationRow], list[EvaluationRow]]:
    """
    Split EvaluationRows into train/val/test sets.

    Args:
        rows: List of EvaluationRow objects
        train_ratio: Proportion for training (default 0.8)
        val_ratio: Proportion for validation (default 0.1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_rows, val_rows, test_rows)
    """
    import random

    # Copy and shuffle
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_rows = shuffled[:train_end]
    val_rows = shuffled[train_end:val_end]
    test_rows = shuffled[val_end:]

    return train_rows, val_rows, test_rows
