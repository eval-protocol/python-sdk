"""
Utility functions for integrating reward-kit reward functions with TRL.

This module provides helper functions for:
1. Converting reward-kit reward functions to TRL-compatible format
2. Combining multiple reward functions with weights
3. Creating GRPO-specific format rewards
"""

import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Union

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from eval_protocol.models import EvaluateResult, MetricResult

# Import reward-kit components
from eval_protocol.reward_function import RewardFunction, reward_function


def create_combined_reward(
    reward_functions: List[RewardFunction],
    weights: Optional[List[float]] = None,
    normalize: bool = True,
) -> Callable:
    """
    Combine multiple reward functions with optional weights.

    Args:
        reward_functions: List of RewardFunction instances
        weights: Optional weights for each reward function
        normalize: Whether to normalize weights to sum to 1.0

    Returns:
        A callable function compatible with TRL
    """
    # Validate inputs
    if len(reward_functions) == 0:
        raise ValueError("Must provide at least one reward function")

    # Normalize weights if provided
    if weights:
        if len(weights) != len(reward_functions):
            raise ValueError("Number of weights must match number of reward functions")
        if normalize:
            weight_sum = sum(weights)
            if weight_sum != 1.0:
                weights = [w / weight_sum for w in weights]
    else:
        # Equal weights for all reward functions
        weights = [1.0 / len(reward_functions) for _ in range(len(reward_functions))]

    # Create adapters for each reward function
    trl_adapters = [rf.get_trl_adapter() for rf in reward_functions]

    def combined_adapter_for_trl(
        prompts: List[List[Dict]], completions: List[str], **kwargs
    ) -> List[float]:
        """
        Combined adapter function that works with TRL.
        It now accepts prompts and completions separately.
        """
        all_individual_scores: List[List[float]] = []

        # Each adapter in trl_adapters expects (prompts, completions, **kwargs)
        for adapter_func in trl_adapters:
            # Pass the full batch of prompts and completions to each underlying adapter
            individual_reward_scores = adapter_func(
                prompts=prompts, completions=completions, **kwargs
            )
            all_individual_scores.append(individual_reward_scores)

        if not all_individual_scores or not all_individual_scores[0]:
            # This case should ideally not happen if prompts/completions are non-empty
            # and adapters return valid score lists.
            return [0.0] * len(prompts)  # Return a list of zeros of appropriate length

        # Combine weighted scores for each sample in the batch
        num_samples = len(all_individual_scores[0])
        final_combined_scores: List[float] = []

        for i in range(num_samples):
            weighted_sum_for_sample = 0.0
            for adapter_idx, individual_scores_list in enumerate(all_individual_scores):
                if i < len(individual_scores_list):  # Check bounds
                    weighted_sum_for_sample += (
                        individual_scores_list[i] * weights[adapter_idx]
                    )
                else:
                    # Handle potential mismatch in lengths of score lists from different adapters, though unlikely
                    # For robustness, could add a default score or raise an error
                    weighted_sum_for_sample += (
                        0.0  # Or some other default/error handling
                    )
            final_combined_scores.append(weighted_sum_for_sample)

        return final_combined_scores

    return combined_adapter_for_trl


@reward_function
def grpo_format_reward(
    messages: List[Dict[str, Any]],
    ground_truth: Optional[
        List[Dict[str, Any]]
    ] = None,  # Changed from original_messages
    think_tag: str = "<think>",
    answer_tag: str = "<answer>",
    **kwargs,
) -> EvaluateResult:
    """
    Reward function that checks if the completion has the GRPO specific format.

    Args:
        messages: List of conversation messages
        ground_truth: Optional ground truth context (not used by this specific function).
        think_tag: Tag to use for reasoning (default: "<think>")
        answer_tag: Tag to use for answers (default: "<answer>")

    Returns:
        EvaluateResult with score based on format compliance
    """
    # Get the assistant's message
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            is_score_valid=False,
            metrics={
                "format": MetricResult(
                    score=0.0, is_score_valid=False, reason="No messages provided"
                )
            },
        )

    # Extract response text from last message (assistant's response)
    response = messages[-1]
    if response.get("role") != "assistant" or not response.get("content"):
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found",
            is_score_valid=False,
            metrics={
                "format": MetricResult(
                    score=0.0, is_score_valid=False, reason="No assistant response"
                )
            },
        )

    text = response.get("content", "")

    # Check for think/answer tags
    think_pattern = (
        f"{re.escape(think_tag)}(.*?){re.escape(think_tag.replace('<', '</'))}"
    )
    answer_pattern = (
        f"{re.escape(answer_tag)}(.*?){re.escape(answer_tag.replace('<', '</'))}"
    )

    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)

    has_think = bool(think_match)
    has_answer = bool(answer_match)

    # Check for correct order (think should come before answer)
    correct_order = True
    if has_think and has_answer:
        think_pos = text.find(think_tag)
        answer_pos = text.find(answer_tag)
        correct_order = think_pos < answer_pos

    # Calculate score based on format compliance
    if has_think and has_answer and correct_order:
        score = 1.0
        reason = "Format is compliant with think/answer tags in correct order"
    elif has_think and has_answer:
        score = 0.5
        reason = "Has both think and answer tags but in incorrect order"
    elif has_think:
        score = 0.3
        reason = "Has think tag but missing answer tag"
    elif has_answer:
        score = 0.2
        reason = "Has answer tag but missing think tag"
    else:
        score = 0.0
        reason = "Missing both think and answer tags"

    # Create metrics
    metrics = {
        "has_think": MetricResult(
            score=1.0 if has_think else 0.0,
            is_score_valid=has_think,
            reason=f"{'Has' if has_think else 'Missing'} think tag",
        ),
        "has_answer": MetricResult(
            score=1.0 if has_answer else 0.0,
            is_score_valid=has_answer,
            reason=f"{'Has' if has_answer else 'Missing'} answer tag",
        ),
        "correct_order": MetricResult(
            score=1.0 if correct_order else 0.0,
            is_score_valid=correct_order,
            reason=f"Tags are in {'correct' if correct_order else 'incorrect'} order",
        ),
    }

    return EvaluateResult(
        score=score, reason=reason, metrics=metrics, is_score_valid=score > 0.0
    )


def create_grpo_reward(
    content_reward: RewardFunction,
    format_weight: float = 0.3,
    content_weight: float = 0.7,
    think_tag: str = "<think>",
    answer_tag: str = "<answer>",
) -> Callable:
    """
    Create a combined reward function for GRPO-style training.

    Args:
        content_reward: RewardFunction for content quality (accuracy, helpfulness, etc.)
        format_weight: Weight for format compliance (default: 0.3)
        content_weight: Weight for content quality (default: 0.7)
        think_tag: Tag to use for reasoning (default: "<think>")
        answer_tag: Tag to use for answers (default: "<answer>")

    Returns:
        A callable function compatible with GRPO
    """
    # Create format reward function
    format_rf = RewardFunction(
        func=grpo_format_reward, think_tag=think_tag, answer_tag=answer_tag
    )

    # Combine rewards
    return create_combined_reward(
        reward_functions=[format_rf, content_reward],
        weights=[format_weight, content_weight],
        normalize=True,
    )


def prepare_grpo_message_format(
    text: str, system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Convert a text response to a message format for GRPO evaluation.

    Args:
        text: The model's text response
        system_prompt: Optional system prompt for context

    Returns:
        List of messages in the format expected by reward functions
    """
    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add a default user prompt if there isn't one
    messages.append({"role": "user", "content": "Respond to this prompt."})

    # Add the response as assistant message
    messages.append({"role": "assistant", "content": text})

    return messages


def apply_reward_to_responses(
    reward_function: Union[RewardFunction, Callable],
    responses: List[str],
    system_prompt: Optional[str] = None,
) -> List[float]:
    """
    Apply a reward function to a list of text responses.

    Args:
        reward_function: RewardFunction or callable
        responses: List of model response strings
        system_prompt: Optional system prompt to include

    Returns:
        List of reward scores
    """
    # Convert responses to message format
    message_batches = []
    for response in responses:
        user_message = {"role": "user", "content": "Evaluate this response."}
        assistant_message = {"role": "assistant", "content": response}

        # Create a conversation with system, user, and assistant messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(user_message)
        messages.append(assistant_message)

        message_batches.append(messages)

    # Check if we need to get the adapter
    if isinstance(reward_function, RewardFunction):
        adapter = reward_function.get_trl_adapter()
    else:
        # Assume it's already an adapter
        adapter = reward_function

    # Apply the adapter
    # We need to separate prompts and completions from message_batches
    # For apply_reward_to_responses, the 'prompt' is effectively just the system_prompt (if any)
    # and the 'completion' is the response string.

    prompts_batch: List[List[Dict[str, str]]] = []
    completions_batch: List[str] = []

    for i, response_str in enumerate(responses):
        # The prompt part for each sample. If system_prompt is given, it's the prompt.
        # Otherwise, the prompt is empty (adapter should handle this).
        current_prompt_messages: List[Dict[str, str]] = []
        if system_prompt:
            current_prompt_messages.append({"role": "system", "content": system_prompt})
        # If there were other "user" turns before the response, they would go here.
        # But apply_reward_to_responses is simpler and assumes response is standalone or with system prompt.

        prompts_batch.append(current_prompt_messages)
        completions_batch.append(response_str)

    return adapter(prompts=prompts_batch, completions=completions_batch)


# Example usage
if __name__ == "__main__":
    # Test the functions with a simple example
    from eval_protocol.rewards.length import length_reward

    # Create a length reward function
    length_rf = RewardFunction(func=length_reward)

    # Create a format reward function
    format_rf = RewardFunction(func=grpo_format_reward)

    # Combine them
    combined_reward = create_combined_reward(
        reward_functions=[format_rf, length_rf], weights=[0.4, 0.6]
    )

    # Create a GRPO-style reward
    grpo_reward = create_grpo_reward(length_rf)

    # Test with some example responses
    test_responses = [
        "<think>This is my reasoning</think><answer>This is my answer</answer>",
        "This is a response without tags",
        "<answer>Answer first</answer><think>Think second</think>",
    ]

    # Apply rewards
    format_scores = apply_reward_to_responses(format_rf, test_responses)
    length_scores = apply_reward_to_responses(length_rf, test_responses)
    combined_scores = apply_reward_to_responses(combined_reward, test_responses)
    grpo_scores = apply_reward_to_responses(grpo_reward, test_responses)

    # Print results
    print("Format scores:", format_scores)
    print("Length scores:", length_scores)
    print("Combined scores:", combined_scores)
    print("GRPO scores:", grpo_scores)
