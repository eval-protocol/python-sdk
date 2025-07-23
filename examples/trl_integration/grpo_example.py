"""
Example demonstrating how to use reward-kit reward functions with TRL's GRPO trainer.

This example shows how to:
1. Define reward functions in reward-kit
2. Convert them to TRL-compatible format
3. Use them with the GRPO trainer
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from eval_protocol.integrations.trl import (  # Import the new generic TRL adapter
    create_trl_adapter,
)
from eval_protocol.models import EvaluateResult, MetricResult

# Import reward-kit components
from eval_protocol.reward_function import (  # RewardFunction class no longer needed for this example's core logic
    reward_function,
)

# Try to import TRL components
try:
    import math_verify  # Import math_verify, may become unused by this script but kept for plan alignment
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from eval_protocol.rewards.math import (
        math_reward as rk_math_reward,  # Import the library's math_reward
    )

    HAS_TRL = True
except ImportError:
    print(
        "TRL or related packages not installed. Install with: pip install 'reward-kit[trl]' math_verify"
    )
    HAS_TRL = False


# Define reward functions compatible with reward-kit
@reward_function
def format_reward(
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
            metrics={
                "format": MetricResult(
                    score=0.0, success=False, reason="No messages provided"
                )
            },
        )

    # Extract response text from last message (assistant's response)
    # After @reward_function decoration, 'messages' is List[Message] (Pydantic models)
    response = messages[-1]
    # Handle both dict and Message object formats
    if isinstance(response, dict):
        role = response.get("role", "")
        content = response.get("content", "")
    else:
        role = response.role
        content = response.content
    if role != "assistant" or not content:
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found",
            metrics={
                "format": MetricResult(
                    score=0.0, success=False, reason="No assistant response"
                )
            },
        )

    text = content if content is not None else ""

    # Check for think/answer tags
    think_pattern = (
        f"{re.escape(think_tag)}(.*?){re.escape(think_tag.replace('<', '</'))}"
    )
    answer_pattern = (
        f"{re.escape(answer_tag)}(.*?){re.escape(answer_tag.replace('<', '</'))}"
    )
    boxed_answer_pattern = r"\\boxed\{.*?}"  # Pattern for \boxed{...}

    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)

    has_think = bool(think_match)
    has_answer_tag = bool(answer_match)
    has_boxed_in_answer = False

    if has_answer_tag and answer_match:  # Ensure answer_match is not None
        answer_content = answer_match.group(1)
        if answer_content:
            has_boxed_in_answer = bool(re.search(boxed_answer_pattern, answer_content))

    # Check for correct order (think should come before answer)
    correct_order = True
    if has_think and has_answer_tag:
        think_pos = text.find(think_tag)
        answer_pos = text.find(answer_tag)
        correct_order = think_pos < answer_pos

    # Calculate score based on format compliance
    # Max score if all conditions met
    if has_think and has_answer_tag and has_boxed_in_answer and correct_order:
        score = 1.0
        reason = "Compliant: think, answer tags in order, and boxed answer present."
    # Penalize if boxed answer is missing
    elif has_think and has_answer_tag and correct_order:
        score = 0.8
        reason = "Partial: think, answer tags in order, but boxed answer missing."
    # Penalize for incorrect order
    elif has_think and has_answer_tag and has_boxed_in_answer:
        score = 0.6
        reason = (
            "Partial: think, answer, boxed answer tags present, but incorrect order."
        )
    # Further penalizations for missing tags
    elif has_think and has_answer_tag:  # Incorrect order, no boxed
        score = 0.4
        reason = (
            "Partial: think, answer tags present, incorrect order, no boxed answer."
        )
    elif (
        has_think and has_boxed_in_answer
    ):  # Missing answer tag but has think and boxed (unlikely)
        score = 0.3
        reason = "Partial: has think and boxed answer, but missing answer tag."
    elif has_answer_tag and has_boxed_in_answer:  # Missing think tag
        score = 0.2
        reason = "Partial: has answer tag with boxed answer, but missing think tag."
    elif has_think:
        score = 0.1
        reason = "Poor: Has think tag but missing answer tag and boxed answer."
    elif has_answer_tag:  # Has answer tag but no boxed and no think
        score = 0.05
        reason = "Poor: Has answer tag but missing think tag and boxed answer."
    else:
        score = 0.0
        reason = "Non-compliant: Missing think, answer, and boxed answer."

    # Create metrics
    metrics = {
        "has_think_tag": MetricResult(
            score=1.0 if has_think else 0.0,
            success=has_think,
            reason=f"{'Has' if has_think else 'Missing'} think tag",
        ),
        "has_answer_tag": MetricResult(
            score=1.0 if has_answer_tag else 0.0,
            success=has_answer_tag,
            reason=f"{'Has' if has_answer_tag else 'Missing'} answer tag",
        ),
        "has_boxed_in_answer": MetricResult(
            score=1.0 if has_boxed_in_answer else 0.0,
            success=has_boxed_in_answer,
            reason=f"{'Has' if has_boxed_in_answer else 'Missing'} boxed answer within answer tag",
        ),
        "correct_tag_order": MetricResult(
            score=1.0 if correct_order else 0.0,
            success=correct_order,
            reason=f"Tags are in {'correct' if correct_order else 'incorrect'} order (if both present)",
        ),
    }

    return EvaluateResult(score=score, reason=reason, metrics=metrics)


# The local math_accuracy_reward function is now removed, replaced by rk_math_reward from the library.


# Helper function to extract user content from a list of message dicts
def _extract_user_content_from_messages(prompt_msg_list: List[Dict[str, str]]) -> str:
    for msg in reversed(prompt_msg_list):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def combine_rewards_with_new_adapter(
    reward_configs: List[
        Dict[str, Any]
    ],  # Each dict contains 'func', 'map', 'static_kwargs'
    weights: Optional[List[float]] = None,
):
    """
    Combine multiple reward functions using the new create_trl_adapter.
    Args:
        reward_configs: A list of dictionaries, each configuring one reward function.
                        Each dict should have:
                        - 'func': The raw reward function.
                        - 'map': dataset_to_reward_kwargs_map for this function.
                        - 'static_kwargs': static_reward_kwargs for this function.
                        - 'user_msg_fn' (optional): custom user_message_fn for this function.
        weights: Optional weights for each reward function.
    Returns:
        A callable function compatible with TRL.
    """
    if not reward_configs:
        raise ValueError("Must provide at least one reward function configuration.")

    if weights:
        if len(weights) != len(reward_configs):
            raise ValueError("Number of weights must match number of reward functions.")
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-6:  # Allow for small floating point inaccuracies
            print(f"Normalizing weights from sum {weight_sum} to 1.0")
            weights = [w / weight_sum for w in weights]
    else:
        weights = [1.0 / len(reward_configs)] * len(reward_configs)

    adapters = []
    for config in reward_configs:
        user_msg_fn_to_use = config.get(
            "user_msg_fn", _extract_user_content_from_messages
        )

        adapter = create_trl_adapter(
            reward_fn=config["func"],
            dataset_to_reward_kwargs_map=config["map"],
            static_reward_kwargs=config.get("static_kwargs", {}),
            user_message_fn=user_msg_fn_to_use,
            # assistant_message_fn can be None for default behavior
        )
        adapters.append(adapter)

    def combined_adapter(prompts: List[Any], completions: List[str], **kwargs):
        """
        Combined adapter function compatible with TRL's expected signature.
        'prompts' here will be List[List[Dict[str, str]]] from this example's dataset.
        """
        if len(prompts) != len(completions):
            raise ValueError("Length of prompts and completions must match.")

        all_scores = []
        for adapter_fn in adapters:
            # Each adapter created by create_trl_adapter expects this signature
            scores = adapter_fn(prompts, completions, **kwargs)
            all_scores.append(scores)

        combined_scores = []
        num_samples = len(completions)
        for i in range(num_samples):
            weighted_sum = sum(
                all_scores[adapter_idx][i] * weight
                for adapter_idx, weight in enumerate(weights)
            )
            combined_scores.append(weighted_sum)
        return combined_scores

    return combined_adapter


def prepare_dataset_for_trl(
    dataset_name,
    split="train",
    prompt_key=None,
    response_key=None,
    system_prompt=None,
    max_samples=None,
):
    """
    Prepare a HuggingFace dataset for use with TRL.

    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to use
        prompt_key: Key for the prompt content
        response_key: Key for the response content
        system_prompt: Optional system prompt to prepend
        max_samples: Maximum samples to include

    Returns:
        Dataset in TRL-compatible format
    """
    if not HAS_TRL:
        print(
            "TRL or related packages not installed. Install with: pip install 'reward-kit[trl]'"
        )
        return None

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    # Limit samples if specified
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    # Default keys for prompt and response
    if prompt_key is None:
        # Try to guess based on common patterns
        for key in ["problem", "question", "input", "prompt"]:
            if key in dataset.features:
                prompt_key = key
                break
        if prompt_key is None:
            raise ValueError(
                "Could not determine prompt key. Please specify prompt_key."
            )

    if response_key is None:
        # Try to guess based on common patterns
        for key in ["solution", "answer", "output", "response"]:
            if key in dataset.features:
                response_key = key
                break
        if response_key is None:
            raise ValueError(
                "Could not determine response key. Please specify response_key."
            )

    # Prepare GRPO style system prompt
    if system_prompt is None:
        # System prompt from Open R1 blog:
        system_prompt = (
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        # We also need to instruct about <think> and <answer> tags for the format reward.
        # Let's combine this with the GRPO structure.
        system_prompt = (
            f"{system_prompt} The assistant first thinks about the reasoning process in the mind and then provides the user "
            "with the answer. The reasoning process and answer are enclosed within <think> </think> and "
            "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>"
            "<answer> \\boxed{answer here} </answer>"  # Show example of boxed answer in the final tag structure
        )

    # Create the dataset in the format expected by TRL
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example[prompt_key]},
            ],
            "solution": example[response_key],
        }

    formatted_dataset = dataset.map(make_conversation)

    # Remove unnecessary columns but keep solution for reward function
    cols_to_remove = [
        col
        for col in formatted_dataset.column_names
        if col not in ["prompt", "solution"]
    ]
    if cols_to_remove:
        formatted_dataset = formatted_dataset.remove_columns(cols_to_remove)

    return formatted_dataset


def train_with_grpo_example():
    """
    Example of training with GRPO using reward-kit reward functions.
    """
    if not HAS_TRL:
        print(
            "TRL or related packages not installed. Install with: pip install 'reward-kit[trl]'"
        )
        return

    print("Setting up GRPO training with reward-kit reward functions...")

    # 1. Define reward function configurations (no RewardFunction class needed here)
    # format_reward is defined locally. rk_math_reward is imported.

    # Configuration for format_reward
    format_reward_config = {
        "func": format_reward,
        "map": {},  # No dynamic kwargs from dataset needed for format_reward
        "static_kwargs": {
            "think_tag": "<think>",
            "answer_tag": "<answer>",
        },  # Default tags
        # user_msg_fn will default to _extract_user_content_from_messages
    }

    # Configuration for rk_math_reward
    # rk_math_reward's 'ground_truth' parameter will be mapped from the 'solution' column of the dataset.
    math_reward_config = {
        "func": rk_math_reward,
        "map": {
            "solution": "ground_truth"
        },  # Map 'solution' dataset column to 'ground_truth' param for rk_math_reward
        "static_kwargs": {},
        # user_msg_fn will default to _extract_user_content_from_messages
    }

    # 2. Prepare dataset
    try:
        print("Preparing dataset...")
        # Using OpenR1-Math-220k as per user choice and blog
        # The blog mentions a "default" split with 94k problems.
        # Error indicated 'train' is the available split.
        dataset = prepare_dataset_for_trl(
            dataset_name="open-r1/OpenR1-Math-220k",
            split="train",  # Using the 'train' split
            prompt_key="problem",  # Assuming 'problem' and 'solution' keys are consistent or handled by prepare_dataset_for_trl
            response_key="solution",
            max_samples=1000,  # Use a larger subset for more meaningful initial runs
        )
        print(
            f"Dataset prepared: {len(dataset)} samples from open-r1/OpenR1-Math-220k (train split)"
        )

    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        return

    # 3. Load model (would be done for actual training)
    # Set to True to enable model loading and training when ready
    ACTUALLY_TRAIN_FLAG = True  # Re-enabled training
    if ACTUALLY_TRAIN_FLAG:  # ACTUALLY_TRAIN_FLAG - Enabling training
        print("Loading model...")
        model_id = "Qwen/Qwen2-7B-Instruct"  # Updated model ID
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",  # Or torch.bfloat16 for H100s
            device_map="auto",
            rope_theta=300000.0,  # For 32k context length, as per blog
        )

        # Configure LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)

    # 4. Configure GRPO training
    training_args = GRPOConfig(
        output_dir="./trl-output",
        learning_rate=5e-5,  # Updated LR from blog
        remove_unused_columns=False,  # Keep solution column for reward function
        gradient_accumulation_steps=16,  # Keep or adjust based on hardware
        num_train_epochs=3,  # Updated epochs from blog
        max_completion_length=2048,  # Increased, blog mentions 16k limit for data gen, 8k for 75%
        num_generations=4,  # Keep or adjust
        max_prompt_length=1024,  # Increased, ensure it accommodates problem statements
        report_to=["tensorboard"],
        logging_steps=10,
        push_to_hub=False,
        save_strategy="steps",
        save_steps=100,  # Save less frequently for larger datasets/longer training
        lr_scheduler_type="linear",  # Linear scheduler
        warmup_ratio=0.1,  # 10% warmup as per blog
        per_device_train_batch_size=1,  # Adjust based on GPU memory
        # per_device_eval_batch_size=1, # If evaluation is added
        # evaluation_strategy="steps", # If evaluation is added
        # eval_steps=100, # If evaluation is added
        # max_steps=5, # Removed to allow training for num_train_epochs
    )

    # 5. Combine reward functions for TRL using the new adapter logic
    print("Creating combined reward function using new adapter...")
    combined_reward_new = combine_rewards_with_new_adapter(
        reward_configs=[format_reward_config, math_reward_config],
        weights=[0.3, 0.7],  # Format is 30%, accuracy is 70%
    )

    # 6. Create and run trainer (would be done for actual training)
    # Set to True to enable model loading and training when ready
    if ACTUALLY_TRAIN_FLAG:  # ACTUALLY_TRAIN_FLAG - Enabling training
        print("Creating GRPO trainer...")
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[combined_reward_new],  # Use the new combined reward
            args=training_args,
            train_dataset=dataset,
        )

        print("Starting training...")
        trainer.train()

        print("Training complete!")

    print(
        "\nExample completed successfully. In a real scenario, the training would now run."
    )
    print(
        "This example shows how reward-kit reward functions can be adapted for TRL's GRPO trainer."
    )

    # Print dataset sample to show the format
    print("\nDataset format example (first sample):")
    first_dataset_sample = dataset[0]
    print(first_dataset_sample)

    # Show how reward functions would be called on a real dataset sample
    print("\nReward function test on the first actual dataset sample:")

    # Construct messages for the reward function using the first dataset sample
    # The 'prompt' field in the dataset is already a list of messages (system, user)
    actual_sample_prompt_messages = first_dataset_sample["prompt"]

    # For testing, let's create a mock assistant response.
    # A perfect response would use the dataset's 'solution' and format it correctly.
    ground_truth_full_solution_text = first_dataset_sample["solution"]

    # Extract the actual final answer from the ground_truth_full_solution_text for math_verify
    # rk_math_reward will parse the full text of the assistant's response and compare
    # its extracted answer against the `ground_truth` parameter (which is ground_truth_full_solution_text here).
    # Context, if needed by rk_math_reward, is derived from the `messages` parameter.

    # The warning about no \boxed in GT is still relevant if the dataset often lacks it.
    boxed_gt_match_in_solution = re.search(
        r"\\boxed\{(.*?)\}", ground_truth_full_solution_text
    )
    if not boxed_gt_match_in_solution:
        print(
            f"Informational: Ground truth solution for sample does not contain \\boxed{{...}}: {ground_truth_full_solution_text[:100]}..."
        )

    # Mock assistant responses. For rk_math_reward, the comparison will be against numbers extracted from ground_truth_full_solution_text.
    # Let's use a known number from the example solution if possible for mock model answer.
    # The example solution is: "## Solution.\n\nLet $t$ be the time required... speed of the river is $v_{R}=4 \\mathrm{~km} / \\mathrm{h}$, and the speed of the boat is $v_{B}=10 \\mathrm{~km} / \\mathrm{h}$."
    # Let's assume the target answer for the mock is 10 (boat speed) or 4 (river speed).
    # For the mock, we'll use "10" as the boxed answer.
    mock_model_boxed_answer_val = "10"

    mock_assistant_content_perfect = f"<think>Some reasoning based on problem: {first_dataset_sample['prompt'][-1]['content'][:50]}...</think><answer>\\boxed{{{mock_model_boxed_answer_val}}}</answer>"
    mock_assistant_content_no_box = f"<think>Some reasoning...</think><answer>{mock_model_boxed_answer_val}</answer>"  # Missing box
    mock_assistant_content_no_think = (
        f"<answer>\\boxed{{{mock_model_boxed_answer_val}}}</answer>"  # Missing think
    )

    # The `original_messages_for_test` variable is no longer needed as rk_math_reward
    # does not take an `original_messages` parameter.

    # Test case 1: "Perfect" format
    print("\n--- Test Case 1: Mock 'Perfect' Format ---")
    messages_test_case_1 = actual_sample_prompt_messages + [
        {"role": "assistant", "content": mock_assistant_content_perfect}
    ]
    # Call raw functions for testing, as they are already decorated by @reward_function
    format_result_1 = format_reward(
        messages_test_case_1, think_tag="<think>", answer_tag="<answer>"
    )
    accuracy_result_1 = rk_math_reward(
        messages=messages_test_case_1,
        ground_truth=ground_truth_full_solution_text,
    )
    print(
        f"Format reward (perfect mock): {format_result_1.score} - {format_result_1.get('reason', 'N/A')}"
    )
    print(
        f"Accuracy reward (perfect mock): {accuracy_result_1.score} - {accuracy_result_1.get('reason', 'N/A')}"
    )
    combined_score_1 = 0.3 * format_result_1.score + 0.7 * accuracy_result_1.score
    print(f"Combined reward (perfect mock): {combined_score_1}")

    # Test case 2: No box in answer (model's response)
    print("\n--- Test Case 2: Mock 'No Box in Answer' ---")
    messages_test_case_2 = actual_sample_prompt_messages + [
        {"role": "assistant", "content": mock_assistant_content_no_box}
    ]
    format_result_2 = format_reward(
        messages_test_case_2, think_tag="<think>", answer_tag="<answer>"
    )
    accuracy_result_2 = rk_math_reward(
        messages=messages_test_case_2,
        ground_truth=ground_truth_full_solution_text,
    )
    print(
        f"Format reward (no box): {format_result_2.score} - {format_result_2.get('reason', 'N/A')}"
    )
    print(
        f"Accuracy reward (no box): {accuracy_result_2.score} - {accuracy_result_2.get('reason', 'N/A')}"
    )
    combined_score_2 = 0.3 * format_result_2.score + 0.7 * accuracy_result_2.score
    print(f"Combined reward (no box): {combined_score_2}")

    # Test case 3: No think tag
    print("\n--- Test Case 3: Mock 'No Think Tag' ---")
    messages_test_case_3 = actual_sample_prompt_messages + [
        {"role": "assistant", "content": mock_assistant_content_no_think}
    ]
    format_result_3 = format_reward(
        messages_test_case_3, think_tag="<think>", answer_tag="<answer>"
    )
    accuracy_result_3 = rk_math_reward(
        messages=messages_test_case_3,
        ground_truth=ground_truth_full_solution_text,
    )
    print(
        f"Format reward (no think): {format_result_3.score} - {format_result_3.get('reason', 'N/A')}"
    )
    print(
        f"Accuracy reward (no think): {accuracy_result_3.score} - {accuracy_result_3.get('reason', 'N/A')}"
    )
    combined_score_3 = 0.3 * format_result_3.score + 0.7 * accuracy_result_3.score
    print(f"Combined reward (no think): {combined_score_3}")


if __name__ == "__main__":
    train_with_grpo_example()
