"""
Working example demonstrating integration of reward-kit with TRL's GRPO trainer.

This example:
1. Uses a real (but small) math dataset from HuggingFace
2. Sets up the GRPO trainer with reward-kit reward functions
3. Runs a minimal training procedure to validate the integration
"""

import logging
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from eval_protocol.integrations.trl import (  # Import the new generic TRL adapter
    create_trl_adapter,
)
from eval_protocol.models import (  # RewardOutput, MetricRewardOutput are legacy
    EvaluateResult,
    MetricResult,
)

# Import reward-kit components
from eval_protocol.reward_function import (  # RewardFunction class might not be needed
    reward_function,
)

# Import TRL components
try:
    import torch
    from datasets import load_dataset
    from math_verify import LatexExtractionConfig, parse, verify
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    HAS_TRL = True
except ImportError as e:
    print(f"Could not import TRL-related packages: {e}")
    print("Install with: pip install 'reward-kit[trl]' math_verify")
    HAS_TRL = False


# Define reward functions compatible with reward-kit
@reward_function
def format_reward(
    messages: List[Dict[str, Any]],
    ground_truth: Optional[List[Dict[str, Any]]] = None,  # Changed from original_messages
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
            metrics={"format": MetricResult(score=0.0, success=False, reason="No messages provided")},
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
            metrics={"format": MetricResult(score=0.0, success=False, reason="No assistant response")},
        )

    text = content if content is not None else ""

    # Check for think/answer tags
    think_pattern = f"{re.escape(think_tag)}(.*?){re.escape(think_tag.replace('<', '</'))}"
    answer_pattern = f"{re.escape(answer_tag)}(.*?){re.escape(answer_tag.replace('<', '</'))}"

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
            success=has_think,
            reason=f"{'Has' if has_think else 'Missing'} think tag",
        ),
        "has_answer": MetricResult(
            score=1.0 if has_answer else 0.0,
            success=has_answer,
            reason=f"{'Has' if has_answer else 'Missing'} answer tag",
        ),
        "correct_order": MetricResult(
            score=1.0 if correct_order else 0.0,
            success=correct_order,
            reason=f"Tags are in {'correct' if correct_order else 'incorrect'} order",
        ),
    }

    return EvaluateResult(score=score, reason=reason, metrics=metrics)


@reward_function
def math_accuracy_reward(
    messages: List[Dict[str, Any]],
    ground_truth: Optional[List[Dict[str, Any]]] = None,  # Changed from original_messages, marked as unused
    solution: Optional[str] = None,  # This is the primary ground truth for this function
    **kwargs,
) -> EvaluateResult:
    """
    Reward function that checks math solution accuracy using math_verify.

    Args:
        messages: List of conversation messages
        ground_truth: Optional ground truth context (not used by this specific function).
        solution: Expected solution/answer string.

    Returns:
        EvaluateResult with score based on solution accuracy
    """
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={"accuracy": MetricResult(score=0.0, success=False, reason="No messages provided")},
        )

    # Extract response text
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
            metrics={"accuracy": MetricResult(score=0.0, success=False, reason="No assistant response")},
        )

    text = content if content is not None else ""

    # If solution is not provided, we can't evaluate accuracy
    if not solution:
        return EvaluateResult(
            score=0.0,
            reason="No solution provided for comparison",
            metrics={"accuracy": MetricResult(score=0.0, success=False, reason="No solution provided")},
        )

    try:
        # Parse reference solution
        gold_parsed = parse(
            solution,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        # Parse model's answer
        answer_parsed = parse(
            text,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        # Calculate score
        if len(gold_parsed) != 0 and len(answer_parsed) != 0:
            score = float(verify(answer_parsed, gold_parsed))
            reason = f"Verification score: {score:.2f}"
            success = score > 0.9
        else:
            # Couldn't extract mathematical expressions
            score = 0.0
            reason = "Failed to extract mathematical expressions"
            success = False
    except Exception as e:
        # Handle errors in parsing/verification
        logger.warning(f"Error in math verification: {str(e)}")
        score = 0.0
        reason = f"Error in verification: {str(e)}"
        success = False

    return EvaluateResult(
        score=score,
        reason=reason,
        metrics={"accuracy": MetricResult(score=score, success=success, reason=reason)},
    )


def run_grpo_training_example():
    """
    Run a minimal GRPO training example using reward-kit reward functions.
    """
    if not HAS_TRL:
        print("TRL or related packages not installed. Install with: pip install 'reward-kit[trl]'")
        return

    print("\n=== Running GRPO Training Example with reward-kit ===\n")

    # 1. Reward functions (format_reward and math_accuracy_reward) are defined above.
    # We will create TRL adapters for them later if needed.

    # 2. Prepare dataset
    try:
        print("Loading dataset...")
        dataset_id = "AI-MO/NuminaMath-TIR"

        # Load a smaller subset to make training faster
        train_dataset = load_dataset(dataset_id, split="train[:10]")
        print(f"Dataset loaded: {len(train_dataset)} samples")

        # Simplify dataset format to avoid chat template issues
        def make_conversation(example):
            # Create a text-only format with clearer instruction about formatting
            prompt = (
                "Question: "
                + example["problem"]
                + "\n\nIMPORTANT: Your response MUST follow this format exactly:\n1. First, put your reasoning process inside <think></think> tags\n2. Then, put your final answer inside <answer></answer> tags\n\nExample format:\n<think>Your step-by-step reasoning goes here...</think>\n<answer>Your final answer goes here</answer>"
            )
            return {
                "prompt": prompt,  # TRL expects a 'prompt' field
                "solution": example["solution"],
            }

        train_dataset = train_dataset.map(make_conversation)

        # Remove unnecessary columns
        train_dataset = train_dataset.remove_columns(["problem", "messages"])
        print("Dataset prepared for GRPO")
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        return

    # 3. Load a small model (tiny model for testing)
    try:
        print("\nLoading model...")
        # Use a better model for more meaningful training
        model_id = "Qwen/Qwen3-4B"  # Using Qwen3-4B as requested
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            device_map="auto",  # Let it decide based on available resources
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Model loaded successfully")

        # Set a chat template for the tokenizer
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

        # Configure LoRA for efficient fine-tuning
        print("Configuring LoRA...")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,  # Rank for LoRA
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
            ],  # For Qwen3 model
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # 4. Configure GRPO training
    print("\nConfiguring GRPO training...")
    training_args = GRPOConfig(
        output_dir="./trl-output",
        learning_rate=1e-5,
        remove_unused_columns=False,  # Keep solution column for reward function
        gradient_accumulation_steps=2,  # Accumulate gradients to save memory
        num_train_epochs=1,  # Just one epoch for faster results
        max_steps=10,  # Only run 10 steps to verify training works
        # More realistic parameters for better training
        max_completion_length=64,  # Longer completions
        num_generations=4,  # More candidates for better exploration
        max_prompt_length=128,  # Longer context
        # Enable logging for better monitoring
        report_to=None,  # Disable wandb/tensorboard
        logging_steps=1,  # Log every step
        push_to_hub=False,
        save_strategy="epoch",  # Save at the end of each epoch
        # Performance optimizations
        bf16=torch.cuda.is_available(),  # Use bf16 if GPU available
        fp16=not torch.cuda.is_available() and torch.cuda.is_available(),  # Use fp16 as fallback
        optim="adamw_torch",  # Standard optimizer
    )

    # 5. Create TRL adapter for the format_reward function (defined at the top)
    print("Creating TRL adapter for format_reward...")
    adapted_format_reward = create_trl_adapter(
        reward_fn=format_reward,  # The @reward_function decorated one
        dataset_to_reward_kwargs_map={},  # No dynamic kwargs from dataset
        static_reward_kwargs={"think_tag": "<think>", "answer_tag": "<answer>"},
        # Prompts are strings, so default user_message_fn is fine.
    )

    # (Optional) Create TRL adapter for math_accuracy_reward if it were to be used
    print("Creating TRL adapter for math_accuracy_reward...")
    adapted_math_accuracy_reward = create_trl_adapter(
        reward_fn=math_accuracy_reward,  # The @reward_function decorated one
        dataset_to_reward_kwargs_map={"solution": "solution"},  # Map dataset 'solution' to 'solution' param
        static_reward_kwargs={},
        # Prompts are strings, so default user_message_fn is fine.
    )

    # Combine the adapted rewards
    def combine_rewards(
        reward_adapter_configs: List[Dict[str, Any]],  # Each dict: {'adapter': callable, 'weight': float}
    ) -> Callable[[List[Any], List[str]], List[float]]:  # Corrected return type hint

        total_weight = sum(c["weight"] for c in reward_adapter_configs)
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Sum of weights is {total_weight}, normalizing to 1.0.")
            for config in reward_adapter_configs:
                config["weight"] /= total_weight

        def combined_adapter_pipeline(prompts: List[Any], completions: List[str], **kwargs) -> List[float]:
            batch_size = len(prompts)
            final_scores = [0.0] * batch_size

            for config in reward_adapter_configs:
                adapter_fn = config["adapter"]
                weight = config["weight"]

                individual_scores = adapter_fn(prompts=prompts, completions=completions, **kwargs)

                for i in range(batch_size):
                    final_scores[i] += individual_scores[i] * weight
            return final_scores

        return combined_adapter_pipeline

    print("Creating combined TRL reward function (format + math_accuracy)...")
    combined_reward_for_trainer = combine_rewards(
        [
            {"adapter": adapted_format_reward, "weight": 0.3},
            {"adapter": adapted_math_accuracy_reward, "weight": 0.7},
        ]
    )

    # The custom 'format_reward_fn' has been removed.
    # We will use 'combined_reward_for_trainer'.

    # Function to test the model's outputs with the same prompt
    def test_model_outputs(model, tokenizer, test_prompt):
        # Generate before function
        print(f"\nTesting model with prompt: {test_prompt}")
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        # Generate with the current model
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return response

    # Test prompt for checking model progress
    test_prompt = "Question: What is 2+3? \n\nIMPORTANT: Your response MUST follow this format exactly:\n1. First, put your reasoning process inside <thinking></thinking> tags\n2. Then, put your final answer inside <answer></answer> tags\n\nExample format:\n<thinking>Your step-by-step reasoning goes here...</thinking>\n<answer>Your final answer goes here</answer>"

    # 6. Create and run trainer
    try:
        # Get initial output before training
        print("\nTesting model BEFORE training...")
        pre_training_output = test_model_outputs(model, tokenizer, test_prompt)
        print(f"Pre-training output: {pre_training_output}")

        print("\nInitializing GRPO trainer...")
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[combined_reward_for_trainer],  # Use the combined reward function
            args=training_args,
            train_dataset=train_dataset,
        )

        print("\nStarting training...")
        # Track training process
        trainer.args.num_train_epochs = 3  # 3 epochs
        trainer.train()

        print("\nTraining completed successfully!")

        # Test model after training
        print("\nTesting model AFTER training...")
        post_training_output = test_model_outputs(model, tokenizer, test_prompt)
        print(f"Post-training output: {post_training_output}")

        # Compare results
        print("\nComparing results:")
        print("BEFORE:\n", pre_training_output[:300])
        print("\nAFTER:\n", post_training_output[:300])

        # Check if format improved
        import re

        pre_has_think = bool(re.search(r"<thinking>(.*?)</thinking>", pre_training_output, re.DOTALL))
        pre_has_answer = bool(re.search(r"<answer>(.*?)</answer>", pre_training_output, re.DOTALL))
        post_has_think = bool(re.search(r"<thinking>(.*?)</thinking>", post_training_output, re.DOTALL))
        post_has_answer = bool(re.search(r"<answer>(.*?)</answer>", post_training_output, re.DOTALL))

        print(
            f"\nFormat compliance before training: {'<thinking>' if pre_has_think else 'No <thinking>'}, "
            + f"{'<answer>' if pre_has_answer else 'No <answer>'}"
        )
        print(
            f"Format compliance after training: {'<thinking>' if post_has_think else 'No <thinking>'}, "
            + f"{'<answer>' if post_has_answer else 'No <answer>'}"
        )

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback

        traceback.print_exc()

    print("\n=== End of GRPO example ===")


if __name__ == "__main__":
    run_grpo_training_example()
