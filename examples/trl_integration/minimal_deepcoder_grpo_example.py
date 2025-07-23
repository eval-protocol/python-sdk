"""
Minimal example demonstrating the DeepCoder-style reward function
with TRL's GRPO trainer.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG to see more logs
logger = logging.getLogger(__name__)

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import torch
    from datasets import Dataset  # To convert our list of dicts to HuggingFace Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    HAS_TRL_AND_TRANSFORMERS = True
except ImportError as e:
    print(
        f"TRL/Transformers/PEFT/Datasets not installed. Install with: pip install 'reward-kit[trl]' transformers bitsandbytes. Error: {e}"
    )
    HAS_TRL_AND_TRANSFORMERS = False

# from eval_protocol.models import Message # No longer directly needed here
from eval_protocol.integrations.trl import create_trl_adapter  # Import the new adapter

# Import reward-kit components
# from eval_protocol.reward_function import RewardFunction # No longer strictly needed here
from eval_protocol.rewards import deepcoder_code_reward
from eval_protocol.rewards.code_execution_utils import prepare_deepcoder_sample_for_trl
from eval_protocol.utils.dataset_helpers import load_jsonl_to_hf_dataset

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"  # Small Qwen model for example
DATASET_PATH = (
    Path(__file__).parent / "data/simulated_deepcoder_raw_sample.jsonl"
)  # This will be processed by prepare_deepcoder_sample_for_trl
LANGUAGE = "python"
ENVIRONMENT = "local"  # "e2b" if configured
TIMEOUT = 10  # seconds for code execution


def load_and_prepare_dataset(raw_data_path: Path) -> Optional[Dataset]:
    """Loads and prepares the DeepCoder-style dataset into HuggingFace Dataset format using reward-kit utilities."""

    required_cols_for_reward = [
        "test_cases",
        "target_function",
    ]  # 'prompt' is handled by default by load_jsonl_to_hf_dataset

    hf_dataset = load_jsonl_to_hf_dataset(
        dataset_path=str(raw_data_path),
        transform_fn=prepare_deepcoder_sample_for_trl,
        required_columns=required_cols_for_reward,
    )

    if hf_dataset is None:
        logger.error(
            f"Failed to load dataset from {raw_data_path} using reward-kit utilities."
        )
        return None

    if len(hf_dataset) == 0:
        logger.error(
            f"No samples loaded from {raw_data_path}. Check dataset content and transform_fn."
        )
        return None

    logger.info(
        f"Dataset loaded and prepared: {len(hf_dataset)} samples. Columns: {hf_dataset.column_names}"
    )
    return hf_dataset


# The custom deepcoder_grpo_reward_adapter function is no longer needed.
# It will be replaced by using create_trl_adapter from eval_protocol.


def generate_for_comparison(model, tokenizer, prompt_text: str, device) -> str:
    """Helper to generate a response from the model for comparison."""
    # Use a more general system prompt that encourages following user instructions.
    # The user prompt (prompt_text) will contain specific formatting instructions from data_utils.py.
    system_prompt = (
        "You are a helpful assistant that writes Python code. "
        "Follow the user's instructions carefully to produce the required code output. "
        "Be concise and generate only the requested code. Avoid any conversational fluff or explanations outside the code block."
    )
    messages_for_generation = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": prompt_text,
        },  # The user prompt contains specific instructions
    ]
    prompt_for_model = tokenizer.apply_chat_template(
        messages_for_generation, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_for_model, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": 4000,  # Increased from 250
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "top_k": 10,
        "top_p": 0.95,
        "temperature": 0.7,
    }
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        response_text = tokenizer.decode(
            outputs[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
    except Exception as e:
        logger.error(f"Error during comparison generation: {e}")
        response_text = f"Error generating: {e}"
    return response_text


def main():
    if not HAS_TRL_AND_TRANSFORMERS:
        return

    logger.info("Starting Minimal DeepCoder GRPO Example...")

    # 1. Initialize Model and Tokenizer
    logger.info(f"Loading model and tokenizer: {MODEL_NAME}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=(
                (
                    torch.bfloat16
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    else torch.float16
                )
                if torch.cuda.is_available()
                else torch.float32
            ),
            # device_map="auto" # Usually good, but can be problematic with small models / CPU
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.chat_template is None:
            # A basic chat template for Qwen2-Instruct if not set
            tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Using device: {device}")

        # Configure LoRA for efficient fine-tuning
        logger.info("Configuring LoRA...")
        # Adjust target_modules based on the model being used. For Qwen2-0.5B:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,  # Often 2*r
            lora_dropout=0.05,  # Reduced dropout
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],  # Common for Qwen2
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    except Exception as e:
        logger.error(
            f"Error loading model/tokenizer or configuring LoRA: {e}", exc_info=True
        )
        return

    # 2. Load and Prepare Dataset
    logger.info(f"Loading dataset from: {DATASET_PATH}")
    train_dataset = load_and_prepare_dataset(DATASET_PATH)
    if train_dataset is None:
        return

    # For GRPO, the dataset should be a HuggingFace Dataset object
    # The load_and_prepare_dataset function now returns this.

    # 3. Create the adapted reward function using the new TRL adapter
    logger.info("Creating TRL adapter for the reward function...")
    adapted_reward_func = create_trl_adapter(
        reward_fn=deepcoder_code_reward,
        dataset_to_reward_kwargs_map={
            "test_cases": "test_cases",  # dataset_column_name maps to reward_fn_param_name
            "target_function": "target_function",
        },
        static_reward_kwargs={
            "language": LANGUAGE,
            "environment": ENVIRONMENT,
            "timeout": TIMEOUT,
        },
        # user_message_fn and assistant_message_fn can be omitted for default behavior
    )

    # 4. Configure GRPO Training
    logger.info("Configuring GRPO training...")
    # Reduce batch size and steps for a quick test
    training_args = GRPOConfig(
        output_dir="./grpo_deepcoder_output",
        per_device_train_batch_size=2,  # Adjusted to be divisible by num_generations
        gradient_accumulation_steps=1,  # Keep small
        learning_rate=1e-5,  # GRPO often uses smaller LRs
        num_train_epochs=1,  # Minimal epochs for testing
        max_steps=5,  # Run very few steps for a quick test
        remove_unused_columns=False,  # We need 'test_cases' and 'target_function' for the reward
        logging_steps=1,
        report_to="none",  # No wandb/tensorboard for this minimal example
        max_prompt_length=4000,  # Max length of prompt
        max_completion_length=4000,  # Max length of completion
        num_generations=2,  # Number of completions to generate per prompt
        beta=0.1,  # GRPO specific: KL divergence weight
        # bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        # fp16=torch.cuda.is_available() and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        # Using fp32 for wider compatibility in this example, can enable bf16/fp16 if desired
    )

    # Select a sample prompt for before/after comparison
    sample_prompt_for_comparison = (
        train_dataset[0]["prompt"]
        if len(train_dataset) > 0
        else "Write a Python function to add two numbers."
    )

    # Generate before training
    logger.info("\n--- Generating with model BEFORE training ---")
    pre_train_response = generate_for_comparison(
        model, tokenizer, sample_prompt_for_comparison, device
    )
    logger.info(f"Prompt: {sample_prompt_for_comparison[:100]}...")
    logger.info(f"Response (before): {pre_train_response[:200]}...")

    # 5. Create and run GRPOTrainer
    try:
        logger.info("Initializing GRPOTrainer...")
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            # tokenizer=tokenizer, # Removed: GRPOTrainer likely infers tokenizer from model or args
            train_dataset=train_dataset,
            reward_funcs=[adapted_reward_func],  # Pass the new adapted reward function
            # peft_config=lora_config, # Already applied with get_peft_model
        )

        logger.info("Starting GRPO training...")
        trainer.train()
        logger.info("GRPO training completed.")

    except Exception as e:
        logger.error(
            f"Error during GRPOTrainer initialization or training: {e}", exc_info=True
        )
        return

    # Generate after training
    logger.info("\n--- Generating with model AFTER training ---")
    # If using PEFT, ensure model is in eval mode or merged for inference if needed
    # model.eval() # Good practice, though generate might handle it
    post_train_response = generate_for_comparison(
        model, tokenizer, sample_prompt_for_comparison, device
    )
    logger.info(f"Prompt: {sample_prompt_for_comparison[:100]}...")
    logger.info(f"Response (after): {post_train_response[:200]}...")

    logger.info("\nMinimal DeepCoder GRPO Example finished.")


if __name__ == "__main__":
    if HAS_TRL_AND_TRANSFORMERS:
        main()
    else:
        print(
            "TRL/Transformers/PEFT/Datasets not found. Please install them to run this example: pip install 'reward-kit[trl]' transformers bitsandbytes"
        )
