"""
Example demonstrating how to use eval-protocol reward functions with TRL's PPO trainer.

This example shows how to:
1. Define a simple reward function in eval-protocol
2. Convert it to TRL-compatible format
3. Use it with the PPO trainer
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure eval-protocol is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from eval_protocol.integrations.trl import (  # Import the new generic TRL adapter
    create_trl_adapter,
)
from eval_protocol.models import (  # RewardOutput, MetricRewardOutput are likely legacy
    EvaluateResult,
    MetricResult,
)

# Import eval-protocol components
from eval_protocol.reward_function import (  # RewardFunction class no longer needed for this example
    reward_function,
)

# Try to import TRL components
try:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    from trl.core import respond_to_batch

    HAS_TRL = True
except ImportError:
    print("TRL or related packages not installed. Install with: pip install 'eval-protocol[trl]'")
    HAS_TRL = False


# Define a simple reward function compatible with eval-protocol
@reward_function
def helpfulness_reward(
    messages: List[Dict[str, Any]],
    ground_truth: Optional[List[Dict[str, Any]]] = None,  # Changed from original_messages
    **kwargs,
) -> EvaluateResult:
    """
    Reward function that evaluates helpfulness based on response length and keywords.

    This is a simplified example - a real helpfulness metric would be more sophisticated.

    Args:
        messages: List of conversation messages
        ground_truth: Optional ground truth context (not used by this specific function).

    Returns:
        EvaluateResult with score based on helpfulness
    """
    # Get the assistant's message
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            is_score_valid=False,
            metrics={"helpfulness": MetricResult(score=0.0, is_score_valid=False, reason="No messages provided")},
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
            is_score_valid=False,
            metrics={"helpfulness": MetricResult(score=0.0, is_score_valid=False, reason="No assistant response")},
        )

    text = content if content is not None else ""

    # Calculate score based on length (simple heuristic)
    word_count = len(text.split())

    # Normalize length score between 0-1 with an ideal range
    if word_count < 10:
        length_score = 0.2  # Too short
    elif word_count < 50:
        length_score = 0.5 + (word_count - 10) * 0.01  # Linear increase
    elif word_count <= 200:
        length_score = 1.0  # Ideal length
    else:
        length_score = 1.0 - (word_count - 200) * 0.002  # Gradually decrease for verbosity
        length_score = max(0.3, length_score)  # Don't go below 0.3

    # Check for helpful phrases (simple keyword heuristic)
    helpful_phrases = [
        "here's how",
        "you can",
        "for example",
        "explanation",
        "step",
        "process",
        "method",
        "approach",
        "solution",
        "answer",
        "result",
    ]

    helpful_count = sum(1 for phrase in helpful_phrases if phrase.lower() in text.lower())
    helpfulness_score = min(1.0, helpful_count / 5)  # Normalize to 0-1

    # Combine scores (70% length, 30% helpful phrases)
    combined_score = 0.7 * length_score + 0.3 * helpfulness_score

    # Prepare reason text
    reason = (
        f"Length score: {length_score:.2f} ({word_count} words), "
        f"Helpful phrases: {helpfulness_score:.2f} ({helpful_count} phrases)"
    )

    return EvaluateResult(
        score=combined_score,
        reason=reason,
        is_score_valid=True,
        metrics={
            "length": MetricResult(
                score=length_score,
                is_score_valid=length_score > 0.7,
                reason=f"Response length: {word_count} words",
            ),
            "helpful_phrases": MetricResult(
                score=helpfulness_score,
                is_score_valid=helpfulness_score > 0.5,
                reason=f"Helpful phrases: {helpful_count} found",
            ),
        },
    )


def prepare_dataset_for_ppo(dataset_name, split="train", max_samples=None):
    """
    Prepare a HuggingFace dataset for use with PPO.

    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to use
        max_samples: Maximum samples to include

    Returns:
        Dataset in PPO-compatible format
    """
    if not HAS_TRL:
        print("TRL or related packages not installed. Install with: pip install 'eval-protocol[trl]'")
        return None

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    # Limit samples if specified
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    # For PPO we just need query/text pairs - default to assume it's a summarization dataset
    def prepare_sample(example):
        return {
            "query": example.get("article", example.get("document", "")),
            "input_ids": None,  # This will be filled by the PPO Trainer
        }

    formatted_dataset = dataset.map(prepare_sample)

    # Keep only the needed columns
    if "query" in formatted_dataset.column_names:
        formatted_dataset = formatted_dataset.remove_columns(
            [c for c in formatted_dataset.column_names if c != "query"]
        )

    return formatted_dataset


def train_with_ppo_example():
    """
    Example of training with PPO using an eval-protocol reward function.
    """
    if not HAS_TRL:
        print("TRL or related packages not installed. Install with: pip install 'eval-protocol[trl]'")
        return

    print("Setting up PPO training with an eval-protocol reward function...")

    # 1. Create reward function adapter using the new create_trl_adapter
    # helpfulness_reward is defined above and decorated with @reward_function
    # It doesn't require special dataset column mapping or static args beyond messages.
    # The default user_message_fn (str) and assistant_message_fn (str) are suitable if
    # the 'prompts' passed to the adapter are strings (which 'query' from dataset is).
    print("Creating TRL adapter for helpfulness_reward...")
    adapted_helpfulness_reward_fn = create_trl_adapter(
        reward_fn=helpfulness_reward,
        dataset_to_reward_kwargs_map={},  # No dynamic kwargs from dataset needed
        static_reward_kwargs={},  # No static kwargs needed
    )

    # 2. Prepare dataset (use a simple summarization dataset)
    try:
        print("Preparing dataset...")
        dataset = prepare_dataset_for_ppo(
            dataset_name="cnn_dailymail",
            split="test[:1%]",  # Use a tiny subset for demonstration
            max_samples=5,
        )
        print(f"Dataset prepared: {len(dataset)} samples")
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        return

    # 3. Set up model (would be used in real training)
    if False:  # Skip for example purposes
        print("Setting up model...")
        # Load model and tokenizer
        model_name = "gpt2"  # Use a small model for example
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Configure PPO
        ppo_config = PPOConfig(
            learning_rate=1e-5,
            batch_size=1,
            mini_batch_size=1,
            gradient_accumulation_steps=1,
            optimize_cuda_cache=True,
            early_stopping=False,
            target_kl=0.1,
            ppo_epochs=4,
            seed=42,
        )

        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(
            model=model,
            args=ppo_config,  # Changed 'config' to 'args'
            tokenizer=tokenizer,
            train_dataset=dataset,  # Changed 'dataset' to 'train_dataset'
            # data_collator=None, # Optional: add if specific collator is needed
            # num_shared_layers=None, # Optional
        )

        # Generate responses and compute rewards
        for _ in range(1):  # In real training, you'd do more iterations
            # Generate model responses
            query_tensors = ppo_trainer.prepare_sample(dataset["query"], truncation=True, max_length=256)
            response_tensors = respond_to_batch(
                ppo_trainer.model,
                query_tensors,
                ppo_trainer.tokenizer,
                max_new_tokens=64,
            )

            # Decode responses and format for reward function
            response_strings = [ppo_trainer.tokenizer.decode(r.squeeze()) for r in response_tensors]
            # The 'prompts' for the reward function are the original query strings from the dataset
            # Assuming `dataset["query"]` is accessible and corresponds to the batch.
            # For simplicity in this loop, let's assume `batch_queries` is a list of strings.
            # In a real loop, you'd get the current batch of queries corresponding to query_tensors.
            # For this example, let's use a placeholder if dataset["query"] isn't directly batch-aligned here.
            # A more robust way would be to iterate through dataset or use ppo_trainer.tokenizer.batch_decode on query_tensors if they are not padded.

            # Let's assume `current_batch_queries` is a list of strings for the current batch.
            # This part needs careful handling in a real PPO loop to align prompts with responses.
            # For now, we'll use dataset["query"] which is the full list; this is not quite right for a batch.
            # However, the key is to show the signature.

            # Correct call to the new adapter:
            # prompts = list of query strings for the batch
            # completions = list of response strings for the batch
            # Example: current_batch_queries = [dataset["query"][i] for i in current_batch_indices]
            # rewards = adapted_helpfulness_reward_fn(prompts=current_batch_queries, completions=response_strings)

            # Simplified for this commented out block, showing intent:
            # Assuming `dataset["query"]` could be sliced or batched appropriately.
            # This is illustrative as the actual batching depends on PPOTrainer's internals or how data is fed.
            # The crucial part is `prompts` should be List[str] and `completions` List[str].

            # If query_tensors are just tokenized versions of dataset["query"], we can decode them too,
            # but PPO usually keeps queries as input_ids.
            # For reward calculation, we need the string prompts.
            # Let's assume we have `batch_query_strings` available.

            # Placeholder for actual batch query strings
            batch_query_strings = dataset["query"][: len(response_strings)]  # Example: align with number of responses

            rewards = adapted_helpfulness_reward_fn(batch_query_strings, response_strings)

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            print(f"Rewards: {rewards}")
            print(f"Stats: {stats}")

    print("\nExample completed successfully. In a real scenario, the PPO training would now run.")
    print("This example shows how an eval-protocol reward function can be adapted for TRL's PPO trainer.")

    # Show how the reward function would be called
    print("\nReward function test on sample data:")
    sample_messages = [
        {"role": "user", "content": "Explain how to make cookies"},
        {
            "role": "assistant",
            "content": "Here's how to make chocolate chip cookies: First, you'll need flour, sugar, butter, eggs, and chocolate chips. The process has several steps. Start by creaming together the butter and sugar, then add eggs. Next, combine flour with baking soda and salt, then mix into the wet ingredients. Finally, fold in chocolate chips and bake at 375°F for 10-12 minutes. For the best results, let them cool for 5 minutes before enjoying.",
        },
    ]

    reward_result_obj = helpfulness_reward(sample_messages)  # helpfulness_reward returns an EvaluateResult object
    print(f"Helpfulness reward score: {reward_result_obj.score} - {reward_result_obj.get('reason')}")

    # Show how the adapter formats for TRL
    # The new adapter expects prompts=List[str], completions=List[str]
    sample_user_prompt = sample_messages[0]["content"]
    sample_assistant_completion = sample_messages[1]["content"]

    adapted_reward_score_list = adapted_helpfulness_reward_fn([sample_user_prompt], [sample_assistant_completion])
    print(f"TRL adapter converted reward (score list): {adapted_reward_score_list}")


if __name__ == "__main__":
    train_with_ppo_example()
