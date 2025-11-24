import asyncio
import os
from typing import List

from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset
from eval_protocol.models import Message
from examples.calibration.evaluator import calibration_evaluator

# Models to test
MODELS = [
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "accounts/fireworks/models/kimi-k2-instruct",
]

# AG News label mapping
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


async def main():
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news", split="test[:100]")  # Use subset for speed

    rollouts = []
    ground_truths = []

    for item in dataset:
        text = item["text"]
        label_id = item["label"]
        label_name = LABEL_MAP[label_id]

        # Create message history (just user prompt)
        rollouts.append([Message(role="user", content=text)])
        ground_truths.append(Message(role="assistant", content=label_name))

    print(f"Loaded {len(rollouts)} samples.")

    for model in MODELS:
        print(f"\nEvaluating model: {model}")
        try:
            results = await calibration_evaluator(
                rollouts_messages=rollouts, ground_truth=ground_truths, model_id=model
            )

            # Extract metrics
            accuracies = [r.metrics["correct"].score for r in results]
            briers = [r.score for r in results]
            ece = results[0].metrics["global_ece"].score

            avg_acc = sum(accuracies) / len(accuracies)
            avg_brier = sum(briers) / len(briers)

            print(f"Results for {model}:")
            print(f"  Accuracy: {avg_acc:.4f}")
            print(f"  Avg Brier Score: {avg_brier:.4f}")
            print(f"  ECE: {ece:.4f}")

        except Exception as e:
            print(f"Failed to evaluate {model}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
