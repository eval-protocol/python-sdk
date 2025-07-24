#!/usr/bin/env python3
"""
Demonstration of N-variant generation to batch evaluation workflow.

This module provides utilities and a demo for the complete workflow:
1. Generate N variants using the evaluation pipeline
2. Transform results into batch evaluation format
3. Run batch evaluation on transformed data

Usage:
    python examples/n_variant_to_batch_demo.py
"""

import json
import tempfile
from pathlib import Path
from typing import List

from eval_protocol.models import EvaluateResult, Message, MetricResult
from eval_protocol.typed_interface import reward_function
from eval_protocol.utils.batch_evaluation import run_batch_evaluation
from eval_protocol.utils.batch_transformation import (
    transform_n_variant_jsonl_to_batch_format,
)


@reward_function(mode="batch")
def demo_batch_reward(
    rollouts_messages: List[List[Message]], ground_truth_for_eval: str = None, **kwargs
) -> List[EvaluateResult]:
    """Demo batch reward function that scores variants based on response quality."""

    results = []

    # Extract assistant responses from each rollout
    assistant_responses = []
    for rollout in rollouts_messages:
        assistant_response = ""
        for msg in rollout:
            if msg.role == "assistant":
                assistant_response = msg.content
                break
        assistant_responses.append(assistant_response)

    # Score based on response quality indicators
    for i, response in enumerate(assistant_responses):
        length = len(response)
        has_details = "because" in response.lower() or "since" in response.lower()
        has_examples = "example" in response.lower() or "for instance" in response.lower()

        # Scoring components
        length_score = 0.2 + min(0.4, length / 200.0)
        explanation_bonus = 0.2 if has_details else 0.0
        example_bonus = 0.2 if has_examples else 0.0

        final_score = min(1.0, length_score + explanation_bonus + example_bonus)

        result = EvaluateResult(
            score=final_score,
            reason=f"Variant {i}: Length={length} (+{length_score:.2f}), "
            f"Explanations={has_details} (+{explanation_bonus:.2f}), "
            f"Examples={has_examples} (+{example_bonus:.2f})",
            is_score_valid=True,
            metrics={
                "length": MetricResult(
                    score=min(1.0, length / 100.0),
                    reason=f"Response length: {length} characters",
                    is_score_valid=True,
                ),
                "explanation_quality": MetricResult(
                    score=1.0 if has_details else 0.0,
                    reason=("Contains explanation words" if has_details else "No explanation words found"),
                    is_score_valid=True,
                ),
            },
        )
        results.append(result)

    return results


def create_sample_data() -> List[dict]:
    """Create sample N-variant data for demonstration."""

    return [
        # Request 1 variants
        {
            "id": "sample_1_v0",
            "request_id": "sample_1",
            "response_id": 0,
            "user_query": "Why is the sky blue?",
            "ground_truth_for_eval": "Due to Rayleigh scattering of sunlight",
            "assistant_response": "Blue color.",
            "full_conversation_history": [
                {"role": "user", "content": "Why is the sky blue?"},
                {"role": "assistant", "content": "Blue color."},
            ],
        },
        {
            "id": "sample_1_v1",
            "request_id": "sample_1",
            "response_id": 1,
            "user_query": "Why is the sky blue?",
            "ground_truth_for_eval": "Due to Rayleigh scattering of sunlight",
            "assistant_response": "The sky is blue because of Rayleigh scattering.",
            "full_conversation_history": [
                {"role": "user", "content": "Why is the sky blue?"},
                {
                    "role": "assistant",
                    "content": "The sky is blue because of Rayleigh scattering.",
                },
            ],
        },
        {
            "id": "sample_1_v2",
            "request_id": "sample_1",
            "response_id": 2,
            "user_query": "Why is the sky blue?",
            "ground_truth_for_eval": "Due to Rayleigh scattering of sunlight",
            "assistant_response": "The sky appears blue because sunlight is scattered by molecules in the atmosphere. This is called Rayleigh scattering, and blue light is scattered more than other colors since it has a shorter wavelength. For example, this is why sunsets appear red - the blue light has been scattered away.",
            "full_conversation_history": [
                {"role": "user", "content": "Why is the sky blue?"},
                {
                    "role": "assistant",
                    "content": "The sky appears blue because sunlight is scattered by molecules in the atmosphere. This is called Rayleigh scattering, and blue light is scattered more than other colors since it has a shorter wavelength. For example, this is why sunsets appear red - the blue light has been scattered away.",
                },
            ],
        },
        # Request 2 variants
        {
            "id": "sample_2_v0",
            "request_id": "sample_2",
            "response_id": 0,
            "user_query": "How do plants make food?",
            "ground_truth_for_eval": "Through photosynthesis using sunlight, CO2, and water",
            "assistant_response": "Photosynthesis.",
            "full_conversation_history": [
                {"role": "user", "content": "How do plants make food?"},
                {"role": "assistant", "content": "Photosynthesis."},
            ],
        },
        {
            "id": "sample_2_v1",
            "request_id": "sample_2",
            "response_id": 1,
            "user_query": "How do plants make food?",
            "ground_truth_for_eval": "Through photosynthesis using sunlight, CO2, and water",
            "assistant_response": "Plants make food through photosynthesis, since they use sunlight to convert carbon dioxide and water into glucose.",
            "full_conversation_history": [
                {"role": "user", "content": "How do plants make food?"},
                {
                    "role": "assistant",
                    "content": "Plants make food through photosynthesis, since they use sunlight to convert carbon dioxide and water into glucose.",
                },
            ],
        },
    ]


def run_demo():
    """Run the complete N-variant to batch evaluation workflow demo."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # File paths
        n_variant_file = temp_path / "n_variant_results.jsonl"
        batch_input_file = temp_path / "batch_input.jsonl"
        batch_output_file = temp_path / "batch_results.jsonl"

        print("=== N-Variant to Batch Evaluation Demo ===\n")

        # Step 1: Create sample data
        print("Step 1: Creating sample N-variant data...")
        sample_data = create_sample_data()

        with open(n_variant_file, "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")

        print(f"Created {len(sample_data)} variants")

        # Step 2: Transform to batch format
        print("\nStep 2: Transforming to batch format...")
        batch_data = transform_n_variant_jsonl_to_batch_format(
            input_file_path=str(n_variant_file), output_file_path=str(batch_input_file)
        )

        print(f"Transformed {len(batch_data)} requests:")
        for entry in batch_data:
            print(f"  - Request '{entry['request_id']}': {entry['num_variants']} variants")

        # Step 3: Run batch evaluation
        print("\nStep 3: Running batch evaluation...")
        batch_results = run_batch_evaluation(
            batch_jsonl_path=str(batch_input_file),
            reward_function_path=f"{__name__}.demo_batch_reward",
            output_path=str(batch_output_file),
        )

        print(f"Generated {len(batch_results)} individual results")

        # Step 4: Show results summary
        print("\nStep 4: Results Summary")
        results_by_request = {}
        for result in batch_results:
            request_id = result["request_id"]
            if request_id not in results_by_request:
                results_by_request[request_id] = []
            results_by_request[request_id].append(result)

        for request_id, request_results in results_by_request.items():
            print(f"\nRequest '{request_id}':")
            request_results.sort(key=lambda x: x["response_id"])

            for result in request_results:
                score = result["evaluation_score"]
                print(f"  Variant {result['response_id']}: Score={score:.3f}")

            best_variant = max(request_results, key=lambda x: x["evaluation_score"])
            print(f"  🏆 Best: Variant {best_variant['response_id']} (score: {best_variant['evaluation_score']:.3f})")


if __name__ == "__main__":
    run_demo()
