"""
CLI command for previewing an evaluator.
"""

import json
import sys  # For sys.exit
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import requests  # For making HTTP requests

from eval_protocol.evaluation import preview_evaluation

# Assuming EvaluationRequest is defined in generic_server.
# For loose coupling, it might be better in models.py or a shared types module.
from eval_protocol.generic_server import EvaluationRequest
from eval_protocol.models import EvaluateResult, Message

# Assuming these helper functions exist or will be created in .common
# If not, their logic for loading samples would need to be integrated here or called differently.
from .common import (
    check_environment,
    load_samples_from_file,
    load_samples_from_huggingface,
)


def preview_command(args):
    """Preview an evaluator with sample data"""

    # Check environment variables
    if not check_environment():
        return 1

    # Validate --remote-url and --metrics-folders usage
    if args.remote_url and args.metrics_folders:
        print("Info: --metrics-folders are ignored when --remote-url is specified.")

    if not args.remote_url and not args.metrics_folders:
        print("Error: Either --remote-url or --metrics-folders must be specified.")
        return 1

    # Ensure either samples or huggingface_dataset is provided (still needed for remote_url)
    if not args.samples and not args.huggingface_dataset:
        print("Error: Either sample file (--samples) or HuggingFace dataset (--huggingface-dataset) is required.")
        return 1

    # If using samples file, verify it exists
    if args.samples and not Path(args.samples).exists():
        print(f"Error: Sample file '{args.samples}' not found")
        return 1

    # Process HuggingFace key mapping if provided
    huggingface_message_key_map = None
    if args.huggingface_key_map:
        try:
            huggingface_message_key_map = json.loads(args.huggingface_key_map)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for --huggingface-key-map")
            return 1

    if args.remote_url:
        # Handle previewing against a remote URL
        print(f"Previewing against remote URL: {args.remote_url}")

        # Ensure the remote URL is a valid base URL
        if not (args.remote_url.startswith("http://") or args.remote_url.startswith("https://")):
            print(f"Error: Invalid --remote-url '{args.remote_url}'. Must start with http:// or https://")
            return 1

        evaluate_endpoint = f"{args.remote_url.rstrip('/')}/evaluate"

        samples_iterator: Union[List[Any], Iterator[Dict[str, Any]]] = []
        try:
            if args.samples:
                # Assuming load_samples_from_file yields dicts with "messages" and optional "ground_truth"
                samples_iterator = load_samples_from_file(args.samples, args.max_samples)
            elif args.huggingface_dataset:
                # Assuming load_samples_from_huggingface yields dicts with "messages" and optional "ground_truth"
                samples_iterator = load_samples_from_huggingface(
                    dataset_name=args.huggingface_dataset,
                    split=args.huggingface_split,
                    prompt_key=args.huggingface_prompt_key,
                    response_key=args.huggingface_response_key,
                    key_map=huggingface_message_key_map,
                    max_samples=args.max_samples,
                )
        except Exception as e:
            print(f"Error loading samples: {e}")
            return 1

        results_count = 0
        for i, sample_data in enumerate(samples_iterator):
            # The load_samples_from_* helpers should ideally cap at max_samples,
            # but we double-check here.
            if i >= args.max_samples:
                break
            results_count += 1

            messages_payload = sample_data.get("messages", [])
            ground_truth_payload = sample_data.get("ground_truth")
            # Allow passing other sample fields as kwargs to the reward function
            sample_kwargs = {k: v for k, v in sample_data.items() if k not in ["messages", "ground_truth"]}

            processed_messages = []
            for msg_item in messages_payload:
                if isinstance(msg_item, Message):  # If helpers return Message objects
                    processed_messages.append(msg_item.model_dump(exclude_none=True))
                elif isinstance(msg_item, dict):  # If helpers return dicts
                    processed_messages.append(msg_item)
                else:
                    print(
                        f"Warning: Sample {i + 1} has unexpected message item type: {type(msg_item)}. Skipping this message item."
                    )

            try:
                request_obj = EvaluationRequest(
                    messages=processed_messages,
                    ground_truth=ground_truth_payload,
                    kwargs=sample_kwargs,
                )
            except Exception as e:  # Pydantic validation for EvaluationRequest
                print(f"\n--- Sample {i + 1} ---")
                print(f"  Error creating request payload for sample: {e}")
                print(f"  Sample data: {sample_data}")
                print("--- End Sample ---")
                continue  # Skip to next sample

            print(f"\n--- Sample {i + 1} ---")

            try:
                response = requests.post(
                    evaluate_endpoint,
                    json=request_obj.model_dump(),  # Use model_dump() for Pydantic v2, or .dict() for v1
                    timeout=30,
                )
                response.raise_for_status()

                result_json = response.json()
                evaluate_result = EvaluateResult(**result_json)

                print(f"  Score: {evaluate_result.score}")
                print(f"  Reason: {evaluate_result.reason if evaluate_result.reason else 'N/A'}")
                print(f"  Is Valid: {evaluate_result.is_score_valid}")
                if evaluate_result.metrics:
                    for k, v_metric in evaluate_result.metrics.items():
                        print(
                            f"  Metric '{k}': Score={v_metric.score}, Valid={v_metric.is_score_valid}, Reason={v_metric.reason}"
                        )

            except requests.exceptions.RequestException as e:
                print(f"  Error calling remote URL '{evaluate_endpoint}': {e}")
            except json.JSONDecodeError:
                print(
                    f"  Error: Invalid JSON response from remote URL. Status: {response.status_code}. Response text: {response.text[:200]}..."
                )
            except Exception as e:
                print(f"  Error processing response from remote URL: {e}")
            print("--- End Sample ---")

        if results_count == 0:
            print("No samples were processed. Check sample source or loading functions.")
        return 0

    else:
        # Original behavior: preview using local metrics_folders
        # This path is taken if args.remote_url is None (or empty string)
        # We already checked above that if not remote_url, then metrics_folders must be present.

        try:
            preview_result = preview_evaluation(
                metric_folders=args.metrics_folders,
                sample_file=args.samples if args.samples else None,
                max_samples=args.max_samples,
                huggingface_dataset=args.huggingface_dataset,
                huggingface_split=args.huggingface_split,
                huggingface_prompt_key=args.huggingface_prompt_key,
                huggingface_response_key=args.huggingface_response_key,
                huggingface_message_key_map=huggingface_message_key_map,
            )

            preview_result.display()
            return 0
        except Exception as e:
            print(f"Error previewing evaluator (local mode): {str(e)}")
            return 1
