"""
Utilities for running batch evaluation on transformed N-variant data.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from ..models import EvaluateResult
from ..utils.module_loader import load_function as load_reward_function

logger = logging.getLogger(__name__)


def run_batch_evaluation(
    batch_jsonl_path: str,
    reward_function_path: str,
    output_path: str,
    reward_function_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Run batch evaluation on transformed N-variant data.

    Args:
        batch_jsonl_path: Path to the batch evaluation JSONL file
        reward_function_path: Path to the batch reward function (e.g., "module.function")
        output_path: Path to write the batch evaluation results
        reward_function_kwargs: Additional kwargs for the reward function

    Returns:
        List of batch evaluation results
    """
    if reward_function_kwargs is None:
        reward_function_kwargs = {}

    # Load the batch reward function
    reward_function = load_reward_function(reward_function_path)

    # Verify it's a batch mode function
    if not hasattr(reward_function, "_reward_function_mode"):
        logger.warning(f"Reward function {reward_function_path} doesn't have mode metadata. Assuming batch mode.")
    elif getattr(reward_function, "_reward_function_mode") != "batch":
        raise ValueError(
            f"Reward function {reward_function_path} is not configured for batch mode. Expected mode='batch'."
        )

    results = []

    try:
        with open(batch_jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    # Extract required fields
                    request_id = data.get("request_id")
                    rollouts_messages = data.get("rollouts_messages")

                    if not request_id:
                        logger.error(f"Line {line_num}: Missing request_id")
                        continue

                    if not rollouts_messages or not isinstance(rollouts_messages, list):
                        logger.error(f"Line {line_num}: Missing or invalid rollouts_messages")
                        continue

                    # Prepare kwargs for the batch reward function
                    batch_kwargs = dict(reward_function_kwargs)

                    # Add other fields from the data as kwargs (excluding the main inputs)
                    excluded_fields = {
                        "request_id",
                        "rollouts_messages",
                        "num_variants",
                        "response_ids",
                    }
                    for key, value in data.items():
                        if key not in excluded_fields:
                            batch_kwargs[key] = value

                    # Call the batch reward function
                    try:
                        batch_results = reward_function(rollouts_messages=rollouts_messages, **batch_kwargs)

                        # Validate results
                        if not isinstance(batch_results, list):
                            raise ValueError(f"Batch reward function must return a list, got {type(batch_results)}")

                        if len(batch_results) != len(rollouts_messages):
                            raise ValueError(
                                f"Batch reward function returned {len(batch_results)} results "
                                f"but expected {len(rollouts_messages)} (one per rollout)"
                            )

                        # Create result entries
                        response_ids = data.get("response_ids", list(range(len(rollouts_messages))))

                        for i, (response_id, eval_result) in enumerate(zip(response_ids, batch_results)):
                            if not isinstance(eval_result, EvaluateResult):
                                logger.error(f"Result {i} is not an EvaluateResult: {type(eval_result)}")
                                continue

                            result_entry = {
                                "request_id": request_id,
                                "response_id": response_id,
                                "rollout_index": i,
                                "evaluation_score": eval_result.score,
                                "evaluation_reason": eval_result.reason,
                                "is_score_valid": eval_result.is_score_valid,
                                "evaluation_metrics": (
                                    {k: v.model_dump() for k, v in eval_result.metrics.items()}
                                    if eval_result.metrics
                                    else {}
                                ),
                                # Include original metadata
                                **{k: v for k, v in data.items() if k not in excluded_fields},
                            }

                            results.append(result_entry)

                    except Exception as e:
                        logger.error(f"Error calling batch reward function for request {request_id}: {e}")
                        # Create error entries for each expected result
                        response_ids = data.get("response_ids", list(range(len(rollouts_messages))))
                        for i, response_id in enumerate(response_ids):
                            error_entry = {
                                "request_id": request_id,
                                "response_id": response_id,
                                "rollout_index": i,
                                "error": f"Batch evaluation failed: {str(e)}",
                                "evaluation_score": 0.0,
                                "is_score_valid": False,
                            }
                            results.append(error_entry)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        raise FileNotFoundError(f"Batch JSONL file not found: {batch_jsonl_path}")

    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Batch evaluation completed. {len(results)} results written to {output_path}")
    return results


def create_sample_batch_reward_function():
    """
    Create a sample batch reward function for testing.
    This is a simple function that compares all variants and scores them relative to each other.
    """
    from ..models import EvaluateResult, Message
    from ..typed_interface import reward_function

    @reward_function(mode="batch")
    def sample_batch_reward(
        rollouts_messages: List[List[Message]],
        ground_truth_for_eval: Optional[str] = None,
        **kwargs: Any,
    ) -> List[EvaluateResult]:
        """
        Sample batch reward function that scores variants relative to each other.

        This function demonstrates how to process multiple rollouts (variants) together
        and return comparative scores.
        """
        from ..models import MetricResult

        results = []

        # Extract the assistant responses from each rollout
        assistant_responses = []
        for rollout in rollouts_messages:
            assistant_msg = None
            for msg in reversed(rollout):  # Find the last assistant message
                if msg.role == "assistant":
                    assistant_msg = msg.content
                    break
            assistant_responses.append(assistant_msg or "")

        # Simple scoring: longer responses get higher scores (just for demonstration)
        response_lengths = [len(response) for response in assistant_responses]
        max_length = max(response_lengths) if response_lengths else 1

        for i, (response, length) in enumerate(zip(assistant_responses, response_lengths)):
            # Normalize score based on length (0.1 to 1.0)
            base_score = 0.1 + 0.9 * (length / max_length)

            # Add some variation based on position (earlier variants get slight bonus)
            position_bonus = 0.1 * (1 - i / len(assistant_responses))
            final_score = min(1.0, base_score + position_bonus)

            result = EvaluateResult(
                score=final_score,
                reason=f"Variant {i}: Length={length}, Base={base_score:.2f}, Position bonus={position_bonus:.2f}",
                is_score_valid=True,
                metrics={
                    "response_length": MetricResult(
                        score=min(1.0, length / 100.0),  # Normalize length score
                        reason=f"Response contains {length} characters",
                        is_score_valid=True,
                    )
                },
            )
            results.append(result)

        return results

    return sample_batch_reward
