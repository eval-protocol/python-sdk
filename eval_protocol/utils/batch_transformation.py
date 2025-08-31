"""
Utilities for transforming N-variant generation results into batch evaluation format.
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def transform_n_variant_jsonl_to_batch_format(
    input_file_path: str,
    output_file_path: Optional[str] = None,
    request_id_field: str = "request_id",
    response_id_field: str = "response_id",
    messages_field: str = "full_conversation_history",
    fallback_messages_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Transform N-variant generation JSONL output into batch evaluation format.

    This function groups variants by request_id and creates rollouts_messages
    containing all variant conversations for each original request.

    Args:
        input_file_path: Path to the N-variant generation JSONL file
        output_file_path: Optional path to write the transformed data (if None, returns data only)
        request_id_field: Field name containing the original request ID (default: "request_id")
        response_id_field: Field name containing the variant response ID (default: "response_id")
        messages_field: Primary field containing conversation messages (default: "full_conversation_history")
        fallback_messages_fields: Fallback fields to construct messages if primary field is missing

    Returns:
        List of batch evaluation entries, each containing rollouts_messages and other metadata

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required fields are missing or data format is invalid
    """
    if fallback_messages_fields is None:
        fallback_messages_fields = ["user_query", "system_prompt", "assistant_response"]

    # Group variants by request_id
    grouped_variants = defaultdict(list)

    try:
        with open(input_file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    # Skip lines with errors
                    if "error" in data:
                        logger.warning(f"Skipping line {line_num} due to error: {data.get('error')}")
                        continue

                    # Validate required fields
                    if request_id_field not in data:
                        raise ValueError(f"Line {line_num}: Missing required field '{request_id_field}'")

                    if response_id_field not in data:
                        raise ValueError(f"Line {line_num}: Missing required field '{response_id_field}'")

                    request_id = data[request_id_field]
                    response_id = data[response_id_field]

                    # Extract messages
                    messages = _extract_messages_from_data(data, messages_field, fallback_messages_fields, line_num)

                    # Store variant data
                    variant_data = {
                        "response_id": response_id,
                        "messages": messages,
                        "original_data": data,  # Keep original data for metadata extraction
                    }

                    grouped_variants[request_id].append(variant_data)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    # Transform grouped variants into batch format
    batch_entries = []

    for request_id, variants in grouped_variants.items():
        # Sort variants by response_id to ensure consistent ordering
        variants.sort(key=lambda x: x["response_id"])

        # Extract rollouts_messages (list of message lists)
        rollouts_messages = [variant["messages"] for variant in variants]

        # Extract common metadata from the first variant (assuming it's consistent across variants)
        first_variant = variants[0]["original_data"]

        # Create batch entry
        batch_entry = {
            "request_id": request_id,
            "rollouts_messages": rollouts_messages,
            "num_variants": len(variants),
            "response_ids": [variant["response_id"] for variant in variants],
        }

        # Add common fields as kwargs (excluding variant-specific fields)
        excluded_fields = {
            "id",
            request_id_field,
            response_id_field,
            messages_field,
            "full_conversation_history",
            "assistant_response",
            "evaluation_score",
            "evaluation_reason",
            "evaluation_metrics",
            "executed_tool_calls",
            "discovered_tools",
            "final_mcp_state_captured",
        }

        for key, value in first_variant.items():
            if key not in excluded_fields:
                batch_entry[key] = value

        batch_entries.append(batch_entry)

    # Write output file if specified
    if output_file_path:
        with open(output_file_path, "w", encoding="utf-8") as f:
            for entry in batch_entries:
                f.write(json.dumps(entry) + "\n")
        logger.info(f"Transformed {len(batch_entries)} batch entries written to {output_file_path}")

    return batch_entries


def _extract_messages_from_data(
    data: Dict[str, Any], primary_field: str, fallback_fields: List[str], line_num: int
) -> List[Dict[str, Any]]:
    """
    Extract conversation messages from variant data.

    Args:
        data: Variant data dictionary
        primary_field: Primary field containing messages
        fallback_fields: Fallback fields to construct messages
        line_num: Line number for error reporting

    Returns:
        List of message dictionaries
    """
    # Try primary field first
    if primary_field in data and data[primary_field]:
        messages = data[primary_field]
        if isinstance(messages, list):
            return messages
        else:
            logger.warning(f"Line {line_num}: {primary_field} is not a list, trying fallback")

    # Try to construct messages from fallback fields
    messages = []

    # Add system message if available
    if "system_prompt" in data and data["system_prompt"]:
        messages.append({"role": "system", "content": data["system_prompt"]})

    # Add user message if available
    if "user_query" in data and data["user_query"]:
        messages.append({"role": "user", "content": data["user_query"]})

    # Add assistant message if available
    if "assistant_response" in data and data["assistant_response"]:
        messages.append({"role": "assistant", "content": data["assistant_response"]})

    if not messages:
        raise ValueError(f"Line {line_num}: Could not extract messages from any available fields")

    return messages


def create_batch_evaluation_dataset(n_variant_jsonl_path: str, output_jsonl_path: str, **transform_kwargs) -> str:
    """
    Convenience function to create a batch evaluation dataset from N-variant generation output.

    Args:
        n_variant_jsonl_path: Path to N-variant generation JSONL file
        output_jsonl_path: Path for the batch evaluation JSONL file
        **transform_kwargs: Additional arguments for transform_n_variant_jsonl_to_batch_format

    Returns:
        Path to the created batch evaluation dataset
    """
    transform_n_variant_jsonl_to_batch_format(
        input_file_path=n_variant_jsonl_path,
        output_file_path=output_jsonl_path,
        **transform_kwargs,
    )
    return output_jsonl_path
