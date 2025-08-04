"""
Common utility functions for the Eval Protocol CLI.
"""

import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


def setup_logging(verbose=False, debug=False):
    """Setup logging configuration"""
    if debug:
        log_level = logging.DEBUG
        # More detailed format for debug
        format_str = "[%(asctime)s][%(name)s][%(levelname)s] - %(pathname)s:%(lineno)d - %(message)s"
    elif verbose:  # --verbose flag
        log_level = logging.INFO
        # Consistent format, similar to user's logs but with name
        format_str = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    else:  # Default (neither --verbose nor --debug)
        log_level = logging.INFO  # Changed from WARNING to INFO
        # Use the same format as verbose for default INFO level
        format_str = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"

    logging.basicConfig(level=log_level, format=format_str, datefmt="%Y-%m-%d %H:%M:%S")

    # Set higher levels for noisy libraries unless in full debug mode
    if not debug:
        noisy_loggers = ["httpx", "mcp", "urllib3", "asyncio", "hpack", "httpcore"]
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Ensure eval_protocol's own loggers respect the overall log_level,
    # overriding any specific DEBUG settings in submodules unless --debug is used.
    # If log_level is WARNING (default), eval_protocol INFO and DEBUG logs will be suppressed.
    # If log_level is INFO (--verbose), eval_protocol DEBUG logs will be suppressed.
    # If log_level is DEBUG (--debug), all eval_protocol logs show.
    logging.getLogger("eval_protocol").setLevel(log_level)


def check_environment():
    """Check if required environment variables are set for general commands."""
    if not os.environ.get("FIREWORKS_API_KEY"):
        logger.warning("FIREWORKS_API_KEY environment variable is not set.")
        logger.warning("This is required for API calls. Set this variable before running the command.")
        logger.warning("Example: FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY reward-kit [command]")
        return False
    return True


def check_agent_environment(test_mode=False):
    """Check if required environment variables are set for agent evaluation commands."""
    missing_vars = []
    if not os.environ.get("MODEL_AGENT"):
        missing_vars.append("MODEL_AGENT")

    if test_mode:
        if missing_vars:
            logger.info(f"Note: The following environment variables are not set: {', '.join(missing_vars)}")
            logger.info("Since you're running in test mode, these are not strictly required for all operations.")
        return True

    if missing_vars:
        logger.warning(f"The following environment variables are not set: {', '.join(missing_vars)}")
        logger.warning(
            "These are typically required for full agent evaluation. Set these variables for full functionality."
        )
        logger.warning("Example: MODEL_AGENT=openai/gpt-4o-mini reward-kit agent-eval [args]")
        logger.warning("Alternatively, use --test-mode for certain validation tasks without requiring all API keys.")
        return False
    return True


# --- Sample Loading Helper Functions ---


def _validate_sample_messages(messages: Any, sample_index: int, line_number: int) -> bool:
    """Helper to validate the 'messages' field of a sample."""
    if not isinstance(messages, list):
        logger.warning(f"Sample {sample_index} (line {line_number}): 'messages' field is not a list. Skipping sample.")
        return False
    if not messages:
        logger.warning(f"Sample {sample_index} (line {line_number}): 'messages' list is empty. Skipping sample.")
        return False
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            logger.warning(
                f"Sample {sample_index} (line {line_number}): message item {i} is not a dictionary. Skipping sample."
            )
            return False
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            logger.warning(
                f"Sample {sample_index} (line {line_number}): message item {i} missing 'role' or 'content' string fields. Skipping sample."
            )
            return False
    return True


def load_samples_from_file(filepath: str, max_samples: int) -> Iterator[Dict[str, Any]]:
    """
    Loads samples from a JSONL file.
    Each line should be a JSON object.
    Each sample must contain a 'messages' key with a list of message dicts (each having 'role' and 'content').
    Yields valid sample dictionaries up to max_samples.
    """
    count = 0
    line_number = 0
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line_number += 1
                if count >= max_samples:
                    logger.info(f"Reached max_samples ({max_samples}). Stopping sample loading from {filepath}.")
                    break
                line_content = line.strip()
                if not line_content:
                    continue
                try:
                    sample = json.loads(line_content)
                except json.JSONDecodeError:
                    logger.warning(f"Line {line_number}: Invalid JSON. Skipping line: {line_content[:100]}...")
                    continue
                if not isinstance(sample, dict):
                    logger.warning(f"Line {line_number}: Content is not a JSON object. Skipping line.")
                    continue
                messages = sample.get("messages")
                if messages is None:
                    logger.warning(f"Sample (line {line_number}): 'messages' field is missing. Skipping sample.")
                    continue
                if not _validate_sample_messages(messages, count + 1, line_number):
                    continue
                yield sample
                count += 1
    except FileNotFoundError:
        logger.error(f"Sample file not found: {filepath}")
    except Exception as e:
        logger.error(f"Error reading or processing sample file {filepath}: {e}")
    if count == 0:
        logger.info(f"No valid samples loaded from {filepath} after processing {line_number} lines.")


def load_samples_from_huggingface(
    dataset_name: str,
    split: str,
    prompt_key: str,
    response_key: str,
    key_map: Optional[Dict[str, str]],
    max_samples: int,
) -> Iterator[Dict[str, Any]]:
    """
    Loads samples from a HuggingFace dataset using the 'datasets' library.
    Constructs 'messages' from prompt_key and response_key.
    Uses key_map to map other dataset fields to custom keys in the output sample.
    Yields valid sample dictionaries up to max_samples.
    """
    try:
        from datasets import (
            Dataset,
            DatasetDict,
            IterableDataset,
            IterableDatasetDict,
            load_dataset,
        )

        # Also consider specific exceptions from datasets like DatasetNotFoundError
    except ImportError:
        logger.error(
            "The 'datasets' library is required to load samples from HuggingFace. "
            "Please install it with 'pip install datasets'."
        )
        return

    count = 0
    processed_records = 0
    try:
        hf_dataset = load_dataset(dataset_name, split=split, streaming=True)  # Use streaming
    except Exception as e:  # Broad exception for now, can be more specific
        logger.error(f"Error loading HuggingFace dataset '{dataset_name}' (split: {split}): {e}")
        return

    if not isinstance(
        hf_dataset, (DatasetDict, Dataset, IterableDatasetDict, IterableDataset)
    ):  # Should be IterableDataset due to streaming=True
        logger.error(f"Loaded HuggingFace dataset '{dataset_name}' is not a recognized Dataset type.")
        return

    logger.info(f"Streaming samples from HuggingFace dataset '{dataset_name}' (split: {split}).")
    for record in hf_dataset:
        processed_records += 1
        if count >= max_samples:
            logger.info(f"Reached max_samples ({max_samples}). Stopping HuggingFace sample loading.")
            break

        if not isinstance(record, dict):
            logger.warning(f"HuggingFace dataset record {processed_records} is not a dictionary. Skipping.")
            continue

        prompt = record.get(prompt_key)
        response_content = record.get(response_key)

        if not isinstance(prompt, str):
            logger.warning(
                f"HuggingFace record {processed_records}: Prompt key '{prompt_key}' (value: {str(prompt)[:50]}...) did not yield a string. Skipping sample."
            )
            continue
        if not isinstance(response_content, str):
            logger.warning(
                f"HuggingFace record {processed_records}: Response key '{response_key}' (value: {str(response_content)[:50]}...) did not yield a string. Skipping sample."
            )
            continue

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_content},
        ]

        if not _validate_sample_messages(messages, count + 1, processed_records):
            continue

        sample_output: Dict[str, Any] = {"messages": messages}

        if key_map:
            for source_key_in_record, target_key_in_sample in key_map.items():
                if source_key_in_record in record:
                    sample_output[target_key_in_sample] = record[source_key_in_record]
                else:
                    logger.warning(
                        f"HuggingFace record {processed_records}: Key '{source_key_in_record}' from key_map not found. It will be omitted."
                    )

        yield sample_output
        count += 1

    if count == 0:
        logger.info(
            f"No valid samples loaded from HuggingFace dataset '{dataset_name}' (split: {split}) after processing {processed_records} records."
        )
