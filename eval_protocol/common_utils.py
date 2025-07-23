import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a JSONL file where each line is a valid JSON object and returns a list of these objects.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        A list of dictionaries, where each dictionary is a parsed JSON object from a line.
        Returns an empty list if the file is not found or if errors occur during parsing,
        with errors logged.
    """
    data: List[Dict[str, Any]] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Error decoding JSON on line {i+1} in {file_path}: {e} - Line: '{line.strip()}'"
                    )
                    # Optionally, re-raise, or return partial data, or handle as per desired strictness
                    # For now, we'll log and continue, returning successfully parsed lines.
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {file_path}: {e}")
        return []
    return data
