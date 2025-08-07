import json
from typing import Any, Dict, List


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a JSONL file where each line is a valid JSON object and returns a list of these objects.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        A list of dictionaries, where each dictionary is a parsed JSON object from a line.
        Returns an empty list if the file is not found or if errors occur during parsing.
    """
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data
