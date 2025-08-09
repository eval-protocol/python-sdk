import json
import re
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
        for line_number, line in enumerate(f):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line for file {file_path} at line {line_number}")
                # attempt to find "row_id" in the line by finding index of "row_id" and performing regex of `"row_id": (.*),`
                row_id_index = line.find("row_id")
                if row_id_index != -1:
                    row_id = re.search(r'"row_id": (.*),', line[row_id_index:])
                    raise ValueError(f"{e.msg} at line {line_number}: {line} ({row_id})")
                raise e
    return data
