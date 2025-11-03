import json
import re
from typing import Any, Dict, List

import requests


def get_user_agent() -> str:
    """
    Returns the user-agent string for eval-protocol CLI requests.

    Format: eval-protocol-cli/{version}

    Returns:
        User-agent string identifying the eval-protocol CLI and version.
    """
    try:
        from . import __version__

        return f"eval-protocol/{__version__}"
    except Exception:
        return "eval-protocol/unknown"


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a JSONL file where each line is a valid JSON object and returns a list of these objects.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        A list of dictionaries, where each dictionary is a parsed JSON object from a line.
        Returns an empty list if the file is not found or if errors occur during parsing. Supports HTTP urls and local file paths.
    """
    data: List[Dict[str, Any]] = []
    if file_path.startswith("http://") or file_path.startswith("https://"):
        resp = requests.get(file_path, stream=True, timeout=30)
        resp.raise_for_status()
        for line_number, raw in enumerate(resp.iter_lines(decode_unicode=True), start=1):
            if raw is None:
                continue
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                data.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line for URL {file_path} at line {line_number}")
                row_id_index = stripped.find("row_id")
                if row_id_index != -1:
                    row_id = re.search(r'"row_id": (.*),', stripped[row_id_index:])
                    raise ValueError(f"{e.msg} at line {line_number}: {stripped} ({row_id})") from e
                raise e
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                # Skip entirely blank or whitespace-only lines to be robust to trailing newlines
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data.append(json.loads(stripped))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line for file {file_path} at line {line_number}")
                    # attempt to find "row_id" in the line by finding index of "row_id" and performing regex of `"row_id": (.*),`
                    row_id_index = line.find("row_id")
                    if row_id_index != -1:
                        row_id = re.search(r'"row_id": (.*),', line[row_id_index:])
                        raise ValueError(f"{e.msg} at line {line_number}: {line} ({row_id})") from e
                    raise e
    return data
