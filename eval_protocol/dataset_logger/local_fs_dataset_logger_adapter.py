import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from eval_protocol.common_utils import load_jsonl
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.dataset_logger.directory_utils import find_eval_protocol_datasets_dir

if TYPE_CHECKING:
    from eval_protocol.models import EvaluationRow


class LocalFSDatasetLoggerAdapter(DatasetLogger):
    """
    Logger that stores logs in the local filesystem with file locking to prevent race conditions.
    """

    def __init__(self):
        self.log_dir = os.path.dirname(find_eval_protocol_datasets_dir())
        self.datasets_dir = find_eval_protocol_datasets_dir()

        # ensure that log file exists
        if not os.path.exists(self.current_jsonl_path):
            with open(self.current_jsonl_path, "w") as f:
                f.write("")

    @property
    def current_date(self) -> str:
        # Use UTC timezone to be consistent across local device/locations/CI
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @property
    def current_jsonl_path(self) -> str:
        """
        The current JSONL file path. Based on the current date.
        """
        return os.path.join(self.datasets_dir, f"{self.current_date}.jsonl")

    def log(self, row: "EvaluationRow") -> None:
        """Log a row, updating existing row with same ID or appending new row."""
        row_id = row.input_metadata.row_id

        # Check if row with this ID already exists in any JSONL file
        if os.path.exists(self.datasets_dir):
            for filename in os.listdir(self.datasets_dir):
                if filename.endswith(".jsonl"):
                    file_path = os.path.join(self.datasets_dir, filename)
                    if os.path.exists(file_path):
                        with open(file_path, "r") as f:
                            lines = f.readlines()

                        # Find the line with matching ID
                        for i, line in enumerate(lines):
                            try:
                                line_data = json.loads(line.strip())
                                if line_data["input_metadata"]["row_id"] == row_id:
                                    # Update existing row
                                    lines[i] = row.model_dump_json(exclude_none=True) + os.linesep
                                    with open(file_path, "w") as f:
                                        f.writelines(lines)
                                    return
                            except json.JSONDecodeError:
                                continue

        # If no existing row found, append new row to current file
        with open(self.current_jsonl_path, "a") as f:
            f.write(row.model_dump_json(exclude_none=True) + os.linesep)

    def read(self, row_id: Optional[str] = None) -> List["EvaluationRow"]:
        """Read rows from all JSONL files in the datasets directory. Also
        ensures that there are no duplicate row IDs."""
        from eval_protocol.models import EvaluationRow

        if not os.path.exists(self.datasets_dir):
            return []

        all_rows = []
        existing_row_ids = set()
        for filename in os.listdir(self.datasets_dir):
            if filename.endswith(".jsonl"):
                file_path = os.path.join(self.datasets_dir, filename)
                data = load_jsonl(file_path)
                for r in data:
                    row = EvaluationRow(**r)
                    if row.input_metadata.row_id not in existing_row_ids:
                        existing_row_ids.add(row.input_metadata.row_id)
                    else:
                        raise ValueError(f"Duplicate Row ID {row.input_metadata.row_id} already exists")
                    all_rows.append(row)

        if row_id:
            # Filter by row_id if specified
            return [row for row in all_rows if getattr(row.input_metadata, "row_id", None) == row_id]
        else:
            return all_rows
