from datetime import datetime, timezone
import json
import os
import tempfile
import shutil
from typing import TYPE_CHECKING, List, Optional
from eval_protocol.common_utils import load_jsonl
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger

if TYPE_CHECKING:
    from eval_protocol.models import EvaluationRow


class LocalFSDatasetLoggerAdapter(DatasetLogger):
    """
    Logger that stores logs in the local filesystem.
    """

    EVAL_PROTOCOL_DIR = ".eval_protocol"
    PYTHON_FILES = ["pyproject.toml", "requirements.txt"]
    DATASETS_DIR = "datasets"

    def __init__(self):
        # recursively look up for a .eval_protocol directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while current_dir != "/":
            if os.path.exists(os.path.join(current_dir, self.EVAL_PROTOCOL_DIR)):
                self.log_dir = os.path.join(current_dir, self.EVAL_PROTOCOL_DIR)
                break
            current_dir = os.path.dirname(current_dir)

        # if not found, recursively look up until a pyproject.toml or requirements.txt is found
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while current_dir != "/":
            if any(os.path.exists(os.path.join(current_dir, f)) for f in self.PYTHON_FILES):
                self.log_dir = os.path.join(current_dir, self.EVAL_PROTOCOL_DIR)
                break
            current_dir = os.path.dirname(current_dir)

        # get the PWD that this python process is running in
        self.log_dir = os.path.join(os.getcwd(), self.EVAL_PROTOCOL_DIR)

        # create the .eval_protocol directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # create the datasets subdirectory
        self.datasets_dir = os.path.join(self.log_dir, self.DATASETS_DIR)
        os.makedirs(self.datasets_dir, exist_ok=True)

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

        # Check if row with this ID already exists
        if os.path.exists(self.current_jsonl_path):
            with open(self.current_jsonl_path, "r") as f:
                lines = f.readlines()

            # Find the line with matching ID
            for i, line in enumerate(lines):
                try:
                    line_data = json.loads(line.strip())
                    if line_data["input_metadata"]["row_id"] == row_id:
                        # Update existing row
                        lines[i] = row.model_dump_json(exclude_none=True) + os.linesep
                        with open(self.current_jsonl_path, "w") as f:
                            f.writelines(lines)
                        return
                except json.JSONDecodeError:
                    continue

        # If no existing row found, append new row
        with open(self.current_jsonl_path, "a") as f:
            f.write(row.model_dump_json(exclude_none=True) + os.linesep)

    def read(self, row_id: Optional[str] = None) -> List["EvaluationRow"]:
        """Read rows from all JSONL files in the datasets directory."""
        from eval_protocol.models import EvaluationRow

        if not os.path.exists(self.datasets_dir):
            return []

        all_rows = []
        for filename in os.listdir(self.datasets_dir):
            if filename.endswith(".jsonl"):
                file_path = os.path.join(self.datasets_dir, filename)
                try:
                    data = load_jsonl(file_path)
                    all_rows.extend([EvaluationRow(**r) for r in data])
                except Exception:
                    continue  # skip files that can't be read/parsed

        if row_id:
            # Filter by row_id if specified
            return [row for row in all_rows if getattr(row.input_metadata, "row_id", None) == row_id]
        else:
            return all_rows
