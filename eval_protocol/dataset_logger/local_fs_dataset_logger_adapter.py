import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from eval_protocol.common_utils import load_jsonl
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.dataset_logger.directory_utils import find_eval_protocol_datasets_dir
from eval_protocol.singleton_lock import acquire_singleton_lock, release_singleton_lock

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

    def _acquire_file_lock(self, file_path: str, timeout: float = 30.0) -> bool:
        """
        Acquire a lock for a specific file using the singleton lock mechanism.

        Args:
            file_path: Path to the file to lock
            timeout: Maximum time to wait for lock acquisition in seconds

        Returns:
            True if lock was acquired, False if timeout occurred
        """
        # Create a lock name based on the file path
        lock_name = f"file_lock_{os.path.basename(file_path)}"
        base_dir = Path(os.path.dirname(file_path))

        start_time = time.time()
        while time.time() - start_time < timeout:
            result = acquire_singleton_lock(base_dir, lock_name)
            if result is None:
                # Successfully acquired lock
                return True
            else:
                # Lock is held by another process, wait and retry
                time.sleep(0.1)

        return False

    def _release_file_lock(self, file_path: str) -> None:
        """
        Release the lock for a specific file.

        Args:
            file_path: Path to the file to unlock
        """
        lock_name = f"file_lock_{os.path.basename(file_path)}"
        base_dir = Path(os.path.dirname(file_path))
        release_singleton_lock(base_dir, lock_name)

    def log(self, row: "EvaluationRow") -> None:
        """Log a row, updating existing row with same ID or appending new row."""
        row_id = row.input_metadata.row_id

        # Check if row with this ID already exists in any JSONL file
        if os.path.exists(self.datasets_dir):
            for filename in os.listdir(self.datasets_dir):
                if filename.endswith(".jsonl"):
                    file_path = os.path.join(self.datasets_dir, filename)
                    if os.path.exists(file_path):
                        if self._acquire_file_lock(file_path):
                            try:
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
                            finally:
                                self._release_file_lock(file_path)

        # If no existing row found, append new row to current file
        if self._acquire_file_lock(self.current_jsonl_path):
            try:
                with open(self.current_jsonl_path, "a") as f:
                    f.write(row.model_dump_json(exclude_none=True) + os.linesep)
            finally:
                self._release_file_lock(self.current_jsonl_path)
        else:
            raise RuntimeError(f"Failed to acquire lock for log file {self.current_jsonl_path}")

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
                if self._acquire_file_lock(file_path):
                    try:
                        data = load_jsonl(file_path)
                        for r in data:
                            row = EvaluationRow(**r)
                            if row.input_metadata.row_id not in existing_row_ids:
                                existing_row_ids.add(row.input_metadata.row_id)
                            else:
                                raise ValueError(f"Duplicate Row ID {row.input_metadata.row_id} already exists")
                            all_rows.append(row)
                    finally:
                        self._release_file_lock(file_path)

        if row_id:
            # Filter by row_id if specified
            return [row for row in all_rows if getattr(row.input_metadata, "row_id", None) == row_id]
        else:
            return all_rows
