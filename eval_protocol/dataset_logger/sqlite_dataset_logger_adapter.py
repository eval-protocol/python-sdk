import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.dataset_logger.directory_utils import find_eval_protocol_dir

if TYPE_CHECKING:
    from eval_protocol.models import EvaluationRow


class SqliteDatasetLoggerAdapter(DatasetLogger):
    def __init__(self, db_path: Optional[str] = None):
        eval_protocol_dir = find_eval_protocol_dir()
        self.db_path = os.path.join(eval_protocol_dir, "logs.db")

    def log(self, row: "EvaluationRow") -> None:
        pass

    def read(self, row_id: Optional[str] = None) -> List["EvaluationRow"]:
        return []
