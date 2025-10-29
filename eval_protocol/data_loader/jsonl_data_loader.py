from __future__ import annotations

import os
from dataclasses import dataclass
from collections.abc import Sequence

from eval_protocol.common_utils import load_jsonl
from eval_protocol.pytest.default_dataset_adapter import default_dataset_adapter
from eval_protocol.data_loader.models import (
    DataLoaderResult,
    DataLoaderVariant,
    EvaluationDataLoader,
)


@dataclass(kw_only=True)
class EvaluationRowJsonlDataLoader(EvaluationDataLoader):
    """Data loader that reads EvaluationRows from a JSONL file path.

    Each line of the JSONL file should be a serialized EvaluationRow dict.
    The loader will construct EvaluationRow objects via the default dataset adapter.
    """

    jsonl_path: str
    id: str = "jsonl"
    description: str | None = None

    def variants(self) -> Sequence[DataLoaderVariant]:
        def _load() -> DataLoaderResult:
            path = self.jsonl_path
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            rows_json = load_jsonl(path)
            eval_rows = default_dataset_adapter(rows_json)
            return DataLoaderResult(
                rows=eval_rows,
                type=self.__class__.__name__,
                variant_id=self.id,
                variant_description=self.description,
            )

        return [_load]
