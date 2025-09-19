"""Data loader abstractions"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable
from typing_extensions import Protocol

from pydantic import BaseModel, Field

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.types import EvaluationTestMode
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger


class DataLoaderContext(BaseModel):
    """Context provided to loader variants when materializing data."""

    max_rows: int | None = Field(default=None, ge=1, description="Maximum number of rows to load")
    preprocess_fn: Callable[[list[EvaluationRow]], list[EvaluationRow]] | None = Field(
        default=None, description="Optional preprocessing function for evaluation rows"
    )
    logger: DatasetLogger = Field(description="Dataset logger for tracking operations")
    invocation_id: str = Field(description="Unique identifier for this invocation")
    experiment_id: str = Field(description="Unique identifier for this experiment")
    mode: EvaluationTestMode = Field(description="The evaluation test mode")

    class Config:
        arbitrary_types_allowed = True  # For Callable and DatasetLogger types


class DataLoaderResult(BaseModel):
    """Rows and metadata returned by a loader variant."""

    rows: list[EvaluationRow] = Field(description="List of evaluation rows loaded")
    source_id: str = Field(description="Unique identifier for the data source")
    source_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the data source"
    )
    raw_payload: Any | None = Field(default=None, description="Raw payload data if available")
    preprocessed: bool = Field(default=False, description="Whether the data has been preprocessed")

    class Config:
        arbitrary_types_allowed = True  # For Any type in raw_payload


class DataLoaderVariant(BaseModel):
    """Single parameterizable variant from a data loader."""

    id: str = Field(description="Unique identifier for this variant")
    description: str = Field(description="Human-readable description of this variant")
    loader: Callable[[DataLoaderContext], DataLoaderResult] = Field(
        description="Function that loads data for this variant"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata for this variant")

    class Config:
        arbitrary_types_allowed = True  # For Callable type

    def load(self, ctx: DataLoaderContext) -> DataLoaderResult:
        """Load a dataset for this variant using the provided context."""

        return self.loader(ctx)


class EvaluationDataLoader(Protocol):
    """Protocol for data loaders that can be consumed by ``evaluation_test``."""

    def variants(self) -> Sequence[DataLoaderVariant]:
        """Return parameterizable variants emitted by this loader."""
        ...
