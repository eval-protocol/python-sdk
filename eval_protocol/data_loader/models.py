"""Data loader abstractions"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable
from typing_extensions import Protocol
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, field_validator

from eval_protocol.models import EvaluationRow


class DataLoaderResult(BaseModel):
    """Rows and metadata returned by a loader variant."""

    rows: list[EvaluationRow] = Field(
        description="List of evaluation rows loaded from the data source. These are the "
        "processed and ready-to-use evaluation data that will be fed into the evaluation pipeline."
    )

    type: str = Field(
        ...,
        description="Type of the data loader that produced this result. Used for identification "
        "and debugging purposes (e.g., 'InlineDataLoader', 'DynamicDataLoader').",
    )

    variant_id: str = Field(
        ...,
        description="Unique identifier for the data loader variant that produced this result. "
        "Used for tracking and organizing evaluation results from different data sources.",
    )

    variant_description: str | None = Field(
        default=None,
        description="Human-readable description of the data loader variant that produced this result. "
        "Provides context about what this variant represents, its purpose, or any special characteristics that distinguish "
        "it from other variants.",
    )

    preprocessed: bool = Field(
        default=False,
        description="Whether the data has been preprocessed. This flag indicates if any "
        "preprocessing functions have been applied to the data, helping to avoid duplicate "
        "processing and track data transformation state.",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("type must be non-empty")
        return v

    @field_validator("variant_id")
    @classmethod
    def validate_variant_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("variant_id must be non-empty")
        return v


class DataLoaderVariant(Protocol):
    """Single parameterizable variant from a data loader."""

    def __call__(self) -> DataLoaderResult:
        """Load a dataset for this variant using the provided context."""
        ...


@dataclass(kw_only=True)
class EvaluationDataLoader(ABC):
    """Abstract base class for data loaders that can be consumed by ``evaluation_test``."""

    preprocess_fn: Callable[[list[EvaluationRow]], list[EvaluationRow]] | None = None
    """Optional preprocessing function for evaluation rows. This function is applied
    to the loaded data before it's returned, allowing for data cleaning, transformation,
    filtering, or other modifications. The function receives a list of EvaluationRow objects
    and should return a modified list of EvaluationRow objects."""

    @abstractmethod
    def variants(self) -> Sequence[DataLoaderVariant]:
        """Return parameterizable variants emitted by this loader."""
        ...

    def load(self) -> list[DataLoaderResult]:
        """Loads all variants of this data loader and return a list of DataLoaderResult."""
        results = []
        for variant in self.variants():
            result = variant()
            result = self._process_variant(result)
            results.append(result)
        return results

    def _process_variant(self, result: DataLoaderResult) -> DataLoaderResult:
        """Process a single variant: preprocess data and apply metadata."""
        # Preprocess data
        original_count = len(result.rows)
        if self.preprocess_fn:
            result.rows = self.preprocess_fn(result.rows)
            result.preprocessed = True
            processed_count = len(result.rows)
        else:
            processed_count = original_count

        # Apply metadata to rows
        self._apply_metadata(result, original_count, processed_count)
        return result

    def _apply_metadata(self, result: DataLoaderResult, original_count: int, processed_count: int) -> None:
        """Apply metadata to all rows in the result."""
        for row in result.rows:
            if row.input_metadata.dataset_info is None:
                row.input_metadata.dataset_info = {}

            # Apply result attributes as metadata
            for attr_name, attr_value in vars(result).items():
                """
                Exclude rows and private attributes from metadata.
                """
                if attr_name != "rows" and not attr_name.startswith("_"):
                    row.input_metadata.dataset_info[f"data_loader_{attr_name}"] = attr_value

            # Apply row counts
            row.input_metadata.dataset_info["data_loader_num_rows"] = original_count
            row.input_metadata.dataset_info["data_loader_num_rows_after_preprocessing"] = processed_count
