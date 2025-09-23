"""Data loader abstractions"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable
from typing_extensions import Protocol

from pydantic import BaseModel, Field, field_validator

from eval_protocol.models import EvaluationRow


class DataLoaderContext(BaseModel):
    """Context provided to loader variants when materializing data. This is mainly used internally by eval-protocol."""

    preprocess_fn: Callable[[list[EvaluationRow]], list[EvaluationRow]] | None = Field(
        default=None,
        description="Optional preprocessing function for evaluation rows. This function is applied "
        "to the loaded data before it's returned, allowing for data cleaning, transformation, "
        "filtering, or other modifications. The function receives a list of EvaluationRow objects "
        "and should return a modified list of EvaluationRow objects.",
    )
    variant_id: str = Field(
        ...,
        description="Unique identifier for the data loader variant. Used to distinguish between "
        "different variants of the same data loader and for tracking purposes in evaluation results.",
    )
    variant_description: str | None = Field(
        default=None,
        description="Human-readable description of the data loader variant. Provides context about what "
        "this variant represents, its purpose, or any special characteristics that distinguish "
        "it from other variants.",
    )

    @field_validator("variant_id")
    @classmethod
    def validate_variant_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("variant_id must be non-empty")
        return v


class DataLoaderResult(BaseModel):
    """Rows and metadata returned by a loader variant."""

    rows: list[EvaluationRow] = Field(
        description="List of evaluation rows loaded from the data source. These are the "
        "processed and ready-to-use evaluation data that will be fed into the evaluation pipeline."
    )
    num_rows: int = Field(
        ...,
        description="Number of rows loaded. This should match the length of the rows list "
        "and is used for validation and reporting purposes.",
    )
    type: str = Field(
        ...,
        description="Type of the data loader that produced this result. Used for identification "
        "and debugging purposes (e.g., 'InlineDataLoader', 'FactoryDataLoader').",
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

    @field_validator("num_rows")
    @classmethod
    def validate_num_rows(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_rows must be greater than 0")
        return v

    @field_validator("variant_id")
    @classmethod
    def validate_variant_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("variant_id must be non-empty")
        return v


class DataLoaderVariant(BaseModel):
    """Single parameterizable variant from a data loader."""

    id: str = Field(
        description="Unique identifier for this variant. Used to distinguish between different "
        "variants of the same data loader and for tracking purposes in evaluation results."
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of this variant. Provides context about what "
        "this variant represents, its purpose, or any special characteristics that distinguish "
        "it from other variants.",
    )
    loader: Callable[[DataLoaderContext], DataLoaderResult] = Field(
        description="Function that loads data for this variant. This callable is invoked with "
        "a DataLoaderContext and should return a DataLoaderResult containing the loaded "
        "evaluation rows and associated metadata. The loader function is responsible for "
        "the actual data retrieval and any necessary processing."
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("DataLoaderVariant.id must be non-empty")
        return v

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

    def load(self, ctx: DataLoaderContext) -> list[DataLoaderResult]:
        """
        Loads all variants of this data loader and return a list of DataLoaderResult.
        """
        variants = self.variants()
        return [variant.load(ctx) for variant in variants]
