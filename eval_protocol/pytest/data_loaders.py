"""Data loader abstractions for evaluation tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from eval_protocol.adapters.base import BaseAdapter
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.types import EvaluationTestMode, InputMessagesParam
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger


@dataclass(slots=True)
class DataLoaderContext:
    """Context provided to loader variants when materializing data."""

    max_rows: int | None
    preprocess_fn: Callable[[list[EvaluationRow]], list[EvaluationRow]] | None
    logger: DatasetLogger
    invocation_id: str
    experiment_id: str
    mode: EvaluationTestMode


@dataclass(slots=True)
class DataLoaderResult:
    """Rows and metadata returned by a loader variant."""

    rows: list[EvaluationRow]
    source_id: str
    source_metadata: dict[str, Any] = field(default_factory=dict)
    raw_payload: Any | None = None
    preprocessed: bool = False


@dataclass(slots=True)
class DataLoaderVariant:
    """Single parameterizable variant from a data loader."""

    id: str
    description: str
    loader: Callable[[DataLoaderContext], DataLoaderResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def load(self, ctx: DataLoaderContext) -> DataLoaderResult:
        """Load a dataset for this variant using the provided context."""

        return self.loader(ctx)


class EvaluationDataLoader(Protocol):
    """Protocol for data loaders that can be consumed by ``evaluation_test``."""

    def variants(self) -> Sequence[DataLoaderVariant]:
        """Return parameterizable variants emitted by this loader."""

        ...


@dataclass(slots=True)
class InlineDataLoader(EvaluationDataLoader):
    """Data loader for inline ``EvaluationRow`` or message payloads."""

    rows: Sequence[EvaluationRow] | None = None
    messages: Sequence[InputMessagesParam] | None = None
    variant_id: str = "inline"
    description: str | None = None

    def __post_init__(self) -> None:
        if self.rows is None and self.messages is None:
            raise ValueError("InlineDataLoader requires rows or messages to be provided")

    def variants(self) -> Sequence[DataLoaderVariant]:
        def _load(ctx: DataLoaderContext) -> DataLoaderResult:
            resolved_rows: list[EvaluationRow] = []
            if self.rows is not None:
                resolved_rows.extend(row.model_copy(deep=True) for row in self.rows)
            if self.messages is not None:
                for dataset_messages in self.messages:
                    row_messages: list[Message] = []
                    for msg in dataset_messages:
                        if isinstance(msg, Message):
                            row_messages.append(msg.model_copy(deep=True))
                        else:
                            row_messages.append(Message.model_validate(msg))
                    resolved_rows.append(EvaluationRow(messages=row_messages))

            if ctx.max_rows is not None:
                resolved_rows = resolved_rows[: ctx.max_rows]

            metadata = {
                "data_loader_variant_id": self.variant_id,
                "data_loader_type": "inline",
                "row_count": len(resolved_rows),
            }

            return DataLoaderResult(
                rows=resolved_rows,
                source_id=self.variant_id,
                source_metadata=metadata,
            )

        description = self.description or self.variant_id
        return [
            DataLoaderVariant(
                id=self.variant_id,
                description=description,
                loader=_load,
                metadata={"type": "inline"},
            )
        ]


@dataclass(slots=True)
class LangfuseLoaderConfig:
    """Configuration for a single Langfuse adapter variant."""

    id: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    description: str | None = None


@dataclass(slots=True)
class LangfuseAdapterLoader(EvaluationDataLoader):
    """Wrap a ``LangfuseAdapter`` (or compatible adapter) as a data loader."""

    adapter: BaseAdapter
    variants_config: Sequence[LangfuseLoaderConfig]

    def variants(self) -> Sequence[DataLoaderVariant]:
        loader_variants: list[DataLoaderVariant] = []

        for config in self.variants_config:

            def _load(ctx: DataLoaderContext, *, _config: LangfuseLoaderConfig = config) -> DataLoaderResult:
                rows = self.adapter.get_evaluation_rows(**_config.kwargs)
                if ctx.max_rows is not None:
                    rows = rows[: ctx.max_rows]

                metadata = {
                    "data_loader_variant_id": _config.id,
                    "data_loader_type": "langfuse",
                    "adapter_kwargs": _config.kwargs,
                }

                return DataLoaderResult(
                    rows=[row.model_copy(deep=True) for row in rows],
                    source_id=_config.id,
                    source_metadata=metadata,
                )

            loader_variants.append(
                DataLoaderVariant(
                    id=config.id,
                    description=config.description or config.id,
                    loader=_load,
                    metadata={"type": "langfuse", "adapter_kwargs": config.kwargs},
                )
            )

        return loader_variants


__all__ = [
    "DataLoaderContext",
    "DataLoaderResult",
    "DataLoaderVariant",
    "EvaluationDataLoader",
    "InlineDataLoader",
    "LangfuseAdapterLoader",
    "LangfuseLoaderConfig",
]
