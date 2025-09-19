from collections.abc import Sequence

from eval_protocol.data_loader.models import (
    DataLoaderContext,
    DataLoaderResult,
    DataLoaderVariant,
    EvaluationDataLoader,
)
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.types import InputMessagesParam


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
