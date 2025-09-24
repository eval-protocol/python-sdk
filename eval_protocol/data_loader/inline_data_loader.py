from collections.abc import Sequence
from dataclasses import dataclass

from eval_protocol.data_loader.models import (
    DataLoaderResult,
    DataLoaderVariant,
    EvaluationDataLoader,
)
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.types import InputMessagesParam


DEFAULT_VARIANT_ID: str = "inline"


@dataclass(kw_only=True)
class InlineDataLoader(EvaluationDataLoader):
    """Data loader for inline ``EvaluationRow`` or message payloads."""

    rows: list[EvaluationRow] | None = None
    """Pre-defined evaluation rows with tools and metadata. Use this when you have complete
    EvaluationRow objects that include tools, input_metadata, and other structured data.
    This is the preferred option when working with tool-calling scenarios or when you need
    to provide additional metadata like row_id, dataset information, or custom fields."""

    messages: Sequence[InputMessagesParam] | None = None
    """Raw chat completion message history. Use this when you only have simple
    conversation history without tools or additional metadata. The messages will be
    automatically converted to EvaluationRow objects. InputMessagesParam is a list of
    Message objects representing the conversation flow (user, assistant, system messages)."""

    id: str = DEFAULT_VARIANT_ID
    """Unique identifier for this data loader variant. Used to label and distinguish
    different input data sources, versions, or configurations. This helps with tracking
    and organizing evaluation results from different data sources."""

    description: str | None = None
    """Optional human-readable description of this data loader. Provides additional
    context about the data source, purpose, or any special characteristics. Used for
    documentation and debugging purposes. If not provided, the variant_id will be used instead."""

    def __post_init__(self) -> None:
        if self.rows is None and self.messages is None:
            raise ValueError("InlineDataLoader requires rows or messages to be provided")

    def variants(self) -> Sequence[DataLoaderVariant]:
        def _load() -> DataLoaderResult:
            resolved_rows: list[EvaluationRow] = []
            if self.rows is not None:
                resolved_rows = [row.model_copy(deep=True) for row in self.rows]
            if self.messages is not None:
                for dataset_messages in self.messages:
                    row_messages: list[Message] = []
                    for msg in dataset_messages:
                        if isinstance(msg, Message):
                            row_messages.append(msg.model_copy(deep=True))
                        else:
                            row_messages.append(Message.model_validate(msg))
                    resolved_rows.append(EvaluationRow(messages=row_messages))

            return DataLoaderResult(
                rows=resolved_rows,
                variant_id=self.id,
                variant_description=self.description,
                type=self.__class__.__name__,
            )

        return [_load]
