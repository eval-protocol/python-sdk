from __future__ import annotations

from typing import Any


import pytest

pytest.importorskip("openai")
pytest.importorskip("loguru")
pytest.importorskip("toml")
pytest.importorskip("addict")
pytest.importorskip("deepdiff")
pytest.importorskip("dotenv")

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import (
    InlineDataLoader,
    LangfuseAdapterLoader,
    LangfuseLoaderConfig,
    NoOpRolloutProcessor,
    evaluation_test,
)


@evaluation_test(
    data_loaders=InlineDataLoader(
        messages=[[Message(role="user", content="What is 2 + 2?")]],
    ),
    completion_params=[{"model": "no-op"}],
    rollout_processor=NoOpRolloutProcessor(),
)
def test_inline_data_loader(row: EvaluationRow) -> EvaluationRow:
    """Inline data loader should feed pre-constructed message bundles."""

    assert row.messages[0].content == "What is 2 + 2?"
    assert row.input_metadata.dataset_info is not None
    assert row.input_metadata.dataset_info.get("data_loader_variant_id") == "inline"
    return row


class _FakeLangfuseAdapter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_evaluation_rows(self, **kwargs: Any) -> list[EvaluationRow]:
        self.calls.append(kwargs)
        return [
            EvaluationRow(messages=[Message(role="user", content="trace-0")]),
            EvaluationRow(messages=[Message(role="user", content="trace-1")]),
        ]


_fake_adapter = _FakeLangfuseAdapter()


def _preprocess_rows(rows: list[EvaluationRow]) -> list[EvaluationRow]:
    for row in rows:
        row.messages[0].content = f"processed-{row.messages[0].content}"
    return rows


@evaluation_test(
    data_loaders=LangfuseAdapterLoader(
        adapter=_fake_adapter,
        variants_config=[LangfuseLoaderConfig(id="recent", kwargs={"limit": 5})],
    ),
    completion_params=[{"model": "no-op"}],
    rollout_processor=NoOpRolloutProcessor(),
    max_dataset_rows=1,
    preprocess_fn=_preprocess_rows,
)
def test_langfuse_data_loader(row: EvaluationRow) -> EvaluationRow:
    """Langfuse data loader should pull traces and respect preprocess/max_rows."""

    assert _fake_adapter.calls == [{"limit": 5}]
    assert row.messages[0].content == "processed-trace-0"
    assert row.input_metadata.dataset_info is not None
    dataset_info = row.input_metadata.dataset_info
    assert dataset_info.get("data_loader_variant_id") == "recent"
    assert dataset_info.get("adapter_kwargs") == {"limit": 5}
    return row
