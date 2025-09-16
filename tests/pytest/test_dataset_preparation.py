from typing import cast

import pytest

pytest.importorskip("openai")

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.dataset_preparation import load_and_prepare_rows
from eval_protocol.pytest.generate_parameter_combinations import ParameterizedTestKwargs


def _make_kwargs(**overrides) -> ParameterizedTestKwargs:
    base: ParameterizedTestKwargs = {
        "dataset_path": None,
        "completion_params": None,
        "input_messages": None,
        "input_rows": None,
        "evaluation_test_kwargs": None,
    }
    base.update(overrides)
    return cast(ParameterizedTestKwargs, base)


def test_load_and_prepare_rows_from_dataset(monkeypatch):
    dataset_contents = {
        "file1": [{"text": "f1a"}, {"text": "f1b"}],
        "file2": [{"text": "f2a"}, {"text": "f2b"}],
    }
    load_calls: list[str] = []

    def fake_load_jsonl(path: str):
        load_calls.append(path)
        return dataset_contents[path]

    monkeypatch.setattr("eval_protocol.pytest.dataset_preparation.load_jsonl", fake_load_jsonl)

    generated_args: list[dict[str, int | None]] = []

    def fake_generate_id(separator: str = "-", seed: int | None = None, index: int | None = None) -> str:
        generated_args.append({"seed": seed, "index": index})
        return f"id-{index}"

    monkeypatch.setattr("eval_protocol.pytest.dataset_preparation.generate_id", fake_generate_id)
    monkeypatch.setattr("eval_protocol.pytest.dataset_preparation.num_combinations", lambda: 10)

    adapter_inputs: list[list[dict[str, str]]] = []

    def dataset_adapter(data):
        adapter_inputs.append(list(data))
        rows: list[EvaluationRow] = []
        for entry in data:
            rows.append(EvaluationRow(messages=[Message(role="user", content=entry["text"])]))
        return rows

    preprocess_calls: list[list[EvaluationRow]] = []

    def preprocess(rows: list[EvaluationRow]) -> list[EvaluationRow]:
        preprocess_calls.append(list(rows))
        return rows

    kwargs = _make_kwargs(dataset_path=["file1", "file2"])

    result = load_and_prepare_rows(
        kwargs,
        dataset_adapter=dataset_adapter,
        preprocess_fn=preprocess,
        max_dataset_rows=3,
    )

    assert load_calls == ["file1", "file2"], "Expected to load all dataset paths"
    assert len(adapter_inputs) == 1
    assert len(adapter_inputs[0]) == 3, "max_dataset_rows should truncate concatenated data"
    assert preprocess_calls and preprocess_calls[0] == result
    assert all(row.input_metadata.row_id is not None for row in result)
    assert len(generated_args) == len(result)
    assert all(call["seed"] == 0 for call in generated_args)
    assert all(0 <= call["index"] < 10 for call in generated_args if call["index"] is not None)


def test_load_and_prepare_rows_from_messages(monkeypatch):
    generated_indices: list[int | None] = []

    def fake_generate_id(separator: str = "-", seed: int | None = None, index: int | None = None) -> str:
        generated_indices.append(index)
        return f"row-{index}"

    monkeypatch.setattr("eval_protocol.pytest.dataset_preparation.generate_id", fake_generate_id)
    monkeypatch.setattr("eval_protocol.pytest.dataset_preparation.num_combinations", lambda: 8)

    kwargs = _make_kwargs(
        input_messages=[
            [Message(role="system", content="system")],
            [Message(role="user", content="question")],
        ]
    )

    result = load_and_prepare_rows(
        kwargs,
        dataset_adapter=lambda data: pytest.fail("dataset_adapter should not be used"),
        preprocess_fn=None,
        max_dataset_rows=None,
    )

    assert [row.messages for row in result] == kwargs["input_messages"]
    assert generated_indices and all(index is not None for index in generated_indices)


def test_load_and_prepare_rows_deep_copies_input_rows(monkeypatch):
    def fail_generate_id(*_args, **_kwargs):  # pragma: no cover - should never be called
        raise AssertionError("generate_id should not be called when row_id already exists")

    monkeypatch.setattr("eval_protocol.pytest.dataset_preparation.generate_id", fail_generate_id)

    original = EvaluationRow(messages=[Message(role="user", content="hi")])
    original.input_metadata.row_id = "existing-id"

    kwargs = _make_kwargs(input_rows=[original])

    result = load_and_prepare_rows(
        kwargs,
        dataset_adapter=lambda data: pytest.fail("dataset_adapter should not be used"),
        preprocess_fn=None,
        max_dataset_rows=None,
    )

    assert len(result) == 1
    assert result[0] is not original
    assert result[0].input_metadata.row_id == "existing-id"

    result[0].messages[0].content = "changed"
    assert original.messages[0].content == "hi", "Deep copy should isolate message objects"
