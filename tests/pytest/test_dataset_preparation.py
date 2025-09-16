from __future__ import annotations

import importlib
import sys
import types
from typing import cast

import pytest
from pydantic import BaseModel, ConfigDict


def _install_dependency_stubs() -> None:
    """Register lightweight stubs for optional runtime dependencies."""

    def _ensure_module(name: str, **attrs) -> None:
        if name in sys.modules:
            return
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module

    try:  # pragma: no cover - prefer real dependency when available
        importlib.import_module("loguru")
    except ModuleNotFoundError:
        class _Logger:  # pragma: no cover - inert logging shim
            def __getattr__(self, _name: str):
                def _noop(*_args, **_kwargs):
                    return None

                return _noop

        _ensure_module("loguru", logger=_Logger())

    def _noop_loader(*_args, **_kwargs):  # pragma: no cover - placeholder loader
        return {}

    optional_stub_attrs = {
        "toml": {"loads": _noop_loader, "load": _noop_loader},
        "datasets": {},
        "addict": {"Dict": dict},
        "deepdiff": {},
        "litellm": {},
        "peewee": {},
        "backoff": {},
    }

    for optional_module, attrs in optional_stub_attrs.items():
        try:
            importlib.import_module(optional_module)
        except ModuleNotFoundError:
            _ensure_module(optional_module, **attrs)

    try:
        importlib.import_module("openai")
        return
    except ModuleNotFoundError:
        pass

    openai_mod = types.ModuleType("openai")
    types_mod = types.ModuleType("openai.types")
    completion_usage_mod = types.ModuleType("openai.types.completion_usage")
    chat_mod = types.ModuleType("openai.types.chat")
    chat_message_mod = types.ModuleType("openai.types.chat.chat_completion_message")
    tool_call_mod = types.ModuleType("openai.types.chat.chat_completion_message_tool_call")

    class CompletionUsage(BaseModel):  # pragma: no cover - simple data container
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None

        model_config = ConfigDict(extra="allow")

    class FunctionCall(BaseModel):  # pragma: no cover - simple data container
        name: str | None = None
        arguments: str | None = None

        model_config = ConfigDict(extra="allow")

    class ChatCompletionMessageToolCall(BaseModel):  # pragma: no cover - simple data container
        id: str | None = None
        type: str | None = None
        function: FunctionCall | None = None

        model_config = ConfigDict(extra="allow")

    types_mod.CompletionUsage = CompletionUsage
    completion_usage_mod.CompletionUsage = CompletionUsage
    chat_message_mod.FunctionCall = FunctionCall
    tool_call_mod.ChatCompletionMessageToolCall = ChatCompletionMessageToolCall

    openai_mod.types = types_mod
    types_mod.completion_usage = completion_usage_mod
    types_mod.chat = chat_mod
    chat_mod.chat_completion_message = chat_message_mod
    chat_mod.chat_completion_message_tool_call = tool_call_mod

    sys.modules["openai"] = openai_mod
    sys.modules["openai.types"] = types_mod
    sys.modules["openai.types.completion_usage"] = completion_usage_mod
    sys.modules["openai.types.chat"] = chat_mod
    sys.modules["openai.types.chat.chat_completion_message"] = chat_message_mod
    sys.modules["openai.types.chat.chat_completion_message_tool_call"] = tool_call_mod


_install_dependency_stubs()

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
