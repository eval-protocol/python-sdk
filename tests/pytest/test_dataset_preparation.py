from __future__ import annotations

import importlib
from importlib.machinery import ModuleSpec
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

    def _field_type(name: str):
        def __init__(self, *_args, **_kwargs):
            return None

        return type(name, (), {"__init__": __init__})

    class _SqliteDatabase:
        def __init__(self, *_args, **_kwargs):
            self.path = None

        def connect(self):  # pragma: no cover - stub connection
            return None

        def close(self):  # pragma: no cover
            return None

        def atomic(self):  # pragma: no cover - context manager shim
            class _Atomic:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *_exc):
                    return False

            return _Atomic()

        def create_tables(self, *_args, **_kwargs):  # pragma: no cover
            return None

        def create_table(self, *_args, **_kwargs):  # pragma: no cover
            return None

        def drop_tables(self, *_args, **_kwargs):  # pragma: no cover
            return None

    optional_stub_attrs = {
        "toml": {"loads": _noop_loader, "load": _noop_loader},
        "datasets": {},
        "addict": {"Dict": dict},
        "deepdiff": {"DeepDiff": type("DeepDiff", (), {})},
        "peewee": {
            "Model": type("Model", (), {}),
            "SqliteDatabase": _SqliteDatabase,
            "CharField": _field_type("CharField"),
            "TextField": _field_type("TextField"),
            "IntegerField": _field_type("IntegerField"),
            "DateTimeField": _field_type("DateTimeField"),
            "AutoField": _field_type("AutoField"),
            "OperationalError": Exception,
        },
        "backoff": {},
        "aiohttp": {"ClientSession": type("ClientSession", (), {})},
        "tqdm": {"tqdm": lambda iterable, *_args, **_kwargs: iterable},
    }

    for optional_module, attrs in optional_stub_attrs.items():
        try:
            importlib.import_module(optional_module)
        except ModuleNotFoundError:
            _ensure_module(optional_module, **attrs)

    try:
        importlib.import_module("litellm")
    except ModuleNotFoundError:
        litellm_mod = types.ModuleType("litellm")

        def _acompletion(*_args, **_kwargs):  # pragma: no cover - stubbed async function
            return None

        def _completion_cost(*_args, **_kwargs):  # pragma: no cover - cost shim
            return 0.0

        litellm_mod.acompletion = _acompletion
        litellm_mod.completion = _acompletion
        litellm_mod.completion_cost = _completion_cost

        caching_pkg = types.ModuleType("litellm.caching")
        caching_submodule = types.ModuleType("litellm.caching.caching")
        caching_submodule.Cache = type("Cache", (), {})
        dual_cache_module = types.ModuleType("litellm.caching.dual_cache")
        dual_cache_module.DualCache = type("DualCache", (), {})
        in_memory_cache_module = types.ModuleType("litellm.caching.in_memory_cache")
        in_memory_cache_module.InMemoryCache = type("InMemoryCache", (), {})
        caching_pkg.caching = caching_submodule
        caching_pkg.dual_cache = dual_cache_module
        caching_pkg.in_memory_cache = in_memory_cache_module
        redis_cache_module = types.ModuleType("litellm.caching.redis_cache")
        redis_cache_module.RedisCache = type("RedisCache", (), {})
        caching_pkg.redis_cache = redis_cache_module

        litellm_mod.caching = caching_pkg

        main_module = types.ModuleType("litellm.main")
        main_module.ModelResponse = type("ModelResponse", (), {})
        main_module.Usage = type("Usage", (), {})

        cost_calculator_mod = types.ModuleType("litellm.cost_calculator")
        cost_calculator_mod.cost_per_token = lambda *_args, **_kwargs: 0.0

        sys.modules["litellm"] = litellm_mod
        sys.modules["litellm.caching"] = caching_pkg
        sys.modules["litellm.caching.caching"] = caching_submodule
        sys.modules["litellm.caching.dual_cache"] = dual_cache_module
        sys.modules["litellm.caching.in_memory_cache"] = in_memory_cache_module
        sys.modules["litellm.caching.redis_cache"] = redis_cache_module
        sys.modules["litellm.main"] = main_module
        sys.modules["litellm.cost_calculator"] = cost_calculator_mod

    try:
        importlib.import_module("playhouse.sqlite_ext")
    except ModuleNotFoundError:
        playhouse_mod = types.ModuleType("playhouse")
        sqlite_ext_mod = types.ModuleType("playhouse.sqlite_ext")
        sqlite_ext_mod.JSONField = type("JSONField", (), {})
        playhouse_mod.sqlite_ext = sqlite_ext_mod

        sys.modules["playhouse"] = playhouse_mod
        sys.modules["playhouse.sqlite_ext"] = sqlite_ext_mod

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
    chat_message_param_mod = types.ModuleType("openai.types.chat.chat_completion_message_param")
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

    class FunctionDefinition(BaseModel):  # pragma: no cover - simple data container
        name: str | None = None
        description: str | None = None
        parameters: dict[str, Any] | None = None

        model_config = ConfigDict(extra="allow")

    class ChatCompletionContentPartTextParam(BaseModel):  # pragma: no cover - simple data container
        text: str | None = None
        type: str = "text"

        model_config = ConfigDict(extra="allow")

    class ChatCompletionMessageToolCall(BaseModel):  # pragma: no cover - simple data container
        id: str | None = None
        type: str | None = None
        function: FunctionCall | None = None

        model_config = ConfigDict(extra="allow")

    class ChatCompletionMessageParam(BaseModel):  # pragma: no cover - simple data container
        content: str | None = None
        role: str | None = None

        model_config = ConfigDict(extra="allow")

    class _NotGiven:  # pragma: no cover - sentinel placeholder
        pass

    types_mod.CompletionUsage = CompletionUsage
    completion_usage_mod.CompletionUsage = CompletionUsage
    chat_message_mod.FunctionCall = FunctionCall
    chat_message_param_mod.ChatCompletionMessageParam = ChatCompletionMessageParam
    tool_call_mod.ChatCompletionMessageToolCall = ChatCompletionMessageToolCall
    chat_mod.ChatCompletionContentPartTextParam = ChatCompletionContentPartTextParam
    types_mod.FunctionDefinition = FunctionDefinition

    openai_mod.__spec__ = ModuleSpec("openai", loader=None)
    types_mod.__spec__ = ModuleSpec("openai.types", loader=None)
    completion_usage_mod.__spec__ = ModuleSpec("openai.types.completion_usage", loader=None)
    chat_mod.__spec__ = ModuleSpec("openai.types.chat", loader=None)
    chat_message_mod.__spec__ = ModuleSpec("openai.types.chat.chat_completion_message", loader=None)
    chat_message_param_mod.__spec__ = ModuleSpec("openai.types.chat.chat_completion_message_param", loader=None)
    tool_call_mod.__spec__ = ModuleSpec("openai.types.chat.chat_completion_message_tool_call", loader=None)

    openai_mod.types = types_mod
    openai_mod.NotGiven = _NotGiven
    openai_mod.NOT_GIVEN = _NotGiven()
    types_mod.completion_usage = completion_usage_mod
    types_mod.chat = chat_mod
    chat_mod.chat_completion_message = chat_message_mod
    chat_mod.chat_completion_message_tool_call = tool_call_mod
    chat_mod.chat_completion_message_param = chat_message_param_mod

    sys.modules["openai"] = openai_mod
    sys.modules["openai.types"] = types_mod
    sys.modules["openai.types.completion_usage"] = completion_usage_mod
    sys.modules["openai.types.chat"] = chat_mod
    sys.modules["openai.types.chat.chat_completion_message"] = chat_message_mod
    sys.modules["openai.types.chat.chat_completion_message_tool_call"] = tool_call_mod
    sys.modules["openai.types.chat.chat_completion_message_param"] = chat_message_param_mod


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
