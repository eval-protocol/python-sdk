import sys
import types
from pathlib import Path
from typing import cast

import pytest

# The evaluation modules depend on optional third-party packages that aren't
# available in the execution environment for these unit tests. To exercise the
# real implementation of ``load_and_prepare_rows`` without installing those
# packages, we register lightweight stubs in ``sys.modules`` that provide the
# minimal interfaces required during import.

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "eval_protocol"
sys.modules.pop("eval_protocol", None)
sys.modules.pop("eval_protocol.pytest", None)

eval_protocol_pkg = types.ModuleType("eval_protocol")
eval_protocol_pkg.__path__ = [str(PACKAGE_ROOT)]  # type: ignore[attr-defined]
sys.modules["eval_protocol"] = eval_protocol_pkg

pytest_pkg = types.ModuleType("eval_protocol.pytest")
pytest_pkg.__path__ = [str(PACKAGE_ROOT / "pytest")]  # type: ignore[attr-defined]
sys.modules["eval_protocol.pytest"] = pytest_pkg
setattr(eval_protocol_pkg, "pytest", pytest_pkg)

if "loguru" not in sys.modules:
    loguru_module = types.ModuleType("loguru")

    class _DummyLogger:
        def __getattr__(self, _name):  # pragma: no cover - dynamic fallback
            def _noop(*_args, **_kwargs):
                return None

            return _noop

    loguru_module.logger = _DummyLogger()  # type: ignore[attr-defined]
    sys.modules["loguru"] = loguru_module

if "toml" not in sys.modules:
    toml_module = types.ModuleType("toml")

    def _noop_load(*_args, **_kwargs):  # pragma: no cover - helper for stubbing
        return {}

    def _noop_dump(*_args, **_kwargs):  # pragma: no cover - helper for stubbing
        return None

    toml_module.load = _noop_load  # type: ignore[attr-defined]
    toml_module.dump = _noop_dump  # type: ignore[attr-defined]
    sys.modules["toml"] = toml_module

if "addict" not in sys.modules:
    addict_module = types.ModuleType("addict")

    class _AddictDict(dict):  # pragma: no cover - simple stub
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

        def __setattr__(self, key, value):
            self[key] = value

    addict_module.Dict = _AddictDict  # type: ignore[attr-defined]
    sys.modules["addict"] = addict_module

if "eval_protocol.mcp_env" not in sys.modules:
    mcp_env_module = types.ModuleType("eval_protocol.mcp_env")

    class _DummyPolicy:  # pragma: no cover - stub placeholder
        pass

    def _noop(*_args, **_kwargs):
        return None

    mcp_env_module.AnthropicPolicy = _DummyPolicy  # type: ignore[attr-defined]
    mcp_env_module.FireworksPolicy = _DummyPolicy  # type: ignore[attr-defined]
    mcp_env_module.LiteLLMPolicy = _DummyPolicy  # type: ignore[attr-defined]
    mcp_env_module.OpenAIPolicy = _DummyPolicy  # type: ignore[attr-defined]
    mcp_env_module.make = _noop  # type: ignore[attr-defined]
    mcp_env_module.rollout = _noop  # type: ignore[attr-defined]
    mcp_env_module.test_mcp = _noop  # type: ignore[attr-defined]
    sys.modules["eval_protocol.mcp_env"] = mcp_env_module

if "eval_protocol.mcp" not in sys.modules:
    sys.modules["eval_protocol.mcp"] = types.ModuleType("eval_protocol.mcp")

if "eval_protocol.rewards" not in sys.modules:
    sys.modules["eval_protocol.rewards"] = types.ModuleType("eval_protocol.rewards")

if "eval_protocol.dataset_logger" not in sys.modules:
    dataset_logger_module = types.ModuleType("eval_protocol.dataset_logger")

    class _StubDatasetLogger:  # pragma: no cover - stub placeholder
        def log(self, *_args, **_kwargs):
            return None

    dataset_logger_module.default_logger = _StubDatasetLogger()  # type: ignore[attr-defined]
    sys.modules["eval_protocol.dataset_logger"] = dataset_logger_module

if "eval_protocol.dataset_logger.dataset_logger" not in sys.modules:
    dataset_logger_pkg = types.ModuleType("eval_protocol.dataset_logger.dataset_logger")
    dataset_logger_pkg.DatasetLogger = _StubDatasetLogger  # type: ignore[attr-defined]
    sys.modules["eval_protocol.dataset_logger.dataset_logger"] = dataset_logger_pkg

if "backoff" not in sys.modules:
    backoff_module = types.ModuleType("backoff")

    def _noop_decorator(*_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator

    backoff_module.on_exception = _noop_decorator  # type: ignore[attr-defined]
    backoff_module.expo = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["backoff"] = backoff_module

if "litellm" not in sys.modules:
    litellm_module = types.ModuleType("litellm")
    cost_calculator_module = types.ModuleType("litellm.cost_calculator")
    cost_calculator_module.cost_per_token = lambda *args, **kwargs: 0.0  # type: ignore[attr-defined]
    sys.modules["litellm"] = litellm_module
    sys.modules["litellm.cost_calculator"] = cost_calculator_module

if "tqdm" not in sys.modules:
    tqdm_module = types.ModuleType("tqdm")

    def _noop_tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else []

    tqdm_module.tqdm = _noop_tqdm  # type: ignore[attr-defined]
    sys.modules["tqdm"] = tqdm_module

if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")
    openai_types_module = types.ModuleType("openai.types")
    openai_chat_module = types.ModuleType("openai.types.chat")
    openai_chat_completion_module = types.ModuleType("openai.types.chat.chat_completion_message")
    openai_chat_tool_module = types.ModuleType("openai.types.chat.chat_completion_message_tool_call")

    class _NotGiven:  # pragma: no cover - stub placeholder
        pass

    openai_module.NOT_GIVEN = _NotGiven()  # type: ignore[attr-defined]
    openai_module.NotGiven = _NotGiven  # type: ignore[attr-defined]

    openai_types_module.CompletionUsage = object  # type: ignore[attr-defined]
    openai_chat_completion_module.FunctionCall = object  # type: ignore[attr-defined]
    openai_chat_tool_module.ChatCompletionMessageToolCall = object  # type: ignore[attr-defined]

    openai_types_module.chat = openai_chat_module  # type: ignore[attr-defined]
    openai_chat_module.chat_completion_message = openai_chat_completion_module  # type: ignore[attr-defined]
    openai_chat_module.chat_completion_message_tool_call = openai_chat_tool_module  # type: ignore[attr-defined]
    openai_module.types = openai_types_module  # type: ignore[attr-defined]

    sys.modules["openai"] = openai_module
    sys.modules["openai.types"] = openai_types_module
    sys.modules["openai.types.chat"] = openai_chat_module
    sys.modules["openai.types.chat.chat_completion_message"] = openai_chat_completion_module
    sys.modules["openai.types.chat.chat_completion_message_tool_call"] = openai_chat_tool_module

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
