"""Unit tests for :class:`FireworksTrainingRolloutProcessor`.

These tests stub out :class:`FireworksV1CompletionsClient` so no
network calls or tokenizers are required. They cover the contract that
managed Fireworks RFT depends on:

* per-completion ``prompt_ids`` / ``completion_ids`` / ``inference_logprobs``
  land on ``EvaluationRow.execution_metadata.extra``
* the first completion is appended as an ``assistant`` message so existing
  evaluators keep working
* ``n`` completions produce ``n``-length lists
* ``finish_reason == 'length'`` is surfaced as ``truncated[i] = True``
* trailing ``assistant`` messages are dropped before sampling by default
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import mock

import pytest

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import FireworksTrainingRolloutProcessor


class _StubConfig:
    """Minimal stand-in for :class:`RolloutProcessorConfig`."""

    def __init__(self, *, n: int = 2, max_tokens: int = 32, model: str = "accounts/fireworks/models/qwen3-8b"):
        self.completion_params: dict[str, Any] = {
            "model": model,
            "temperature": 1.0,
            "max_tokens": max_tokens,
            "n": n,
        }
        self.semaphore = asyncio.Semaphore(4)


class _FakeCompletionsClient:
    """Mimics ``FireworksV1CompletionsClient`` just enough for the processor.

    Each call to ``create_completion_from_prompt_ids`` returns a distinct
    completion so we can assert per-completion indexing.
    """

    def __init__(self, completions: list[dict[str, Any]], prompt_token_ids: list[int] | None = None) -> None:
        self._completions = list(completions)
        self._prompt_token_ids = prompt_token_ids or [1, 2, 3]
        self._call_count = 0
        self.close_calls = 0

    def build_prompt_token_ids(self, *, messages: list[dict[str, Any]], tools: Any = None) -> list[int]:
        return list(self._prompt_token_ids)

    async def create_completion_from_prompt_ids(
        self, *, prompt_token_ids: list[int], tools: Any = None
    ) -> dict[str, Any]:
        idx = self._call_count
        self._call_count += 1
        # Cycle through the provided completions so repeated calls return
        # distinct payloads.
        return self._completions[idx % len(self._completions)]

    async def close(self) -> None:
        self.close_calls += 1


def _make_row(*, trailing_assistant: bool = False) -> EvaluationRow:
    messages = [Message(role="user", content="What is 2+2?")]
    if trailing_assistant:
        messages.append(Message(role="assistant", content="old cached answer"))
    return EvaluationRow(messages=messages)


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, client: _FakeCompletionsClient) -> None:
    """Patch the client constructor so the processor uses our fake."""
    import eval_protocol.integrations.fireworks_v1_completions_client as mod

    monkeypatch.setattr(mod, "FireworksV1CompletionsClient", lambda **_kwargs: client)


@pytest.mark.asyncio
async def test_produces_per_completion_token_ids_and_logprobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """With n=2, the processor returns a row whose ``extra`` has two-entry lists."""
    completions = [
        {
            "choices": [{"message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}],
            "prompt_ids": [10, 11, 12],
            "completion_ids": [40, 41],
            "completion_logprobs": [-0.1, -0.2],
            "finish_reason": "stop",
        },
        {
            "choices": [{"message": {"role": "assistant", "content": "four"}, "finish_reason": "length"}],
            "prompt_ids": [10, 11, 12],
            "completion_ids": [42, 43, 44],
            "completion_logprobs": [-0.3, -0.4, -0.5],
            "finish_reason": "length",
        },
    ]
    client = _FakeCompletionsClient(completions=completions, prompt_token_ids=[10, 11, 12])
    _install_fake_client(monkeypatch, client)

    processor = FireworksTrainingRolloutProcessor()
    processor.setup()

    tasks = processor([_make_row()], _StubConfig(n=2))
    result = await tasks[0]

    extra = result.execution_metadata.extra
    assert extra is not None
    assert extra["prompt_ids"] == [10, 11, 12]
    assert extra["completion_ids"] == [[40, 41], [42, 43, 44]]
    assert extra["inference_logprobs"] == [[-0.1, -0.2], [-0.3, -0.4, -0.5]]
    assert extra["completions_text"] == ["4", "four"]
    assert extra["truncated"] == [False, True]
    assert extra["finish_reasons"] == ["stop", "length"]
    # The first completion must be exposed as the assistant message so
    # evaluators that call ``last_assistant_message()`` still score.
    assert result.messages[-1].role == "assistant"
    assert result.messages[-1].content == "4"
    # Usage should aggregate across completions.
    assert result.execution_metadata.usage is not None
    assert result.execution_metadata.usage.prompt_tokens == 3
    assert result.execution_metadata.usage.completion_tokens == 5


@pytest.mark.asyncio
async def test_n_equals_one_single_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """n=1 still produces the list-of-lists shape — just with length 1."""
    completions = [
        {
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "prompt_ids": [1, 2],
            "completion_ids": [9],
            "completion_logprobs": [-0.01],
            "finish_reason": "stop",
        }
    ]
    client = _FakeCompletionsClient(completions=completions, prompt_token_ids=[1, 2])
    _install_fake_client(monkeypatch, client)

    processor = FireworksTrainingRolloutProcessor()
    processor.setup()

    tasks = processor([_make_row()], _StubConfig(n=1))
    result = await tasks[0]

    extra = result.execution_metadata.extra
    assert extra["completion_ids"] == [[9]]
    assert extra["inference_logprobs"] == [[-0.01]]
    assert extra["truncated"] == [False]


@pytest.mark.asyncio
async def test_drops_trailing_assistant_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trailing ``assistant`` messages are dropped before the prompt is built."""
    captured_messages: dict[str, Any] = {}

    class _CaptureClient(_FakeCompletionsClient):
        def build_prompt_token_ids(self, *, messages: list[dict[str, Any]], tools: Any = None) -> list[int]:
            captured_messages["messages"] = messages
            return [1, 2]

    completions = [
        {
            "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
            "prompt_ids": [1, 2],
            "completion_ids": [3],
            "completion_logprobs": [-0.1],
            "finish_reason": "stop",
        }
    ]
    client = _CaptureClient(completions=completions)
    _install_fake_client(monkeypatch, client)

    processor = FireworksTrainingRolloutProcessor()
    processor.setup()

    tasks = processor([_make_row(trailing_assistant=True)], _StubConfig(n=1))
    await tasks[0]

    sent = captured_messages["messages"]
    assert [m["role"] for m in sent] == ["user"], "Trailing assistant should have been dropped"


@pytest.mark.asyncio
async def test_keeps_trailing_assistant_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicitly disable the drop to support continuations."""
    captured_messages: dict[str, Any] = {}

    class _CaptureClient(_FakeCompletionsClient):
        def build_prompt_token_ids(self, *, messages: list[dict[str, Any]], tools: Any = None) -> list[int]:
            captured_messages["messages"] = messages
            return [1, 2]

    completions = [
        {
            "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
            "prompt_ids": [1, 2],
            "completion_ids": [3],
            "completion_logprobs": [-0.1],
            "finish_reason": "stop",
        }
    ]
    client = _CaptureClient(completions=completions)
    _install_fake_client(monkeypatch, client)

    processor = FireworksTrainingRolloutProcessor(drop_trailing_assistant_messages=False)
    processor.setup()

    tasks = processor([_make_row(trailing_assistant=True)], _StubConfig(n=1))
    await tasks[0]

    sent = captured_messages["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_missing_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """``completion_params`` must carry a ``model`` id."""
    _install_fake_client(monkeypatch, _FakeCompletionsClient(completions=[]))
    processor = FireworksTrainingRolloutProcessor()
    processor.setup()

    config = _StubConfig()
    del config.completion_params["model"]

    with pytest.raises(ValueError, match="completion_params.model"):
        tasks = processor([_make_row()], config)
        await tasks[0]


@pytest.mark.asyncio
async def test_invalid_n_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """``n < 1`` is rejected."""
    _install_fake_client(monkeypatch, _FakeCompletionsClient(completions=[]))
    processor = FireworksTrainingRolloutProcessor()
    processor.setup()

    config = _StubConfig()
    config.completion_params["n"] = 0

    with pytest.raises(ValueError, match="n must be >= 1"):
        tasks = processor([_make_row()], config)
        await tasks[0]


@pytest.mark.asyncio
async def test_acleanup_closes_cached_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every cached client should be ``.close()``-ed on cleanup."""
    completions = [
        {
            "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
            "prompt_ids": [1],
            "completion_ids": [2],
            "completion_logprobs": [0.0],
            "finish_reason": "stop",
        }
    ]
    client = _FakeCompletionsClient(completions=completions)
    _install_fake_client(monkeypatch, client)

    processor = FireworksTrainingRolloutProcessor()
    processor.setup()
    tasks = processor([_make_row()], _StubConfig(n=1))
    await tasks[0]
    assert client.close_calls == 0

    await processor.acleanup()
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_preserves_existing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-existing ``execution_metadata.extra`` keys must not be clobbered."""
    completions = [
        {
            "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
            "prompt_ids": [1],
            "completion_ids": [2],
            "completion_logprobs": [0.0],
            "finish_reason": "stop",
        }
    ]
    client = _FakeCompletionsClient(completions=completions)
    _install_fake_client(monkeypatch, client)

    processor = FireworksTrainingRolloutProcessor()
    processor.setup()

    row = _make_row()
    row.execution_metadata.extra = {"my_custom_field": "hello"}
    tasks = processor([row], _StubConfig(n=1))
    result = await tasks[0]

    assert result.execution_metadata.extra["my_custom_field"] == "hello"
    assert result.execution_metadata.extra["prompt_ids"] == [1]
