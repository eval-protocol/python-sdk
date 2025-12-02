import asyncio
from types import SimpleNamespace

import pytest

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import SingleTurnRolloutProcessor


class _DummyConfig:
    def __init__(self):
        self.completion_params = {"model": "fake-model", "temperature": 0}
        self.semaphore = asyncio.Semaphore(10)


@pytest.mark.asyncio
async def test_single_turn_drops_trailing_assistant_by_default(monkeypatch):
    # Arrange dataset row with trailing assistant message
    row = EvaluationRow(
        messages=[
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="Old response"),
        ]
    )

    # Capture the messages payload passed to the LLM call
    captured = {}

    # Patch module-level imports in the processor module
    import eval_protocol.pytest.default_single_turn_rollout_process as mod

    class StubChoices:
        pass

    class StubModelResponse:
        def __init__(self, text: str):
            self.choices = [StubChoices()]
            # Emulate OpenAI-like response.message fields
            self.choices[0].message = SimpleNamespace(content=text, tool_calls=None)
            # Minimal usage payload
            self.usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    async def fake_acompletion(**kwargs):
        # Verify that trailing assistant was dropped before sending
        msgs = kwargs.get("messages", [])
        assert msgs, "Expected non-empty messages payload"
        captured["messages"] = msgs
        assert msgs[-1]["role"] != "assistant", "Trailing assistant should be dropped by default"
        return StubModelResponse(text="4")

    # Monkeypatch the processor module's symbols to avoid dependency on litellm types
    monkeypatch.setattr(mod, "ModelResponse", StubModelResponse, raising=True)
    monkeypatch.setattr(mod, "Choices", StubChoices, raising=True)
    monkeypatch.setattr(mod, "acompletion", fake_acompletion, raising=True)

    processor = SingleTurnRolloutProcessor()
    config = _DummyConfig()

    # Act
    tasks = processor([row], config)
    out = await tasks[0]

    # Assert: request trimmed the trailing assistant
    sent_msgs = captured["messages"]
    assert len(sent_msgs) == 1
    assert sent_msgs[0]["role"] == "user"
    assert out.messages[-1].role == "assistant"
    assert out.messages[-1].content == "4"
    # Ensure previous trailing assistant was not duplicated
    assert [m.role for m in out.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_single_turn_keeps_trailing_assistant_when_disabled(monkeypatch):
    # Arrange dataset row with trailing assistant message
    row = EvaluationRow(
        messages=[
            Message(role="user", content="Say hi"),
            Message(role="assistant", content="Hi!"),
        ]
    )

    captured = {}

    import eval_protocol.pytest.default_single_turn_rollout_process as mod

    class StubChoices:
        pass

    class StubModelResponse:
        def __init__(self, text: str):
            self.choices = [StubChoices()]
            self.choices[0].message = SimpleNamespace(content=text, tool_calls=None)
            self.usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    async def fake_acompletion(**kwargs):
        msgs = kwargs.get("messages", [])
        captured["messages"] = msgs
        # With opt-out, trailing assistant is preserved
        assert msgs[-1]["role"] == "assistant"
        return StubModelResponse(text="Hello again")

    monkeypatch.setattr(mod, "ModelResponse", StubModelResponse, raising=True)
    monkeypatch.setattr(mod, "Choices", StubChoices, raising=True)
    monkeypatch.setattr(mod, "acompletion", fake_acompletion, raising=True)

    processor = SingleTurnRolloutProcessor(drop_trailing_assistant_messages=False)
    config = _DummyConfig()

    # Act
    tasks = processor([row], config)
    out = await tasks[0]

    # Assert: both original messages plus new assistant
    sent_msgs = captured["messages"]
    assert [m["role"] for m in sent_msgs] == ["user", "assistant"]
    assert [m.role for m in out.messages] == ["user", "assistant", "assistant"]
    assert out.messages[-1].content == "Hello again"


@pytest.mark.asyncio
async def test_single_turn_handles_missing_usage_block(monkeypatch):
    row = EvaluationRow(messages=[Message(role="user", content="Describe the picture")])

    import eval_protocol.pytest.default_single_turn_rollout_process as mod

    class StubChoices:
        pass

    class StubModelResponse:
        def __init__(self, text: str):
            self.choices = [StubChoices()]
            self.choices[0].message = SimpleNamespace(content=text, tool_calls=None)
            self.usage = None

    async def fake_acompletion(**kwargs):
        return StubModelResponse(text="It looks like creme brulee")

    class StubLogger:
        def __init__(self):
            self.logged = []

        def log(self, row):
            self.logged.append(row)

        def read(self, rollout_id=None):
            return list(self.logged)

    stub_logger = StubLogger()

    monkeypatch.setattr(mod, "ModelResponse", StubModelResponse, raising=True)
    monkeypatch.setattr(mod, "Choices", StubChoices, raising=True)
    monkeypatch.setattr(mod, "acompletion", fake_acompletion, raising=True)
    monkeypatch.setattr(mod, "default_logger", stub_logger, raising=False)

    processor = SingleTurnRolloutProcessor()
    config = _DummyConfig()

    tasks = processor([row], config)
    out = await tasks[0]

    assert [m.role for m in out.messages] == ["user", "assistant"]
    assert out.messages[-1].content == "It looks like creme brulee"
    # Usage should remain unset when the provider omits it
    assert out.execution_metadata.usage is None
