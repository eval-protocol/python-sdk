from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.default_langchain_rollout_processor import LangGraphRolloutProcessor


class DummyLCMessage:
    def __init__(self, message_type: str, content: str):  # noqa: A002
        self.type = message_type
        self.content = content


class DummyGraph:
    def __init__(self, out_messages: List[Any]):
        self._out_messages = out_messages

    async def ainvoke(self, payload: Dict[str, Any], **_: Any):
        # Echo back the provided messages plus our out_messages
        return {"messages": list(payload.get("messages") or []) + list(self._out_messages)}


def _make_processor_with_defaults(out_messages: List[Any]) -> LangGraphRolloutProcessor:
    def graph_factory(_: Dict[str, Any]):
        return DummyGraph(out_messages)

    return LangGraphRolloutProcessor(graph_factory=graph_factory)


@pytest.mark.asyncio
async def test_apply_result_preserves_user_role_and_appends_assistant_from_lc():
    # Arrange: EP user message in, LC assistant out
    row = EvaluationRow(messages=[Message(role="user", content="hi")])
    lc_assistant = DummyLCMessage(message_type="ai", content="hello")
    processor = _make_processor_with_defaults([lc_assistant])

    # Act
    tasks = processor(
        [row],
        type(
            "Cfg",
            (),
            {
                "completion_params": {},
                "semaphore": asyncio.Semaphore(10),
                "mcp_config_path": "",
                "logger": None,
                "server_script_path": None,
                "steps": 1,
                "kwargs": {},
                "exception_handler_config": None,
            },
        )(),
    )
    result_row = await asyncio.gather(*tasks)
    out = result_row[0]

    # Assert
    assert out.messages[0].role == "user"
    assert out.messages[-1].role == "assistant"
    assert out.messages[-1].content == "hello"


@pytest.mark.asyncio
async def test_apply_result_handles_dict_messages_with_missing_role():
    row = EvaluationRow(messages=[Message(role="user", content="Q")])
    dict_msg = {"content": "A"}  # no role provided
    processor = _make_processor_with_defaults([dict_msg])

    tasks = processor(
        [row],
        type(
            "Cfg",
            (),
            {
                "completion_params": {},
                "semaphore": asyncio.Semaphore(10),
                "mcp_config_path": "",
                "logger": None,
                "server_script_path": None,
                "steps": 1,
                "kwargs": {},
                "exception_handler_config": None,
            },
        )(),
    )
    out = (await asyncio.gather(*tasks))[0]

    assert out.messages[0].role == "user"
    assert out.messages[-1].role == "assistant"
    assert out.messages[-1].content == "A"


@pytest.mark.asyncio
async def test_to_input_converts_ep_messages_to_lc_via_adapter(monkeypatch):
    # Arrange
    ep_row = EvaluationRow(messages=[Message(role="user", content="Hello")])
    called = {"ok": False}

    def fake_to_lc(messages):
        called["ok"] = True
        return [DummyLCMessage(message_type="human", content=messages[0].content)]

    # Patch the adapter function at its source module, since the processor imports it inside the function
    import eval_protocol.adapters.langchain as lc_adapter

    monkeypatch.setattr(lc_adapter, "serialize_ep_messages_to_lc", fake_to_lc, raising=True)

    # Dummy graph that returns what it receives
    class EchoGraph:
        async def ainvoke(self, payload, **_):
            # Ensure our adapter-produced messages flow through
            return payload

    processor = LangGraphRolloutProcessor(graph_factory=lambda _: EchoGraph())

    # Act
    tasks = processor(
        [ep_row],
        type(
            "Cfg",
            (),
            {
                "completion_params": {},
                "semaphore": asyncio.Semaphore(10),
                "mcp_config_path": "",
                "logger": None,
                "server_script_path": None,
                "steps": 1,
                "kwargs": {},
                "exception_handler_config": None,
            },
        )(),
    )
    await asyncio.gather(*tasks)

    # Assert that adapter was used
    assert called["ok"] is True


@pytest.mark.asyncio
async def test_apply_result_copies_tool_calls_from_lc_ai_and_toolmessage():
    from langchain_core.messages import AIMessage, ToolMessage

    # Arrange: EP user message in, LC assistant with tool_calls + LC tool message out
    row = EvaluationRow(messages=[Message(role="user", content="count tracks")])
    tool_call_id = "call_1"
    # Use LangChain-native tool_call schema (name/args) so AIMessage validates
    ai_with_tool = AIMessage(
        content="I'll call the tool.",
        tool_calls=[
            {
                "id": tool_call_id,
                "name": "count_tracks",
                "args": {},
            }
        ],
    )
    tool_msg = ToolMessage(content="3503", name="count_tracks", tool_call_id=tool_call_id, status="success")
    processor = _make_processor_with_defaults([ai_with_tool, tool_msg])

    # Act
    tasks = processor(
        [row],
        type(
            "Cfg",
            (),
            {
                "completion_params": {},
                "semaphore": asyncio.Semaphore(10),
                "mcp_config_path": "",
                "logger": None,
                "server_script_path": None,
                "steps": 1,
                "kwargs": {},
                "exception_handler_config": None,
            },
        )(),
    )
    out = (await asyncio.gather(*tasks))[0]

    # Assert: assistant message has tool_calls, and tool message is present
    assistants = [m for m in out.messages if m.role == "assistant"]
    tools = [m for m in out.messages if m.role == "tool"]
    assert assistants, "No assistant messages found"
    assert tools, "No tool messages found"
    assert assistants[-1].tool_calls is not None and len(assistants[-1].tool_calls) == 1
    assert assistants[-1].tool_calls[0].id, "tool_call id should be present"
    assert tools[-1].content and "3503" in (tools[-1].content or "")


@pytest.mark.asyncio
async def test_apply_result_copies_tool_calls_from_additional_kwargs():
    from langchain_core.messages import AIMessage, ToolMessage

    # Arrange: tool_calls provided via additional_kwargs instead of attribute
    row = EvaluationRow(messages=[Message(role="user", content="count tracks")])
    tool_call_id = "call_2"
    ai_with_tool = AIMessage(
        content="I'll call the tool.",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "name": "count_tracks",
                    "args": {},
                }
            ]
        },
    )
    tool_msg = ToolMessage(content="3503", name="count_tracks", tool_call_id=tool_call_id, status="success")
    processor = _make_processor_with_defaults([ai_with_tool, tool_msg])

    # Act
    tasks = processor(
        [row],
        type(
            "Cfg",
            (),
            {
                "completion_params": {},
                "semaphore": asyncio.Semaphore(10),
                "mcp_config_path": "",
                "logger": None,
                "server_script_path": None,
                "steps": 1,
                "kwargs": {},
                "exception_handler_config": None,
            },
        )(),
    )
    out = (await asyncio.gather(*tasks))[0]

    # Assert
    assistants = [m for m in out.messages if m.role == "assistant"]
    tools = [m for m in out.messages if m.role == "tool"]
    assert assistants and assistants[-1].tool_calls is not None
    assert any(tc.id for tc in assistants[-1].tool_calls), "Expected tool_call with id"
    assert tools and "3503" in (tools[-1].content or "")
