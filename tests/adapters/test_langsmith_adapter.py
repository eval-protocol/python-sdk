import types
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from eval_protocol.adapters.langsmith import LangSmithAdapter
from eval_protocol.models import Message


class FakeClient:
    def __init__(self, runs: List[Any]):
        self._runs = runs

    def list_runs(self, *, project_name: str, is_root: bool, limit: int, select: List[str]):  # type: ignore[no-untyped-def]
        return iter(self._runs[:limit])


def _msg(role: str, content: str, **kwargs: Any) -> Dict[str, Any]:
    m = {"role": role, "content": content}
    m.update(kwargs)
    return m


def test_outputs_messages_preferred_and_dedup_user():
    # outputs.messages exists with duplicate consecutive user messages
    runs = [
        SimpleNamespace(
            id="r1",
            inputs={"messages": [_msg("user", "hi")]},
            outputs={
                "messages": [
                    _msg("user", "hi"),
                    _msg("user", "hi"),  # duplicate
                    _msg("assistant", "hello"),
                ]
            },
        )
    ]
    adapter = LangSmithAdapter(client=FakeClient(runs))  # pyright: ignore[reportArgumentType]
    rows = adapter.get_evaluation_rows(project_name="p", limit=10)
    assert len(rows) == 1
    msgs = rows[0].messages
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[0].content == "hi"
    assert msgs[1].content == "hello"


def test_inputs_variants_prompt_user_input_input():
    runs = [
        SimpleNamespace(id="p1", inputs={"prompt": "A"}, outputs={"content": "OA"}),
        SimpleNamespace(id="p2", inputs={"user_input": "B"}, outputs={"result": "OB"}),
        SimpleNamespace(id="p3", inputs={"input": "C"}, outputs={"answer": "OC"}),
        SimpleNamespace(id="p4", inputs="D", outputs="OD"),
    ]
    adapter = LangSmithAdapter(client=FakeClient(runs))  # pyright: ignore[reportArgumentType]
    rows = adapter.get_evaluation_rows(project_name="p", limit=10)
    texts = [[(m.role, m.content) for m in r.messages] for r in rows]
    assert ("user", "A") in texts[0]
    assert ("assistant", "OA") in texts[0]
    assert ("user", "B") in texts[1]
    assert ("assistant", "OB") in texts[1]
    assert ("user", "C") in texts[2]
    assert ("assistant", "OC") in texts[2]
    assert ("user", "D") in texts[3]
    assert ("assistant", "OD") in texts[3]


def test_outputs_variants_and_list_payloads():
    runs = [
        SimpleNamespace(id="o1", inputs=[], outputs={"output": "X"}),
        SimpleNamespace(id="o2", inputs=[_msg("user", "U")], outputs=[_msg("assistant", "V")]),
    ]
    adapter = LangSmithAdapter(client=FakeClient(runs))  # pyright: ignore[reportArgumentType]
    rows = adapter.get_evaluation_rows(project_name="p", limit=10)
    msgs1 = rows[0].messages
    assert any(m.role == "assistant" and m.content == "X" for m in msgs1)
    msgs2 = rows[1].messages
    assert any(m.role == "user" and m.content == "U" for m in msgs2)
    assert any(m.role == "assistant" and m.content == "V" for m in msgs2)


def test_tool_calls_and_tool_role_preserved():
    tool_args = '{"a":2,"b":3}'
    assistant_with_tool = _msg(
        "assistant",
        "Tool Calls:\ncalculator_add\n" + tool_args,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "calculator_add", "arguments": tool_args},
            }
        ],
    )
    tool_msg = {"role": "tool", "name": "calculator_add", "tool_call_id": "call_1", "content": "5"}
    runs = [
        SimpleNamespace(
            id="t1",
            inputs={"messages": [_msg("user", "Add 2 and 3")]},
            outputs={
                "messages": [
                    _msg("user", "Add 2 and 3"),
                    assistant_with_tool,
                    tool_msg,
                    _msg("assistant", "The result is 5."),
                ]
            },
        )
    ]
    adapter = LangSmithAdapter(client=FakeClient(runs))  # pyright: ignore[reportArgumentType]
    rows = adapter.get_evaluation_rows(project_name="p", limit=10)
    msgs = rows[0].messages
    # Ensure tool role present
    assert any(m.role == "tool" and str(m.content or "").strip() == "5" for m in msgs)
    # Ensure assistant with tool_calls preserved
    assistants = [m for m in msgs if m.role == "assistant" and m.tool_calls]
    assert len(assistants) >= 1
    assert assistants[0].tool_calls is not None
    tc = assistants[0].tool_calls[0]
    # tool_calls may be provider-native objects; normalize via getattr first
    fname = None
    if hasattr(tc, "function"):
        fn = getattr(tc, "function")
        fname = getattr(fn, "name", None)
    elif isinstance(tc, dict):
        fname = tc.get("function", {}).get("name")
    assert fname == "calculator_add"


def test_system_prompt_first_and_multiple_user_allowed():
    runs = [
        SimpleNamespace(
            id="s1",
            inputs={
                "messages": [
                    _msg("system", "You are helpful"),
                    _msg("user", "hi"),
                    _msg("user", "hi again"),
                ]
            },
            outputs={"content": "hello there"},
        )
    ]
    adapter = LangSmithAdapter(client=FakeClient(runs))  # pyright: ignore[reportArgumentType]
    rows = adapter.get_evaluation_rows(project_name="p", limit=10)
    msgs = rows[0].messages
    roles = [m.role for m in msgs]
    assert roles[0] == "system"
    # both user messages retained (not deduped since content differs)
    assert roles[1] == "user" and roles[2] == "user"
    assert roles[-1] == "assistant"


def test_parallel_tool_calls_normalized():
    # Two tool calls in a single assistant message
    tool_args1 = '{"a":2,"b":3}'
    tool_args2 = '{"a":4,"b":5}'
    assistant_with_tools = _msg(
        "assistant",
        "Two calls",
        tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "calculator_add", "arguments": tool_args1}},
            {"id": "c2", "type": "function", "function": {"name": "calculator_add", "arguments": tool_args2}},
        ],
    )
    runs = [
        SimpleNamespace(
            id="pt1",
            inputs={"messages": [_msg("user", "sum two pairs")]},
            outputs={"messages": [assistant_with_tools]},
        ),
    ]
    adapter = LangSmithAdapter(client=FakeClient(runs))  # pyright: ignore[reportArgumentType]
    rows = adapter.get_evaluation_rows(project_name="p", limit=10)
    msgs = rows[0].messages
    assistants = [m for m in msgs if m.role == "assistant" and m.tool_calls]
    assert len(assistants) == 1
    tcs = assistants[0].tool_calls
    assert isinstance(tcs, list) and len(tcs) == 2
    names = [getattr(tc.function, "name", None) if hasattr(tc, "function") else None for tc in tcs]
    assert names == ["calculator_add", "calculator_add"]
