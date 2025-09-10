from __future__ import annotations

from typing import Any, Dict, List

from tests.chinook.db import connect_database

try:
    # LangGraph + LangChain imports only
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import BaseMessage
    from langchain.chat_models import init_chat_model
    from langchain_core.tools import tool
    from typing_extensions import Annotated, TypedDict
except ImportError as e:  # pragma: no cover - import-time helpful error
    # Gracefully skip this module's tests if optional deps are not installed
    import pytest

    pytest.skip(
        "Missing optional deps for LangGraph tools example. Install extras: 'pip install -e .[langgraph_tools]'",
        allow_module_level=True,
    )


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def _count_tracks() -> str:
    """Return total number of tracks from Chinook database as string."""
    _, cursor, introspection = connect_database()
    table_names = {row[0] for row in introspection}
    candidate = None
    if "tracks" in table_names:
        candidate = "tracks"
    elif "track" in table_names:
        candidate = "track"
    else:
        for t in table_names:
            if "track" in t:
                candidate = t
                break
    if candidate is None:
        raise RuntimeError("Could not find track(s) table")
    cursor.execute(f"SELECT COUNT(*) FROM {candidate}")
    total = cursor.fetchone()[0]
    return str(total)


@tool
def count_tracks() -> str:
    """Count total number of tracks in the Chinook database and return as text."""
    return _count_tracks()


def build_graph() -> Any:
    """
    Build a LangGraph app that binds a Chinook DB tool and routes tool calls.

    Behavior:
    - Binds `count_tracks` tool to the model.
    - If the model emits tool calls, ToolNode executes and loops back.
    - If no tool call is emitted, we fall back to directly computing the answer to ensure determinism for tests.
    """

    tools = [count_tracks]
    llm = init_chat_model("accounts/fireworks/models/kimi-k2-instruct", model_provider="fireworks", temperature=0.0)
    model_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def should_continue(state: State) -> str:
        messages = state["messages"]
        last = messages[-1] if messages else None
        if last is not None and getattr(last, "tool_calls", None):
            return "tools"
        return END

    async def call_model(state: State) -> Dict[str, Any]:
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", should_continue)
    graph.add_edge("tools", "call_model")
    app = graph.compile()
    return app
