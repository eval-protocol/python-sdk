from __future__ import annotations

from typing import Any, Dict, List

from tests.chinook.db import connect_database

try:
    # LangGraph import only
    from langgraph.graph import END, StateGraph
    from langchain_core.runnables import RunnableConfig
    from langchain_core.messages import BaseMessage, AIMessage
except ImportError as e:  # pragma: no cover - import-time helpful error
    # Gracefully skip this module's tests if optional deps are not installed
    import pytest

    pytest.skip(
        "Missing optional deps for LangGraph example. Install extras: 'pip install -e .[langgraph]'",
        allow_module_level=True,
    )


def build_graph() -> Any:
    """
    Build and return a minimal LangGraph app that:
    - Accepts state {"messages": List[eval_protocol.models.Message]}
    - Answers via Supabase-backed Chinook database using tests/chinook/db.py
    - Appends the assistant reply to messages
    - Returns {"messages": List[Message]}

    Model configuration (RunnableConfig) is accepted but unused here.
    """

    def call_model(state: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
        del config  # parameter accepted for signature compatibility; not used in this graph
        messages: List[BaseMessage] = state.get("messages") or []

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
        reply_text = f"Direct query result from Chinook database: {str(total)}"

        updated_messages = list(messages) + [AIMessage(content=reply_text)]
        return {"messages": updated_messages}

    graph = StateGraph(dict)
    graph.add_node("call_model", call_model)
    graph.set_entry_point("call_model")
    graph.add_edge("call_model", END)
    app = graph.compile()
    return app
