"""Quick script to send a few throwaway traces to LangSmith.

Usage:
  export LANGSMITH_API_KEY=...  # required
  export LANGSMITH_TRACING=true  # recommended
  python python-sdk/examples/langsmith/dump_traces_langsmith.py

Notes:
- This does not require any external model keys. It logs a few synthetic
  traced function calls, and optionally a tiny LangGraph flow if available.
"""

import asyncio
import os
from typing import Any, Dict, List
import importlib


def _ensure_env_defaults() -> None:
    # Prefer modern env vars; fall back maintained for compatibility.
    if os.environ.get("LANGSMITH_TRACING") is None:
        os.environ["LANGSMITH_TRACING"] = "true"
    # Project name helps organize traces in the LangSmith UI
    os.environ.setdefault("LANGCHAIN_PROJECT", "ep-langgraph-examples")


def _log_synthetic_traces() -> None:
    traceable = None
    try:
        mod = importlib.import_module("langsmith")
        traceable = getattr(mod, "traceable", None)
    except ImportError:
        pass
    if traceable is None:
        print("LangSmith not installed; skipping @traceable demo. `pip install langsmith`.")
        return

    @traceable(name="toy_pipeline")
    def toy_pipeline(user_input: str) -> Dict[str, Any]:
        reversed_text = user_input[::-1]
        upper_text = reversed_text.upper()
        return {"result": upper_text, "len": len(upper_text)}

    print("Emitting synthetic traces via @traceable...")
    toy_pipeline("hello langsmith")
    toy_pipeline("trace number two")
    toy_pipeline("final short run")


async def _maybe_run_tiny_langgraph() -> None:
    """Optionally run a tiny LangGraph flow to log a couple of runs.

    This avoids any external LLM providers by using a pure-Python node.
    """
    try:
        graph_mod = importlib.import_module("langgraph.graph")
        msg_mod = importlib.import_module("langgraph.graph.message")
        lc_msgs = importlib.import_module("langchain_core.messages")
        te_mod = importlib.import_module("typing_extensions")
    except ImportError:
        print("LangGraph/LangChain not installed; skipping tiny graph demo. `pip install langgraph langchain-core`.")
        return

    END = getattr(graph_mod, "END")
    StateGraph = getattr(graph_mod, "StateGraph")
    add_messages = getattr(msg_mod, "add_messages")
    AIMessage = getattr(lc_msgs, "AIMessage")
    BaseMessage = getattr(lc_msgs, "BaseMessage")
    HumanMessage = getattr(lc_msgs, "HumanMessage")
    Annotated = getattr(te_mod, "Annotated")
    TypedDict = getattr(te_mod, "TypedDict")

    class State(TypedDict):  # type: ignore[misc]
        messages: Annotated[List[BaseMessage], add_messages]  # type: ignore[index]

    async def echo_node(state: State, **_: Any) -> Dict[str, Any]:
        messages: List[BaseMessage] = state.get("messages", [])
        last_user = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        content = getattr(last_user, "content", "")
        reply = AIMessage(content=f"Echo: {content}")
        return {"messages": [reply]}

    graph = StateGraph(State)
    graph.add_node("echo", echo_node)
    graph.set_entry_point("echo")
    graph.add_edge("echo", END)
    app = graph.compile()

    print("Emitting a couple LangGraph runs...")
    await app.ainvoke({"messages": [HumanMessage(content="hi there")]})
    await app.ainvoke({"messages": [HumanMessage(content="how are you?")]})


def main() -> None:
    _ensure_env_defaults()

    if not os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        print("Missing LangSmith API key. Set LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) and rerun.")
        return

    _log_synthetic_traces()

    try:
        asyncio.run(_maybe_run_tiny_langgraph())
    except RuntimeError:
        # Fallback for event loop already running (e.g. in notebooks)
        loop = asyncio.get_event_loop()
        loop.create_task(_maybe_run_tiny_langgraph())
        loop.run_until_complete(asyncio.sleep(0.1))

    print("Done. Visit LangSmith to see your new traces.")


if __name__ == "__main__":
    main()
