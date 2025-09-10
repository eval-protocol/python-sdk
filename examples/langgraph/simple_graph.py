from typing import Any, Dict, List
from typing_extensions import TypedDict, Annotated


def build_simple_graph(
    model: str = "accounts/fireworks/models/kimi-k2-instruct",
    *,
    model_provider: str = "fireworks",
    temperature: float = 0.0,
) -> Any:
    """
    Real LangGraph-based simple graph using LangChain-native messages:
    - State: {"messages": List[langchain_core.messages.BaseMessage]}
    - Single node that calls Fireworks via ChatFireworks
    - Exposes compiled app with .ainvoke
    Requires FIREWORKS_API_KEY to be set; no offline fallback.
    """

    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import BaseMessage
    from langchain.chat_models import init_chat_model

    class State(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]

    llm = init_chat_model(model, model_provider=model_provider, temperature=temperature)

    async def call_model(state: State, **_: Any) -> Dict[str, Any]:
        messages: List[BaseMessage] = state.get("messages", [])  # type: ignore[assignment]
        resp = await llm.ainvoke(messages)
        # Return only the delta; reducer will append
        return {"messages": [resp]}

    g = StateGraph(State)
    g.add_node("call_model", call_model)
    g.set_entry_point("call_model")
    g.add_edge("call_model", END)
    return g.compile()
