from typing import Any, Dict, List
from typing_extensions import Annotated, TypedDict


def build_reasoning_graph(
    *,
    model: str = "accounts/fireworks/models/gpt-oss-120b",
    model_provider: str = "fireworks",
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
) -> Any:
    """
    LangGraph example: use Fireworks reasoning model gpt-oss-120b with structured state.

    Requirements:
    - Install: `pip install langchain fireworks-ai`.
    - Env: export `FIREWORKS_API_KEY`.

    Notes:
    - You can control reasoning behavior via extra_body (reasoning_effort). Common values: "low", "medium", "high".
    - The graph is a single-node message app that calls the model and appends the response.

    Example:
        graph = build_reasoning_graph(reasoning_effort="high")
        out = await graph.ainvoke({"messages": [{"role": "user", "content": "Explain why the sky is blue."}]})
    """

    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import BaseMessage

    class State(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]

    # Initialize Fireworks reasoning model
    llm = init_chat_model(
        model,
        model_provider=model_provider,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )

    async def call_model(state: State) -> Dict[str, Any]:
        response = await llm.ainvoke(state["messages"])  # type: ignore[assignment]
        return {"messages": [response]}

    g = StateGraph(State)
    g.add_node("call_model", call_model)
    g.set_entry_point("call_model")
    g.add_edge("call_model", END)
    return g.compile()
