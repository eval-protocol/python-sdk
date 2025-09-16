from typing import Any, Dict, List
from typing_extensions import TypedDict, Annotated


def build_tools_graph() -> Any:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import BaseMessage
    from langchain.chat_models import init_chat_model

    class State(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]

    # Use fireworks provider; expects FIREWORKS_API_KEY
    llm = init_chat_model(
        "accounts/fireworks/models/kimi-k2-instruct",
        model_provider="fireworks",
        temperature=0.0,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator_add",
                    "description": "Add two integers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ],
    )

    async def tool_router(state: State, **_: Any) -> Dict[str, Any]:
        msgs: List[BaseMessage] = state.get("messages", [])
        resp = await llm.ainvoke(msgs)
        # If tool call requested, synthesize tool result message
        try:
            tcs = getattr(resp, "tool_calls", None)
            if tcs:
                # naive parse for demo
                a, b = 0, 0
                try:
                    import json

                    args = json.loads(tcs[0].function.arguments)
                    a = int(args.get("a", 0))
                    b = int(args.get("b", 0))
                except Exception:
                    pass
                result = a + b
                from langchain_core.messages import ToolMessage

                tool_msg = ToolMessage(content=str(result), tool_call_id=tcs[0].id, name=tcs[0].function.name)
                return {"messages": [resp, tool_msg]}
        except Exception:
            pass
        return {"messages": [resp]}

    g = StateGraph(State)
    g.add_node("tool_router", tool_router)
    g.set_entry_point("tool_router")
    g.add_edge("tool_router", END)
    return g.compile()
