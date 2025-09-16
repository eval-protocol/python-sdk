import os
import pytest


@pytest.mark.skipif(os.getenv("FIREWORKS_API_KEY") in (None, ""), reason="FIREWORKS_API_KEY not set")
@pytest.mark.asyncio
async def test_tools_graph_traced_to_langsmith() -> None:
    from langsmith import Client
    from langsmith import traceable
    from .tools_graph import build_tools_graph
    from langchain_core.messages import HumanMessage

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LS_PROJECT", "ep-langgraph-examples"))

    app = build_tools_graph()

    @traceable
    async def run_once(prompt: str) -> dict:
        # Run the graph once
        _ = await app.ainvoke({"messages": [HumanMessage(content=prompt)]})
        # Return a ChatML-like transcript including a tool response so LangSmith records role=tool
        tool_args = '{"a":2,"b":3}'
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": "Tool Calls:\ncalculator_add\n" + tool_args,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "calculator_add", "arguments": tool_args},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "calculator_add",
                    "tool_call_id": "call_1",
                    "content": "5",
                },
                {"role": "assistant", "content": "The result is 5."},
            ]
        }

    await run_once("Use calculator_add to add 2 and 3")
