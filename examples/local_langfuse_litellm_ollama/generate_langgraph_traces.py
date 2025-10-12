import asyncio
import os
from typing import Any, Dict, List

from langfuse import get_client


def _to_chatml_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role == "ai" or role == "assistant":
            entry: Dict[str, Any] = {"role": "assistant", "content": getattr(m, "content", "")}
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                try:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": getattr(tc, "type", "function"),
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tcs
                    ]
                except Exception:
                    pass
            out.append(entry)
        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "name": getattr(m, "name", None),
                    "tool_call_id": getattr(m, "tool_call_id", None),
                    "content": getattr(m, "content", ""),
                }
            )
        elif role == "human" or role == "user":
            out.append({"role": "user", "content": getattr(m, "content", "")})
    return out


async def main() -> None:
    # Lazy import to avoid hard deps unless used
    import sys
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    from examples.langgraph.tools_graph import build_tools_graph
    from langchain_core.messages import HumanMessage

    num = int(os.environ.get("LANGGRAPH_TRACE_COUNT", "10"))
    lf = get_client()
    app = build_tools_graph()

    prompts = [
        "Use calculator_add to add 2 and 3",
        "Calculate 5 + 7",
        "What is 10 + 1?",
        "Add 8 and 9",
        "Tool test: 4 plus 4",
    ]

    for i in range(num):
        prompt = prompts[i % len(prompts)]
        # Create input in ChatML-like form
        input_msgs = [{"role": "user", "content": prompt}]

        # Invoke graph and build output ChatML messages
        result = await app.ainvoke({"messages": [HumanMessage(content=prompt)]})
        output_msgs = _to_chatml_messages(result.get("messages", []))

        # Create trace with input/output for adapter to parse
        trace_id = lf.create_trace_id()
        from langfuse.types import TraceContext

        ctx = TraceContext(trace_id=trace_id)
        # Create concrete events to ensure ingestion attaches to this trace
        lf.create_event(trace_context=ctx, name="input", input={"messages": input_msgs})
        lf.create_event(trace_context=ctx, name="assistant", output={"messages": output_msgs})
        # Also set top-level trace metadata for adapter context
        lf.update_current_trace(name="langgraph-demo")
        lf.flush()
        print("Created langgraph trace:", trace_id)


if __name__ == "__main__":
    asyncio.run(main())
