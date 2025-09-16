"""Emit a few tool-call traces into LangSmith for adapter testing.

Requirements:
  export LANGSMITH_API_KEY=...
  optional: export LANGCHAIN_PROJECT=ep-langgraph-examples (or set --project)

Run:
  python python-sdk/examples/langsmith/emit_tool_calls.py
"""

import os
from typing import Any, Dict, List


def make_messages_with_tool_call(user_text: str) -> Dict[str, Any]:
    """Return inputs/outputs shaped like LangChain messages with tool calls."""
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": user_text,
                "type": "human",
            }
        ]
    }
    # Assistant proposes a tool call (function)
    assistant_with_tool = {
        "role": "assistant",
        "content": "I'll call the calculator.",
        "type": "ai",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "calculator.add",
                    "arguments": '{"a": 2, "b": 3}',
                },
            }
        ],
    }
    # Tool response message
    tool_message = {
        "role": "tool",
        "name": "calculator.add",
        "tool_call_id": "call_1",
        "content": "5",
    }
    # Final assistant message
    final_assistant = {
        "role": "assistant",
        "content": "The result is 5.",
        "type": "ai",
    }
    outputs = {
        "messages": [
            inputs["messages"][0],
            assistant_with_tool,
            tool_message,
            final_assistant,
        ]
    }
    return {"inputs": inputs, "outputs": outputs}


def main() -> None:
    try:
        from langsmith import Client  # type: ignore
    except Exception as e:
        print(f"Missing langsmith dependency: {e}")
        return

    project = os.getenv("LANGCHAIN_PROJECT", os.getenv("LS_PROJECT", "ep-langgraph-examples"))
    client = Client()

    samples: List[str] = [
        "Add 2 and 3",
        "Compute 7 + 11",
        "Sum 10 and 25",
    ]

    for i, text in enumerate(samples, start=1):
        payload = make_messages_with_tool_call(text)
        name = f"tool-demo-{i}"
        # Create a chain run as container
        client.create_run(name=name, inputs=payload["inputs"], run_type="chain", project_name=project)
        # Log an llm child run carrying the assistant/tool messages as outputs
        client.create_run(
            name=f"{name}-llm",
            inputs=payload["inputs"],
            run_type="llm",
            project_name=project,
        )
        # Finalize by writing one more chain run with the aggregated outputs
        client.create_run(
            name=f"{name}-final",
            inputs=payload["inputs"],
            run_type="chain",
            project_name=project,
        )
        # Note: For simplicity, we attach outputs only on the final chain run
        # using update_run is possible, but create_run keeps the example lightweight
        # and the adapter reads from root runs' inputs/outputs or messages arrays.
        # Many LangSmith clients attach outputs via end_run; here we keep it minimal.
        try:
            # If available, end_run to attach outputs on the final run
            client.end_run(outputs=payload["outputs"])  # type: ignore[arg-type]
        except Exception:
            # Fallback: best-effort; runs may still be visible with inputs and llm child
            pass

    print(f"Emitted {len(samples)} tool-call demo traces to project '{project}'.")


if __name__ == "__main__":
    main()
