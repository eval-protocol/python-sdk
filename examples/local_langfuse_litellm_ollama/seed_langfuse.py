from langfuse import get_client
from langfuse.types import TraceContext


def main() -> None:
    lf = get_client()
    trace_id = lf.create_trace_id()
    ctx = TraceContext(trace_id=trace_id)
    lf.update_current_trace(
        name="local-demo",
        tags=["local", "demo"],
        input={"messages": [{"role": "user", "content": "What is 2+2?"}]},
    )
    lf.start_generation(trace_context=ctx, name="final")
    lf.update_current_generation(output={"messages": [{"role": "assistant", "content": "It is 4."}]})
    lf.flush()
    print("Created trace:", trace_id)


if __name__ == "__main__":
    main()
