import os
import random
import time
from typing import List

from langfuse import get_client
from langfuse.types import TraceContext


def _random_prompt(i: int) -> str:
    prompts = [
        "Summarize the benefits of local inference.",
        "What is 2+2?",
        "Explain how LiteLLM routes requests.",
        "Give a short description of the Chinook sample database.",
        "List three ways to evaluate model quality.",
    ]
    return prompts[i % len(prompts)]


def create_trace(lf, user_text: str, assistant_text: str, tags: List[str]) -> str:
    trace_id = lf.create_trace_id()
    ctx = TraceContext(trace_id=trace_id)
    # Attach input to trace
    lf.update_current_trace(
        name="local-synth", tags=tags, input={"messages": [{"role": "user", "content": user_text}]}
    )
    # Add a generation observation for the assistant reply
    lf.start_observation(trace_context=ctx, as_type="generation", name="assistant")
    lf.update_current_generation(output={"messages": [{"role": "assistant", "content": assistant_text}]})
    lf.flush()
    return trace_id


def main() -> None:
    count = int(os.environ.get("SYNTHETIC_TRACE_COUNT", "25"))
    lf = get_client()
    tags = ["local", "demo", "synthetic"]

    for i in range(count):
        user_q = _random_prompt(i)
        assistant_a = f"Synthetic response {i}: {random.choice(['Sure.', 'Okay.', 'Here you go.', 'Result: 4'])}"
        tid = create_trace(lf, user_q, assistant_a, tags)
        print(f"Created synthetic trace: {tid}")
        time.sleep(0.1)


if __name__ == "__main__":
    main()
