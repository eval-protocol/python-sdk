import os
import threading
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langfuse.openai import openai  # pyright: ignore[reportPrivateImportUsage]


app = FastAPI()


class InitRequest(BaseModel):
    rollout_id: str
    model: str
    messages: list[dict]
    tools: list[dict] | None = None
    metadata: dict
    num_turns: int = 2


_STATE: Dict[str, Dict[str, Any]] = {}


ALLOWED_MESSAGE_FIELDS = {"role", "content", "tool_calls", "tool_call_id", "name"}


def _clean_messages_for_api(messages: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        cm = {k: v for k, v in msg.items() if k in ALLOWED_MESSAGE_FIELDS and v is not None}
        # Some providers dislike empty content on assistant messages; keep if present
        cleaned.append(cm)
    return cleaned


@app.post("/init")
def init(req: InitRequest):
    # Persist state
    _STATE[req.rollout_id] = {"terminated": False}

    # Kick off worker thread that runs multi-turn chat via Langfuse OpenAI integration
    def _worker():
        try:
            # Prepare tags for Langfuse filtering
            metadata = {
                "langfuse_tags": [
                    f"invocation_id:{req.metadata.get('invocation_id')}",
                    f"experiment_id:{req.metadata.get('experiment_id')}",
                    f"rollout_id:{req.metadata.get('rollout_id')}",
                    f"run_id:{req.metadata.get('run_id')}",
                    f"row_id:{req.metadata.get('row_id')}",
                ]
            }

            messages = req.messages

            # Simulate N-1 assistant turns (single-shot or simple echo)
            for _ in range(max(1, req.num_turns)):
                completion_kwargs = {
                    "model": req.model,
                    "messages": _clean_messages_for_api(messages),
                    "metadata": metadata,
                }

                if req.tools:
                    completion_kwargs["tools"] = req.tools

                completion = openai.chat.completions.create(**completion_kwargs)
                assistant_message = completion.choices[0].message

                # Convert to dict format for next turn
                assistant_dict = {"role": "assistant", "content": assistant_message.content}
                if assistant_message.tool_calls:
                    assistant_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in assistant_message.tool_calls
                    ]

                # Append assistant for next turn
                messages = messages + [assistant_dict]

        except Exception:
            # Best-effort; mark as done even on error to unblock polling
            pass
        finally:
            _STATE[req.rollout_id]["terminated"] = True

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return {"ok": True}


@app.get("/status")
def status(rollout_id: str):
    st = _STATE.get(rollout_id)
    if not st:
        raise HTTPException(status_code=404, detail="unknown rollout_id")
    return {"terminated": bool(st.get("terminated", False))}


def main():
    host = os.getenv("REMOTE_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("REMOTE_SERVER_PORT", "7077"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
