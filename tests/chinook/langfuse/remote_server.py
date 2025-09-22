import os
import threading
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests


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

    # Kick off worker thread that runs multi-turn chat via LiteLLM proxy
    def _worker():
        try:
            base_url = os.getenv(
                "LITELLM_BASE_URL",
                "https://litellm-cloud-proxy-prod-644257448872.us-central1.run.app",
            )
            url = f"{base_url}/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('FIREWORKS_API_KEY', '')}",
                "Content-Type": "application/json",
            }

            # Prepare metadata payload to attach for Langfuse filtering
            metadata = {
                "invocation_id": req.metadata.get("invocation_id"),
                "experiment_id": req.metadata.get("experiment_id"),
                "rollout_id": req.metadata.get("rollout_id"),
                "run_id": req.metadata.get("run_id"),
                "row_id": req.metadata.get("row_id"),
            }

            messages = req.messages

            # Simulate N-1 assistant turns (single-shot or simple echo)
            for _ in range(max(1, req.num_turns)):
                payload = {
                    "model": req.model,
                    "messages": _clean_messages_for_api(messages),
                    "metadata": metadata,
                }
                if req.tools:
                    payload["tools"] = req.tools
                r = requests.post(url, json=payload, headers=headers, timeout=60)
                r.raise_for_status()
                data = r.json()
                assistant = data.get("choices", [{}])[0].get("message", {})
                # Append assistant for next turn
                messages = messages + [assistant]

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
