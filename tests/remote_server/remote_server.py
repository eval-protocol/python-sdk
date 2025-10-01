import os
import threading
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from openai import OpenAI

from eval_protocol.types.remote_rollout_processor import (
    InitRequest,
    StatusResponse,
)


app = FastAPI()


_STATE: Dict[str, Dict[str, Any]] = {}


@app.post("/init")
def init(req: InitRequest):
    # Persist state
    _STATE[req.metadata.rollout_id] = {"terminated": False}

    # Kick off worker thread that does a single-turn chat via Langfuse OpenAI integration
    def _worker():
        try:
            if not req.messages:
                raise ValueError("messages is required")

            completion_kwargs = {
                "model": req.model,
                "messages": req.messages,
            }

            if req.tools:
                completion_kwargs["tools"] = req.tools

            client = OpenAI(base_url=req.model_base_url, api_key=os.environ.get("FIREWORKS_API_KEY"))

            completion = client.chat.completions.create(**completion_kwargs)

        except Exception as e:
            # Best-effort; mark as done even on error to unblock polling
            print(f"❌ Error in rollout {req.metadata.rollout_id}: {e}")
            pass
        finally:
            _STATE[req.metadata.rollout_id]["terminated"] = True

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


@app.get("/status", response_model=StatusResponse)
def status(rollout_id: str) -> StatusResponse:
    st = _STATE.get(rollout_id)
    if not st:
        raise HTTPException(status_code=404, detail="unknown rollout_id")
    return StatusResponse(terminated=bool(st.get("terminated", False)))


def main():
    host = os.getenv("REMOTE_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("REMOTE_SERVER_PORT", "3000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
