import os
import threading
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from langfuse.openai import openai  # pyright: ignore[reportPrivateImportUsage]

from eval_protocol.types.remote_rollout_processor import (
    InitRequest,
    StatusResponse,
    create_langfuse_config_tags,
)
from eval_protocol.models import Message


app = FastAPI()


_STATE: Dict[str, Dict[str, Any]] = {}


@app.post("/init")
def init(req: InitRequest):
    # Persist state
    _STATE[req.rollout_id] = {"terminated": False}

    # Kick off worker thread that does a single-turn chat via Langfuse OpenAI integration
    def _worker():
        try:
            metadata = {"langfuse_tags": create_langfuse_config_tags(req)}

            completion_kwargs = {
                "model": req.model,
                "messages": req.messages,
                "metadata": metadata,
            }

            if req.tools:
                completion_kwargs["tools"] = req.tools

            completion = openai.chat.completions.create(**completion_kwargs)

        except Exception as e:
            # Best-effort; mark as done even on error to unblock polling
            print(f"❌ Error in rollout {req.rollout_id}: {e}")
            pass
        finally:
            _STATE[req.rollout_id]["terminated"] = True

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
    port = int(os.getenv("REMOTE_SERVER_PORT", "7077"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
