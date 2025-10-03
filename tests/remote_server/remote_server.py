import os
import random
import threading

import uvicorn
from fastapi import FastAPI
from openai import OpenAI
import logging

from eval_protocol.models import Status
from eval_protocol.types.remote_rollout_processor import (
    InitRequest,
)
from eval_protocol.logging.elasticsearch_direct_http_handler import ElasticsearchDirectHttpHandler

app = FastAPI()

handler = ElasticsearchDirectHttpHandler()
# attach handler to root logger
logging.getLogger().addHandler(handler)
logger = logging.getLogger(__name__)


@app.post("/init")
def init(req: InitRequest):
    if req.elastic_search_config:
        handler.configure(req.elastic_search_config)

    # with a 50% chance, log that rollout has finished
    if random.random() < 0.5:
        logger.info(
            f"Rollout {req.metadata.rollout_id} finished",
            extra={"status": Status.rollout_finished(), "rollout_id": req.metadata.rollout_id},
        )
        return

    # Kick off worker thread that does a single-turn chat via Langfuse OpenAI integration
    def _worker(rollout_id: str):
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
            logger.info(
                f"Rollout {req.metadata.rollout_id} completed",
                extra={"status": Status.rollout_finished(), "rollout_id": rollout_id},
            )

    t = threading.Thread(target=_worker, daemon=True, args=(req.metadata.rollout_id,))
    t.start()


def main():
    host = os.getenv("REMOTE_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("REMOTE_SERVER_PORT", "3000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
