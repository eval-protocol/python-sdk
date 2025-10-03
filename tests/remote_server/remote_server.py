import os
import random
import threading

import uvicorn
from fastapi import FastAPI
from openai import OpenAI
import logging

from eval_protocol import Status, InitRequest, ElasticsearchDirectHttpHandler, RolloutIdFilter


app = FastAPI()

# attach handler to root logger
handler = ElasticsearchDirectHttpHandler()
logging.getLogger().addHandler(handler)


@app.post("/init")
def init(req: InitRequest):
    if req.elastic_search_config:
        handler.configure(req.elastic_search_config)

    # attach rollout_id filter to logger
    logger = logging.getLogger(f"{__name__}.{req.metadata.rollout_id}")
    logger.addFilter(RolloutIdFilter(req.metadata.rollout_id))

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

            logger.info(f"Sending completion request to model {req.model}")
            completion = client.chat.completions.create(**completion_kwargs)
            logger.info(f"Completed response: {completion}")

        except Exception as e:
            # Best-effort; mark as done even on error to unblock polling
            print(f"❌ Error in rollout {req.metadata.rollout_id}: {e}")
            pass
        finally:
            logger.info(
                f"Rollout {req.metadata.rollout_id} completed",
                extra={"status": Status.rollout_finished()},
            )

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def main():
    host = os.getenv("REMOTE_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("REMOTE_SERVER_PORT", "3000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
