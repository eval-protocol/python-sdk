import os
import random
import threading
import argparse

import uvicorn
from fastapi import FastAPI
from openai import OpenAI
import logging

from eval_protocol import Status, InitRequest, ElasticsearchDirectHttpHandler, RolloutIdFilter


app = FastAPI()

# attach handler to root logger
handler = ElasticsearchDirectHttpHandler()
logging.getLogger().addHandler(handler)


force_early_error_message = None


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

            # If force_early_error is set via command-line arg, log the error and return early
            if force_early_error_message:
                logger.error(
                    force_early_error_message,
                    extra={"status": Status.rollout_error(force_early_error_message)},
                )
                raise RuntimeError(force_early_error_message)

        except Exception as e:
            # Best-effort; mark as done even on error to unblock polling
            logger.error(f"❌ Error in rollout {req.metadata.rollout_id}: {e}")
            pass
        finally:
            if not force_early_error_message:
                logger.info(
                    f"Rollout {req.metadata.rollout_id} completed",
                    extra={"status": Status.rollout_finished()},
                )

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def main():
    global force_early_error_message

    parser = argparse.ArgumentParser(description="Run the remote server for evaluation protocol")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("REMOTE_SERVER_HOST", "127.0.0.1"),
        help="Host to bind the server to (default: 127.0.0.1 or REMOTE_SERVER_HOST env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("REMOTE_SERVER_PORT", "3000")),
        help="Port to bind the server to (default: 3000 or REMOTE_SERVER_PORT env var)",
    )
    parser.add_argument(
        "--force-early-error",
        type=str,
        default=None,
        help="If set, /init will immediately return after logging a rollout_error with this message",
    )

    args = parser.parse_args()
    force_early_error_message = args.force_early_error

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
