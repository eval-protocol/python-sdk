import os
import random
import threading
import argparse

import uvicorn
from fastapi import FastAPI
from openai import OpenAI
import logging

from eval_protocol import Status, InitRequest, FireworksTracingHttpHandler, RolloutIdFilter


app = FastAPI()

# Configure logging for the remote server (required for INFO-level logs to be emitted)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

# Attach Fireworks tracing handler to root logger
fireworks_handler = FireworksTracingHttpHandler()
logging.getLogger().addHandler(fireworks_handler)


force_early_error_message = None


@app.post("/init")
def init(req: InitRequest):
    # Attach rollout_id filter to logger
    logger = logging.getLogger(f"{__name__}.{req.metadata.rollout_id}")
    logger.addFilter(RolloutIdFilter(req.metadata.rollout_id))

    # Kick off worker thread that does a single-turn chat via Langfuse OpenAI integration
    def _worker():
        try:
            if not req.messages:
                raise ValueError("messages is required")

            model = req.completion_params.get("model")
            if not model:
                raise ValueError("model is required in completion_params")

            # Convert Eval Protocol Message objects into OpenAI-compatible dicts,
            # excluding any None fields (Fireworks rejects extra keys even when null).
            messages_payload = []
            for m in req.messages:
                if hasattr(m, "dump_mdoel_for_chat_completion_request"):
                    md = m.dump_mdoel_for_chat_completion_request()  # type: ignore[attr-defined]
                elif hasattr(m, "model_dump"):
                    md = m.model_dump(exclude_none=True)  # type: ignore[call-arg]
                elif isinstance(m, dict):
                    md = {k: v for k, v in m.items() if v is not None}
                else:
                    md = {"role": getattr(m, "role", None), "content": getattr(m, "content", None)}
                    md = {k: v for k, v in md.items() if v is not None}
                messages_payload.append(md)

            # Spread completion_params; omit base_url (client uses req.model_base_url; gateway
            # encodes inference base_url into the tracing path via build_init_request).
            completion_kwargs = {
                "messages": messages_payload,
                **{k: v for k, v in req.completion_params.items() if k != "base_url"},
            }

            if req.tools:
                completion_kwargs["tools"] = req.tools

            logger.info(f"Final completion_kwargs: {completion_kwargs}")

            client = OpenAI(base_url=req.model_base_url, api_key=os.environ.get("FIREWORKS_API_KEY"))

            logger.info(f"Sending completion request to model {model}")
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
