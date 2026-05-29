import argparse
import logging
import os
import threading

import uvicorn
from fastapi import FastAPI
from openai import OpenAI

from eval_protocol import FireworksTracingHttpHandler, InitRequest, RolloutIdFilter, Status


app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

fireworks_handler = FireworksTracingHttpHandler()
logging.getLogger().addHandler(fireworks_handler)


def _clean_messages(messages):
    clean_messages = []
    for message in messages:
        if hasattr(message, "dump_mdoel_for_chat_completion_request"):
            message_dict = message.dump_mdoel_for_chat_completion_request()
        elif hasattr(message, "model_dump"):
            message_dict = message.model_dump(exclude_none=True)
        elif isinstance(message, dict):
            message_dict = {key: value for key, value in message.items() if value is not None}
        else:
            message_dict = {
                "role": getattr(message, "role", None),
                "content": getattr(message, "content", None),
            }
            message_dict = {key: value for key, value in message_dict.items() if value is not None}
        clean_messages.append(message_dict)
    return clean_messages


@app.post("/init")
def init(req: InitRequest):
    logger = logging.getLogger(f"{__name__}.{req.metadata.rollout_id}")
    logger.addFilter(RolloutIdFilter(req.metadata.rollout_id))

    def _worker():
        try:
            if not req.messages:
                raise ValueError("messages is required")

            model = req.completion_params.get("model")
            if not model:
                raise ValueError("model is required in completion_params")

            completion_params = {key: value for key, value in req.completion_params.items() if key != "base_url"}
            client = OpenAI(base_url=req.model_base_url, api_key=os.environ.get("FIREWORKS_API_KEY"))

            conversation_history = _clean_messages(req.messages)
            logger.info("Turn 1: sending completion request to model %s", model)
            completion = client.chat.completions.create(
                messages=conversation_history,
                **completion_params,
            )
            assistant_content = completion.choices[0].message.content or ""
            conversation_history.append({"role": "assistant", "content": assistant_content})
            logger.info("Turn 1 response: %s", assistant_content[:100])

            follow_up = "Use that answer in one short sentence."
            conversation_history.append({"role": "user", "content": follow_up})
            logger.info("Turn 2: user asks: %s", follow_up)
            completion = client.chat.completions.create(
                messages=conversation_history,
                **completion_params,
            )
            assistant_content = completion.choices[0].message.content or ""
            logger.info("Turn 2 response: %s", assistant_content[:100])

        except Exception as e:
            logger.error("Error in rollout %s: %s", req.metadata.rollout_id, e)
            pass
        finally:
            logger.info(
                "Rollout %s completed",
                req.metadata.rollout_id,
                extra={"status": Status.rollout_finished()},
            )

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def main():
    parser = argparse.ArgumentParser(description="Run the two-turn logprobs remote server")
    parser.add_argument("--host", default=os.getenv("REMOTE_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("REMOTE_SERVER_PORT", "3000")))
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
