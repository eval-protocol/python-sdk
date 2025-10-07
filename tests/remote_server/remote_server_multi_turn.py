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

    # Kick off worker thread that does a multi-turn chat (6 turns total)
    def _worker():
        try:
            if not req.messages:
                raise ValueError("messages is required")

            client = OpenAI(base_url=req.model_base_url, api_key=os.environ.get("FIREWORKS_API_KEY"))

            # Build up conversation over 6 turns (3 user messages + 3 assistant responses)
            # Convert Message objects to dicts for OpenAI API
            conversation_history = [{"role": m.role, "content": m.content} for m in req.messages]

            follow_up_questions = [
                "Tell me more about that.",
                "What else can you share about this topic?",
            ]

            # First completion (turns 1-2: initial user message + assistant response)
            logger.info(f"Turn 1-2: Sending initial completion request to model {req.model}")
            completion = client.chat.completions.create(
                model=req.model,
                messages=conversation_history,  # type: ignore
            )
            assistant_message = completion.choices[0].message
            assistant_content = assistant_message.content or ""
            conversation_history.append({"role": "assistant", "content": assistant_content})
            logger.info(f"Turn 2 response: {assistant_content[:100]}...")

            # Second completion (turns 3-4: follow-up user message + assistant response)
            conversation_history.append({"role": "user", "content": follow_up_questions[0]})
            logger.info(f"Turn 3: User asks: {follow_up_questions[0]}")
            completion = client.chat.completions.create(
                model=req.model,
                messages=conversation_history,  # type: ignore
            )
            assistant_message = completion.choices[0].message
            assistant_content = assistant_message.content or ""
            conversation_history.append({"role": "assistant", "content": assistant_content})
            logger.info(f"Turn 4 response: {assistant_content[:100]}...")

            # Third completion (turns 5-6: another follow-up user message + assistant response)
            conversation_history.append({"role": "user", "content": follow_up_questions[1]})
            logger.info(f"Turn 5: User asks: {follow_up_questions[1]}")
            completion = client.chat.completions.create(
                model=req.model,
                messages=conversation_history,  # type: ignore
            )
            assistant_message = completion.choices[0].message
            assistant_content = assistant_message.content or ""
            conversation_history.append({"role": "assistant", "content": assistant_content})
            logger.info(f"Turn 6 response: {assistant_content[:100]}...")

            logger.info(f"Completed 6-turn conversation with {len(conversation_history)} messages total")

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
