import logging
import os
import threading

from fastapi import FastAPI
from openai import OpenAI

from eval_protocol import FireworksTracingHttpHandler, InitRequest, RolloutIdFilter, Status


app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logging.getLogger().addHandler(FireworksTracingHttpHandler())


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/init")
def init(req: InitRequest) -> dict[str, str]:
    logger = logging.getLogger(f"{__name__}.{req.metadata.rollout_id}")
    logger.addFilter(RolloutIdFilter(req.metadata.rollout_id))

    def _worker() -> None:
        try:
            messages = [message.model_dump(exclude_none=True) for message in req.messages or []]
            completion_kwargs = {
                "messages": messages,
                **{k: v for k, v in req.completion_params.items() if k != "base_url"},
            }
            if req.tools:
                completion_kwargs["tools"] = req.tools

            api_key = req.api_key or os.environ.get("FIREWORKS_API_KEY")
            if not api_key:
                raise ValueError("FIREWORKS_API_KEY is required locally or in the /init payload")

            client = OpenAI(base_url=req.model_base_url, api_key=api_key)
            completion = client.chat.completions.create(**completion_kwargs)
            logger.info("Completed rollout response: %s", completion)
        except Exception as exc:
            logger.error("Rollout failed: %s", exc, extra={"status": Status.rollout_error(str(exc))})
        else:
            logger.info("Rollout completed", extra={"status": Status.rollout_finished()})

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}
