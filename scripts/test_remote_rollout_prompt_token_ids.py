#!/usr/bin/env python3
"""E2E check: RemoteRolloutProcessor reads prompt_token_ids trace payloads.

This starts a tiny local `/init` server, sends one chat completion through the
Fireworks tracing gateway with `return_token_ids`, and verifies that
RemoteRolloutProcessor hydrates `assistant_turn_payloads[*].prompt_token_ids`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import socket
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
from fastapi import FastAPI
from openai import OpenAI

from eval_protocol import FireworksTracingHttpHandler, InitRequest, RolloutIdFilter, Status
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

logger = logging.getLogger("remote_rollout_prompt_token_ids")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _message_to_dict(message: Message | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, Message):
        return message.dump_mdoel_for_chat_completion_request()
    return {k: v for k, v in dict(message).items() if v is not None}


def _make_app(gateway_url: str) -> FastAPI:
    app = FastAPI()
    app_logger = logging.getLogger(f"{__name__}.server")
    app_logger.setLevel(logging.INFO)

    @app.get("/")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/init")
    def init(req: InitRequest) -> dict[str, str]:
        rollout_logger = logging.getLogger(f"{__name__}.{req.metadata.rollout_id}")
        rollout_logger.addFilter(RolloutIdFilter(req.metadata.rollout_id))
        if not any(isinstance(handler, FireworksTracingHttpHandler) for handler in rollout_logger.handlers):
            rollout_logger.addHandler(FireworksTracingHttpHandler(gateway_base_url=gateway_url))
        rollout_logger.setLevel(logging.INFO)

        def _worker() -> None:
            try:
                conversation = [_message_to_dict(message) for message in (req.messages or [])]
                params = dict(req.completion_params or {})
                params.pop("base_url", None)
                params["extra_body"] = {
                    **dict(params.get("extra_body") or {}),
                    "return_token_ids": True,
                }
                params.setdefault("temperature", 0)
                params.setdefault("max_tokens", 8)

                if not req.model_base_url:
                    raise ValueError("model_base_url is required")
                if not params.get("model"):
                    raise ValueError("completion_params.model is required")

                client = OpenAI(base_url=req.model_base_url, api_key=req.api_key)
                response = client.chat.completions.create(messages=conversation, **params)
                content = response.choices[0].message.content or ""
                logger.info("remote server generated content=%r", content)

                rollout_logger.info(
                    "rollout %s finished",
                    req.metadata.rollout_id,
                    extra={"status": Status.rollout_finished()},
                )
            except Exception as exc:
                rollout_logger.exception(
                    "rollout %s failed",
                    req.metadata.rollout_id,
                    extra={"status": Status.rollout_unknown_error(str(exc))},
                )

        threading.Thread(target=_worker, daemon=True).start()
        return {"status": "started"}

    return app


def _wait_ready(url: str, timeout_seconds: float = 30.0) -> None:
    import requests

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"server not ready: {url}")


async def _run(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.getenv("FIREWORKS_DEV_API_KEY") or os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("Set FIREWORKS_DEV_API_KEY or FIREWORKS_API_KEY")

    # FireworksTracingHttpHandler reads FIREWORKS_API_KEY.
    os.environ["FIREWORKS_API_KEY"] = api_key
    os.environ["EP_REMOTE_API_KEY"] = api_key

    port = args.port or _free_port()
    remote_base_url = f"http://127.0.0.1:{port}"
    app = _make_app(args.gateway_url)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_ready(f"{remote_base_url}/")

    rollout_id = f"rrp-prompt-ids-{int(time.time())}"
    row = EvaluationRow(
        messages=[Message(role="user", content="Reply with exactly: ok")],
    )
    row.input_metadata.row_id = "row-0"
    row.input_metadata.completion_params = {
        "model": args.model,
        "base_url": args.api_base_url,
        "temperature": 0,
        "max_tokens": 8,
    }
    row.execution_metadata.rollout_id = rollout_id
    row.execution_metadata.invocation_id = "inv-0"
    row.execution_metadata.experiment_id = "fir2-1747-rrp-e2e"
    row.execution_metadata.run_id = "run-0"

    processor = RemoteRolloutProcessor(
        remote_base_url=remote_base_url,
        model_base_url=args.gateway_url,
        include_payloads=True,
        timeout_seconds=args.timeout_seconds,
        poll_interval=args.poll_interval,
    )
    try:
        task = processor(
            [row],
            RolloutProcessorConfig(
                completion_params=row.input_metadata.completion_params,
                mcp_config_path="",
                semaphore=asyncio.Semaphore(1),
                steps=1,
            ),
        )[0]
        completed = await task
    finally:
        await processor.acleanup()
        server.should_exit = True
        thread.join(timeout=5)

    extra = completed.execution_metadata.extra or {}
    turn_payloads = extra.get("assistant_turn_payloads") or []
    prompt_ids = None
    if turn_payloads:
        prompt_ids = turn_payloads[0].get("prompt_token_ids")
    if prompt_ids is None:
        prompt_ids = extra.get("prompt_token_ids")

    print(f"rollout_id={rollout_id}")
    print(f"messages={len(completed.messages)}")
    print(f"assistant_turn_payloads={turn_payloads}")
    print(f"prompt_token_ids_len={len(prompt_ids) if isinstance(prompt_ids, list) else None}")
    print(f"prompt_token_ids_head={prompt_ids[:8] if isinstance(prompt_ids, list) else None}")

    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise AssertionError("RemoteRolloutProcessor did not hydrate prompt_token_ids")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default=os.getenv("EP_MODEL_BASE_URL", "https://litellm-gateway-dev-j4kzagdteq-uc.a.run.app"))
    parser.add_argument("--api-base-url", default=os.getenv("FIREWORKS_API_BASE_URL", "https://dev.api.fireworks.ai/inference/v1"))
    parser.add_argument("--model", default=os.getenv("TRACING_E2E_MODEL", "accounts/pyroworks-dev/deployments/malaysia2-intended-butterfly"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
