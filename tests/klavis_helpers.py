"""Helpers shared by live Klavis evaluations and offline tests."""

import json
import logging
import os

from openai import AsyncOpenAI
from pydantic import BaseModel

from eval_protocol.models import EvaluateResult, EvaluationRow, Message

logger = logging.getLogger(__name__)
MODEL = os.environ.get("EP_KLAVIS_MODEL", "accounts/fireworks/models/gpt-oss-120b")


class ResponseFormat(BaseModel):
    score: float


async def judge_klavis_response(row: EvaluationRow) -> EvaluationRow:
    """Judge a completed Klavis rollout; service errors must fail the live test."""
    ground_truth = row.ground_truth

    async with AsyncOpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/inference/v1"
    ) as client:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are judging the output of the model versus the ground truth. Return score = 1 if the output contains the ground truth, 0 otherwise.",
                },
                {
                    "role": "user",
                    "content": f"Final model output: {row.messages[-1].content}\nGround truth: {ground_truth}",
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ResponseFormat", "schema": ResponseFormat.model_json_schema()},
            },
        )
        response_text = response.choices[0].message.content
        logger.info("response_text: %s", response_text)
        try:
            parsed = json.loads(response_text or "{}")
            score = parsed.get("score", 0.0)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse response as JSON: %s", response_text)
            score = 0.0

        row.evaluation_result = EvaluateResult(score=score, reason=response_text)
    return row


def klavis_gmail_sandbox_dataset_adapter(rows: list[dict]) -> list[EvaluationRow]:
    """Dataset adapter for sandbox JSONL rows.

    Supports the new schema:
      - initialize_data: dict (passed to Klavis sandbox initializer)
      - messages: str (task instruction)
      - ground_truth: dict (expected final sandbox state)

    """
    adapted: list[EvaluationRow] = []
    system_prompt = (
        "You are a helpful assistant with access to Gmail. "
        "You can send emails, draft emails, and manage messages, etc."
    )

    for r in rows:
        if isinstance(r.get("messages"), str) and "initialize_data" in r:
            init_data = r.get("initialize_data") or {}
            task = r.get("messages") or ""
            ground_truth = r.get("ground_truth")

            row = EvaluationRow(
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=task),
                ],
                ground_truth=ground_truth,
            )
            row.input_metadata.session_data = {
                "initialize_data": init_data,
                "task": task,
            }
            adapted.append(row)
        else:
            adapted.append(EvaluationRow(**r))

    return adapted
