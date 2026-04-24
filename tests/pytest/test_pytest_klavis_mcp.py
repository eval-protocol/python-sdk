from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import AgentRolloutProcessor, evaluation_test
from openai import AsyncOpenAI
import json
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
import os


class ResponseFormat(BaseModel):
    score: float


"""
You should copy https://painted-tennis-ebc.notion.site/MCPMark-Source-Hub-23181626b6d7805fb3a7d59c63033819
into your Notion for the notion test.
"""


@evaluation_test(
    input_dataset=["tests/pytest/datasets/klavis_mcp_test.jsonl"],
    rollout_processor=AgentRolloutProcessor(),
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/kimi-k2p5"}],
    mode="pointwise",
    mcp_config_path="tests/pytest/mcp_configurations/klavis_strata_mcp.json",
)
async def test_pytest_klavis_mcp(row: EvaluationRow) -> EvaluationRow:
    ground_truth = row.ground_truth
    # check if the final messages contains the ground truth

    async with AsyncOpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/inference/v1"
    ) as client:
        response = await client.chat.completions.create(
            model="accounts/fireworks/models/kimi-k2p5",
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

        row.evaluation_result = EvaluateResult(
            score=score,
            reason=response_text,
        )
    return row
