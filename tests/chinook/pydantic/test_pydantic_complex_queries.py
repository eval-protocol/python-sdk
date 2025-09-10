import os
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
import pytest

from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.types import RolloutProcessorConfig
from tests.chinook.dataset import collect_dataset
from tests.chinook.pydantic.agent import setup_agent
from tests.pytest.test_pydantic_agent import PydanticAgentRolloutProcessor

LLM_JUDGE_PROMPT = (
    "Your job is to compare the response to the expected answer.\n"
    "The response will be a narrative report of the query results.\n"
    "If the response contains the same or well summarized information as the expected answer, return 1.0.\n"
    "If the response does not contain the same information or is missing information, return 0.0."
)


def agent_factory(config: RolloutProcessorConfig) -> Agent:
    model_name = config.completion_params["model"]
    provider = config.completion_params["provider"]
    model = OpenAIChatModel(model_name, provider=provider)
    return setup_agent(model)


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[collect_dataset()],
    completion_params=[
        {
            "model": "accounts/fireworks/models/kimi-k2-instruct",
            "provider": "fireworks",
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(agent_factory),
)
async def test_pydantic_complex_queries(row: EvaluationRow) -> EvaluationRow:
    """
    Evaluation of complex queries for the Chinook database using PydanticAI
    """
    last_assistant_message = row.last_assistant_message()
    if last_assistant_message is None:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            reason="No assistant message found",
        )
    elif not last_assistant_message.content:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            reason="No assistant message found",
        )
    else:
        model = OpenAIChatModel(
            "accounts/fireworks/models/kimi-k2-instruct",
            provider="fireworks",
        )

        class Response(BaseModel):
            """
            A score between 0.0 and 1.0 indicating whether the response is correct.
            """

            score: float

            """
            A short explanation of why the response is correct or incorrect.
            """
            reason: str

        comparison_agent = Agent(
            model=model,
            system_prompt=LLM_JUDGE_PROMPT,
            output_type=Response,
            output_retries=5,
        )
        result = await comparison_agent.run(
            f"Expected answer: {row.ground_truth}\nResponse: {last_assistant_message.content}"
        )
        row.evaluation_result = EvaluateResult(
            score=result.output.score,
            reason=result.output.reason,
        )
    return row
