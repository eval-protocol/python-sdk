from pydantic import BaseModel
from pydantic_ai import Agent
import pytest

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import evaluation_test

from eval_protocol.pytest.default_pydantic_ai_rollout_processor import PydanticAgentRolloutProcessor
from tests.chinook.agent import setup_agent
import os
from pydantic_ai.models.openai import OpenAIModel

from tests.chinook.dataset import collect_dataset

LLM_JUDGE_PROMPT = (
    "Your job is to compare the response to the expected answer.\n"
    "The response will be a narrative report of the query results.\n"
    "If the response contains the same or well summarized information as the expected answer, return 1.0.\n"
    "If the response does not contain the same information or is missing information, return 0.0."
)


@pytest.mark.asyncio
@evaluation_test(
    input_messages=[[Message(role="user", content="What is the total number of tracks in the database?")]],
    completion_params=[
        {
            "model": {
                "orchestrator_agent_model": {
                    "model": "accounts/fireworks/models/kimi-k2-instruct",
                    "provider": "fireworks",
                }
            }
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(),
    rollout_processor_kwargs={"agent": setup_agent},
    mode="pointwise",
)
async def test_simple_query(row: EvaluationRow) -> EvaluationRow:
    """
    Super simple query for the Chinook database
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
        model = OpenAIModel(
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
            system_prompt=LLM_JUDGE_PROMPT,
            output_type=Response,
            model=model,
        )
        result = await comparison_agent.run(f"Expected answer: 3503\nResponse: {last_assistant_message.content}")
        row.evaluation_result = EvaluateResult(
            score=result.output.score,
            reason=result.output.reason,
        )
    return row


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@evaluation_test(
    input_rows=collect_dataset(),
    completion_params=[
        {
            "model": {
                "orchestrator_agent_model": {
                    "model": "accounts/fireworks/models/kimi-k2-instruct",
                    "provider": "fireworks",
                }
            }
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(),
    rollout_processor_kwargs={"agent": setup_agent},
    mode="pointwise",
)
async def test_complex_queries(row: EvaluationRow) -> EvaluationRow:
    """
    Complex queries for the Chinook database
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
        model = OpenAIModel(
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
            system_prompt=LLM_JUDGE_PROMPT,
            output_type=Response,
            model=model,
            result_retries=5,
        )
        result = await comparison_agent.run(
            f"Expected answer: {row.ground_truth}\nResponse: {last_assistant_message.content}"
        )
        row.evaluation_result = EvaluateResult(
            score=result.output.score,
            reason=result.output.reason,
        )
    return row
