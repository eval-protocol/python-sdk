from pydantic import BaseModel
from pydantic_ai import Agent
import pytest

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import evaluation_test

from eval_protocol.pytest.default_pydantic_ai_rollout_processor import PydanticAgentRolloutProcessor
from agent import setup_agent
from pydantic_ai.models.openai import OpenAIModel


@pytest.mark.asyncio
@evaluation_test(
    input_messages=[Message(role="user", content="What is the total number of tracks in the database?")],
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
    num_runs=5,
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
            reasoning="No assistant message found",
        )
    elif not last_assistant_message.content:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            reasoning="No assistant message found",
        )
    else:
        model = OpenAIModel(
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
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
            system_prompt=(
                "Your job is to compare the response to the expected answer."
                "If the response is correct, return 1.0. If the response is incorrect, return 0.0."
            ),
            output_type=Response,
            model=model,
        )
        result = await comparison_agent.run(f"Expected answer: 3503\nResponse: {last_assistant_message.content}")
        row.evaluation_result = EvaluateResult(
            score=result.output.score,
            reasoning=result.output.reason,
        )
    return row
