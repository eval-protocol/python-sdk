import pytest

pytest.skip(
    "Skipping Chinook pydantic-ai integration tests (requires external LLM access/credentials).",
    allow_module_level=True,
)

from pydantic import BaseModel
from pydantic_ai import Agent

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import evaluation_test

from eval_protocol.pytest.default_pydantic_ai_rollout_processor import PydanticAgentRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from tests.chinook.pydantic.agent import setup_agent
import os
from pydantic_ai.models.openai import OpenAIChatModel

from tests.chinook.dataset import collect_dataset

LLM_JUDGE_PROMPT = (
    "Your job is to compare the response to the expected answer.\n"
    "The response will be a narrative report of the query results.\n"
    "If the response contains the same or well summarized information as the expected answer, return 1.0.\n"
    "If the response does not contain the same information or is missing information, return 0.0."
)


def agent_factory(config: RolloutProcessorConfig) -> Agent:
    model_name = config.completion_params["model"]
    provider = config.completion_params["provider"] if "provider" in config.completion_params else "openai"
    model = OpenAIChatModel(model_name, provider=provider)
    return setup_agent(model)


@pytest.mark.parametrize(
    "completion_params",
    [
        {
            "model": "accounts/fireworks/models/kimi-k2p5",
            "provider": "fireworks",
            "reasoning_effort": "none",
        },
        {
            "model": "gpt-5",
        },
    ],
)
@evaluation_test(
    input_messages=[[[Message(role="user", content="What is the total number of tracks in the database?")]]],
    rollout_processor=PydanticAgentRolloutProcessor(agent_factory),
    mode="pointwise",
)
async def test_simple_query(row: EvaluationRow) -> EvaluationRow:
    """
    Super simple query for the Chinook database
    """
    expected_tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_sql",
                "parameters": {
                    "additionalProperties": False,
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "type": "object",
                },
            },
        }
    ]
    assert hasattr(row, "tools"), "Row missing 'tools' attribute"
    assert row.tools == expected_tools, f"Tools validation failed. Expected: {expected_tools}, Got: {row.tools}"

    # assert that there is a system message
    assert row.messages[0].role == "system"

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
            "accounts/fireworks/models/kimi-k2p5",
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
