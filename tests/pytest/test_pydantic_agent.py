from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
import pytest

from eval_protocol.models import EvaluationRow, Message, Status
from eval_protocol.pytest import evaluation_test

from eval_protocol.pytest.default_pydantic_ai_rollout_processor import PydanticAgentRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig


def agent_factory(config: RolloutProcessorConfig) -> Agent:
    model = OpenAIChatModel(config.completion_params["model"], provider="fireworks")
    return Agent(model=model)


@pytest.mark.asyncio
@evaluation_test(
    input_messages=[[[Message(role="user", content="Hello, how are you?")]]],
    completion_params=[
        {"model": "accounts/fireworks/models/gpt-oss-120b"},
    ],
    rollout_processor=PydanticAgentRolloutProcessor(agent_factory),
    mode="pointwise",
)
async def test_pydantic_agent(row: EvaluationRow) -> EvaluationRow:
    """
    Super simple hello world test for Pydantic AI.
    """
    assert row.rollout_status.code == Status.Code.FINISHED
    return row
