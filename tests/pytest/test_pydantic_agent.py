import os
import pytest

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from pydantic_ai import Agent

from eval_protocol.pytest.default_pydantic_ai_rollout_processor import PydanticAgentRolloutProcessor

agent = Agent()


@pytest.mark.asyncio
@evaluation_test(
    input_messages=[Message(role="user", content="Hello, how are you?")],
    completion_params=[
        {"model": "accounts/fireworks/models/gpt-oss-120b", "provider": "fireworks"},
    ],
    rollout_processor=PydanticAgentRolloutProcessor(),
    rollout_processor_kwargs={"agent": agent},
    mode="pointwise",
)
async def test_pydantic_agent(row: EvaluationRow) -> EvaluationRow:
    """
    Super simple hello world test for Pydantic AI.
    """
    return row
