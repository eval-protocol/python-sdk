from collections.abc import Awaitable, Callable
import os
from typing_extensions import cast
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
import pytest

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.types import RolloutProcessorConfig
from tests.chinook.dataset import collect_dataset
from tests.chinook.pydantic.agent import setup_agent
from tests.pytest.test_pydantic_agent import PydanticAgentRolloutProcessor

# IMPORTANT: import must be renamed to something without the "test_" prefix to
# avoid pytest discovering the import as a test
from tests.chinook.pydantic.test_pydantic_complex_queries import test_pydantic_complex_queries as eval


def agent_factory(config: RolloutProcessorConfig) -> Agent:
    model_name = config.completion_params["model"]
    reasoning = config.completion_params.get("reasoning")
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort=reasoning,
    )
    model = OpenAIResponsesModel(model_name, settings=settings)
    return setup_agent(model)


@pytest.mark.skipif(  # pyright: ignore[reportAttributeAccessIssue]
    os.environ.get("CI") == "true",
    reason="This was only run locally to generate traces in Responses API",
)
@pytest.mark.asyncio  # pyright: ignore[reportAttributeAccessIssue]
@evaluation_test(
    input_rows=[collect_dataset()],
    completion_params=[
        {
            "model": "gpt-5",
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(agent_factory),
)
async def test_pydantic_complex_queries_responses(row: EvaluationRow) -> EvaluationRow:
    """
    Evaluation of complex queries for the Chinook database using PydanticAI
    """
    casted_evaluation_test = cast(Callable[[EvaluationRow], Awaitable[EvaluationRow]], eval)
    evaluated_row = await casted_evaluation_test(row)
    return evaluated_row
