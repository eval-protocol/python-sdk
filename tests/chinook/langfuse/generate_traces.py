import pytest
import os

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test

from eval_protocol.pytest.default_pydantic_ai_rollout_processor import PydanticAgentRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from tests.chinook.pydantic.agent import setup_agent

from tests.chinook.dataset import collect_dataset

try:
    from langfuse import get_client, observe  # pyright: ignore[reportPrivateImportUsage]
    from pydantic_ai.agent import Agent
    from pydantic_ai.models.openai import OpenAIModel

    LANGFUSE_AVAILABLE = True
    langfuse_client = get_client()

    Agent.instrument_all()

except ImportError:
    LANGFUSE_AVAILABLE = False
    langfuse_client = None

    def observe(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if args and callable(args[0]) else decorator


LLM_JUDGE_PROMPT = (
    "Your job is to compare the response to the expected answer.\n"
    "The response will be a narrative report of the query results.\n"
    "If the response contains the same or well summarized information as the expected answer, return 1.0.\n"
    "If the response does not contain the same information or is missing information, return 0.0."
)


def agent_factory(config: RolloutProcessorConfig) -> Agent:
    model_name = config.completion_params["model"]
    provider = config.completion_params["provider"]
    model = OpenAIModel(model_name, provider=provider)
    return setup_agent(model)


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@observe()
@evaluation_test(
    input_rows=[collect_dataset()[0:1]],
    completion_params=[
        {
            "model": "accounts/fireworks/models/kimi-k2-instruct",
            "provider": "fireworks",
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(agent_factory),
    mode="pointwise",
)
async def test_complex_query_0(row: EvaluationRow) -> EvaluationRow:
    """
    Complex queries - PydanticAI automatically creates rich Langfuse traces.
    """
    # Have to postprocess tools because row.tools isn't set until during rollout
    if langfuse_client:
        langfuse_client.update_current_trace(tags=["chinook_sql"], metadata={"tools": row.tools})

    return row


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@observe()
@evaluation_test(
    input_rows=[collect_dataset()[1:2]],
    completion_params=[
        {
            "model": "accounts/fireworks/models/kimi-k2-instruct",
            "provider": "fireworks",
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(agent_factory),
    mode="pointwise",
)
async def test_complex_query_1(row: EvaluationRow) -> EvaluationRow:
    """
    Complex queries - PydanticAI automatically creates rich Langfuse traces.
    """
    if langfuse_client:
        langfuse_client.update_current_trace(tags=["chinook_sql"], metadata={"tools": row.tools})

    return row


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@observe()
@evaluation_test(
    input_rows=[collect_dataset()[2:3]],
    completion_params=[
        {
            "model": "accounts/fireworks/models/kimi-k2-instruct",
            "provider": "fireworks",
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(agent_factory),
    mode="pointwise",
)
async def test_complex_query_2(row: EvaluationRow) -> EvaluationRow:
    """
    Complex queries - PydanticAI automatically creates rich Langfuse traces.
    """
    if langfuse_client:
        langfuse_client.update_current_trace(tags=["chinook_sql"], metadata={"tools": row.tools})

    return row


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@observe()
@evaluation_test(
    input_rows=[collect_dataset()[3:4]],
    completion_params=[
        {
            "model": "accounts/fireworks/models/kimi-k2-instruct",
            "provider": "fireworks",
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(agent_factory),
    mode="pointwise",
)
async def test_complex_query_3(row: EvaluationRow) -> EvaluationRow:
    """
    Complex queries - PydanticAI automatically creates rich Langfuse traces.
    """
    if langfuse_client:
        langfuse_client.update_current_trace(tags=["chinook_sql"], metadata={"tools": row.tools})

    return row


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@observe()
@evaluation_test(
    input_rows=[collect_dataset()[4:5]],
    completion_params=[
        {
            "model": "accounts/fireworks/models/kimi-k2-instruct",
            "provider": "fireworks",
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(agent_factory),
    mode="pointwise",
)
async def test_complex_query_4(row: EvaluationRow) -> EvaluationRow:
    """
    Complex queries - PydanticAI automatically creates rich Langfuse traces.
    """
    if langfuse_client:
        langfuse_client.update_current_trace(tags=["chinook_sql"], metadata={"tools": row.tools})

    return row


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@observe()
@evaluation_test(
    input_rows=[collect_dataset()[5:6]],
    completion_params=[
        {
            "model": "accounts/fireworks/models/kimi-k2-instruct",
            "provider": "fireworks",
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(agent_factory),
    mode="pointwise",
)
async def test_complex_query_5(row: EvaluationRow) -> EvaluationRow:
    """
    Complex queries - PydanticAI automatically creates rich Langfuse traces.
    """
    if langfuse_client:
        langfuse_client.update_current_trace(tags=["chinook_sql"], metadata={"tools": row.tools})

    return row
