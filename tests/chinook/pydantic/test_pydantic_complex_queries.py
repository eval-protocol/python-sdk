import os
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
import pytest

from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.types import RolloutProcessorConfig
from tests.chinook.dataset import collect_dataset
from tests.chinook.pydantic.agent import setup_agent
from tests.pytest.test_pydantic_agent import PydanticAgentRolloutProcessor

LLM_JUDGE_PROMPT = (
    "You will be given the user task, the response of an AI assistant, and the expected answer.\n"
    "Compare the response to the expected answer. The response is a narrative report of the query results.\n"
    "Reward for core information being provided correctly and ignore formatting issues.\n"
    "Return a score from 0.0 to 1.0:\n"
    "- 1.0: response contains all or well summarized information as the expected answer\n"
    "- 0.0: response does not contain the same information or is missing all key information\n"
    "- Values in between: partial credit for responses that are partially correct or contain some but not all of the expected information."
)


def agent_factory(config: RolloutProcessorConfig) -> Agent:
    model_name = config.completion_params["model"]
    provider = config.completion_params.get("provider")
    reasoning = config.completion_params.get("reasoning")
    settings = OpenAIChatModelSettings(openai_reasoning_effort=reasoning)
    model = OpenAIChatModel(model_name, provider=provider or "openai", settings=settings)
    return setup_agent(model)


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Only run this test locally (skipped in CI)",
)
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[collect_dataset()],
    completion_params=[
        # {
        #     "model": "accounts/fireworks/models/kimi-k2-instruct",
        #     "provider": "fireworks",
        # },
        # {
        #     "model": "accounts/fireworks/models/deepseek-v3p1",
        #     "provider": "fireworks",
        # },
        # {
        #     "model": "accounts/fireworks/models/kimi-k2p5",
        #     "provider": "fireworks",
        # },
        {"model": "gpt-5"},
        # {"model": "gpt-5", "reasoning": "high"},
    ],
    rollout_processor=PydanticAgentRolloutProcessor(agent_factory),
    num_runs=2,
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
            "gpt-5",
        )

        first_user_message = row.get_first_user_message()
        if first_user_message is None:
            row.evaluation_result = EvaluateResult(
                score=0.0,
                reason="No user message found",
            )
            return row

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
            f"User task: {first_user_message.content}\nResponse: {last_assistant_message.content}\nExpected answer: {row.ground_truth}"
        )
        row.evaluation_result = EvaluateResult(
            score=result.output.score,
            reason=result.output.reason,
        )
    return row
