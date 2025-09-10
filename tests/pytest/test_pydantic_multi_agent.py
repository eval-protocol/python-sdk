"""
Copied and modified for eval-protocol from https://ai.pydantic.dev/multi-agent-applications/#agent-delegation

To test your Pydantic AI multi-agent application, you can pass a factory that
sets up the agenet based on the completion_params. The function should accept a
RolloutProcessorConfig. In completion_params, you can provide mappings of model
to agent based on key.
"""

from pydantic_ai.models.openai import OpenAIChatModel
import pytest

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from pydantic_ai import Agent

from eval_protocol.pytest.default_pydantic_ai_rollout_processor import PydanticAgentRolloutProcessor
from pydantic_ai import RunContext
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from eval_protocol.pytest.types import RolloutProcessorConfig


def setup_agent(joke_generation_model: Model, joke_selection_model: Model) -> Agent:
    """
    This is an extra step that most applications will probably need to do to
    parameterize the model that their agents use. But we believe that this is a
    necessary step for multi-agent applications if developers want to solve the
    model selection problem.
    """
    joke_selection_agent = Agent(
        model=joke_selection_model,
        system_prompt=(
            "Use the `joke_factory` to generate some jokes, then choose the best. You must return just a single joke."
        ),
    )
    joke_generation_agent = Agent(joke_generation_model, output_type=list[str])

    @joke_selection_agent.tool
    async def joke_factory(ctx: RunContext[None], count: int) -> list[str]:  # pyright: ignore[reportUnusedFunction]
        r = await joke_generation_agent.run(
            f"Please generate {count} jokes.",
            usage=ctx.usage,
        )
        return r.output

    return joke_selection_agent


def agent_factory(config: RolloutProcessorConfig) -> Agent:
    joke_generation_model = OpenAIChatModel(
        config.completion_params["model"]["joke_generation_model"], provider="fireworks"
    )
    joke_selection_model = OpenAIChatModel(
        config.completion_params["model"]["joke_selection_model"], provider="fireworks"
    )
    return setup_agent(
        joke_generation_model,
        joke_selection_model,
    )


@pytest.mark.asyncio
@evaluation_test(
    input_messages=[[[Message(role="user", content="Tell me a joke.")]]],
    completion_params=[
        # multi-agent
        {
            "model": {
                "joke_generation_model": "accounts/fireworks/models/kimi-k2-instruct",
                "joke_selection_model": "accounts/fireworks/models/deepseek-v3p1",
            }
        },
    ],
    rollout_processor=PydanticAgentRolloutProcessor(
        agent_factory, UsageLimits(request_limit=5, total_tokens_limit=1000)
    ),
    mode="pointwise",
)
async def test_pydantic_multi_agent(row: EvaluationRow) -> EvaluationRow:
    """
    Super simple hello world test for Pydantic AI.
    """
    return row
