import asyncio

import litellm
import pytest

from eval_protocol.dataset_logger import default_logger
from eval_protocol.litellm_compat import allow_litellm_logging_to_start
from eval_protocol.mcp.execution.policy import LiteLLMPolicy
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor
from eval_protocol.pytest.exception_config import get_default_exception_handler_config
from eval_protocol.pytest.types import RolloutProcessorConfig
from vendor.tau2.data_model.message import UserMessage
from vendor.tau2.utils.llm_utils import generate


@pytest.mark.parametrize("index", range(4))
@pytest.mark.asyncio
async def test_acompletion_across_pytest_event_loops(index: int) -> None:
    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "ping"}],
        api_key="test",
        mock_response=f"ok-{index}",
    )
    await allow_litellm_logging_to_start()

    assert response.choices[0].message.content == f"ok-{index}"


@pytest.mark.parametrize(("index", "stream"), [(0, False), (1, True), (2, False), (3, True)])
@pytest.mark.asyncio
async def test_single_turn_processor_across_pytest_event_loops(index: int, stream: bool) -> None:
    config = RolloutProcessorConfig(
        completion_params={
            "model": "openai/gpt-4o-mini",
            "api_key": "test",
            "mock_response": f"single-turn-{index}",
            "stream": stream,
        },
        mcp_config_path="",
        semaphore=asyncio.Semaphore(1),
        server_script_path=None,
        steps=1,
        logger=default_logger,
        exception_handler_config=get_default_exception_handler_config(),
    )
    row = EvaluationRow(messages=[Message(role="user", content="ping")])

    result = await SingleTurnRolloutProcessor()([row], config)[0]

    assert result.messages[-1].content == f"single-turn-{index}"


@pytest.mark.parametrize(("index", "stream"), [(0, False), (1, True), (2, False), (3, True)])
@pytest.mark.asyncio
async def test_litellm_policy_across_pytest_event_loops(index: int, stream: bool) -> None:
    policy = LiteLLMPolicy(
        model_id="openai/gpt-4o-mini",
        use_caching=False,
        api_key="test",
        mock_response=f"policy-{index}",
        stream=stream,
    )

    result = await policy._make_llm_call([{"role": "user", "content": "ping"}], tools=[])

    assert result["choices"][0]["message"]["content"] == f"policy-{index}"


@pytest.mark.parametrize("index", range(2))
@pytest.mark.asyncio
async def test_tau2_generate_across_pytest_event_loops(index: int) -> None:
    result = await generate(
        model="openai/gpt-4o-mini",
        messages=[UserMessage(role="user", content="ping")],
        api_key="test",
        mock_response=f"tau2-{index}",
    )

    assert result.content == f"tau2-{index}"
