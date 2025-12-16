import asyncio

import pytest
from litellm.types.utils import Choices, Message as LLMMessage, ModelResponse

from eval_protocol.dataset_logger import default_logger
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor
from eval_protocol.pytest.exception_config import get_default_exception_handler_config
from eval_protocol.pytest.types import RolloutProcessorConfig


def test_single_turn_rollout_captures_logprobs(monkeypatch):
    processor = SingleTurnRolloutProcessor(drop_trailing_assistant_messages=False)

    config = RolloutProcessorConfig(
        completion_params={"model": "test-model", "logprobs": True, "top_logprobs": 2},
        mcp_config_path="",
        semaphore=asyncio.Semaphore(1),
        server_script_path=None,
        steps=1,
        logger=default_logger,
        exception_handler_config=get_default_exception_handler_config(),
    )

    row = EvaluationRow(messages=[Message(role="user", content="hi")])

    async def fake_acompletion(**kwargs):
        assert kwargs["logprobs"] is True
        assert kwargs["top_logprobs"] == 2
        logprobs = {"content": [{"token": "hello", "logprob": -0.1, "top_logprobs": []}]}
        return ModelResponse(
            id="resp-1",
            choices=[
                Choices(
                    index=0,
                    message=LLMMessage(role="assistant", content="hello"),
                    finish_reason="stop",
                    logprobs=logprobs,
                )
            ],
            created=0,
            model="test-model",
        )

    monkeypatch.setattr("eval_protocol.pytest.default_single_turn_rollout_process.acompletion", fake_acompletion)

    async def _run() -> None:
        tasks = processor([row], config)
        completed_rows = await asyncio.gather(*tasks)

        assert completed_rows[0].messages[-1].content == "hello"
        assistant_logprobs = completed_rows[0].messages[-1].logprobs
        assert isinstance(assistant_logprobs, dict)
        assert assistant_logprobs["content"][0]["token"] == "hello"
        assert assistant_logprobs["content"][0]["logprob"] == -0.1

    asyncio.run(_run())
