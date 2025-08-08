import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
)

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.pytest.default_single_turn_rollout_process import (
    default_single_turn_rollout_processor,
)


def test_handles_function_call_messages() -> None:
    async def run_test() -> None:
        tool_call = ChatCompletionMessageToolCall(
            id="call_1",
            type="function",
            function=ChatCompletionMessageToolCallFunction(
                name="get_weather", arguments="{}"
            ),
        )
        row = EvaluationRow(
            messages=[
                Message(role="user", content="Hi"),
                Message(role="assistant", tool_calls=[tool_call], content=""),
                Message(role="tool", tool_call_id="call_1", content="sunny"),
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        config = RolloutProcessorConfig(
            model="gpt-4o-mini", input_params={}, mcp_config_path=""
        )

        captured_messages: List[Dict[str, Any]] = []

        async def fake_acompletion(**kwargs: Any) -> Any:
            nonlocal captured_messages
            captured_messages = kwargs["messages"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="done",
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id="call_2",
                                    type="function",
                                    function=ChatCompletionMessageToolCallFunction(
                                        name="foo", arguments="{}"
                                    ),
                                )
                            ],
                            function_call=None,
                        )
                    )
                ]
            )

        with mock.patch(
            "eval_protocol.pytest.default_single_turn_rollout_process.acompletion",
            side_effect=fake_acompletion,
        ):
            dataset = await default_single_turn_rollout_processor([row], config)

        assert captured_messages[1]["tool_calls"][0]["id"] == "call_1"
        assert captured_messages[2]["tool_call_id"] == "call_1"
        result_row = dataset[0]
        assert result_row.messages[-1].tool_calls[0].id == "call_2"

    asyncio.run(run_test())
