import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, List

import asyncio
import pytest
from pydantic import BaseModel
from unittest import mock


# ---- Stub external dependencies ----
openai = types.ModuleType("openai")
types_mod = types.ModuleType("openai.types")
chat_mod = types.ModuleType("openai.types.chat")
chat_msg_mod = types.ModuleType("openai.types.chat.chat_completion_message")


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolFunction(BaseModel):
    name: str
    arguments: str


class ChatCompletionMessageToolCall(BaseModel):
    id: str
    type: str
    function: ToolFunction


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


chat_msg_mod.FunctionCall = FunctionCall
chat_msg_mod.ChatCompletionMessageToolCall = ChatCompletionMessageToolCall
chat_mod.chat_completion_message = chat_msg_mod
openai.types = types_mod
types_mod.chat = chat_mod
types_mod.CompletionUsage = CompletionUsage
sys.modules["openai"] = openai
sys.modules["openai.types"] = types_mod
sys.modules["openai.types.chat"] = chat_mod
sys.modules["openai.types.chat.chat_completion_message"] = chat_msg_mod


# Stub litellm
litellm = types.ModuleType("litellm")


async def acompletion(**kwargs):
    raise NotImplementedError


litellm.acompletion = acompletion
sys.modules["litellm"] = litellm


# Stub eval_protocol models and types
class Message(BaseModel):
    role: str
    content: Any = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: List[ChatCompletionMessageToolCall] | None = None
    function_call: FunctionCall | None = None


class EvaluationRow(BaseModel):
    messages: List[Message]
    tools: Any = None
    ground_truth: Any = None


@dataclass
class RolloutProcessorConfig:
    model: str
    input_params: Dict[str, Any]
    mcp_config_path: str
    server_script_path: str | None = None
    max_concurrent_rollouts: int = 8
    steps: int = 30


# Register stub modules
import_path = "/workspace/python-sdk/eval_protocol"
eval_protocol_pkg = types.ModuleType("eval_protocol")
eval_protocol_pkg.__path__ = [import_path]
models_module = types.ModuleType("eval_protocol.models")
models_module.Message = Message
models_module.EvaluationRow = EvaluationRow
pytest_pkg = types.ModuleType("eval_protocol.pytest")
pytest_pkg.__path__ = [f"{import_path}/pytest"]
types_module = types.ModuleType("eval_protocol.pytest.types")
types_module.RolloutProcessorConfig = RolloutProcessorConfig

sys.modules["eval_protocol"] = eval_protocol_pkg
sys.modules["eval_protocol.models"] = models_module
sys.modules["eval_protocol.pytest"] = pytest_pkg
sys.modules["eval_protocol.pytest.types"] = types_module


# Now we can import the rollout processor
from eval_protocol.pytest.default_single_turn_rollout_process import (
    default_single_turn_rollout_processor,
)


def test_handles_function_call_messages():
    async def run_test():
        tool_call = ChatCompletionMessageToolCall(
            id="call_1",
            type="function",
            function=ToolFunction(name="get_weather", arguments="{}"),
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

        async def fake_acompletion(**kwargs):
            nonlocal captured_messages
            captured_messages = kwargs["messages"]
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="done",
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id="call_2",
                                    type="function",
                                    function=ToolFunction(name="foo", arguments="{}"),
                                )
                            ],
                            function_call=None,
                        )
                    )
                ]
            )

        with pytest.raises(NotImplementedError):
            await acompletion()

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
