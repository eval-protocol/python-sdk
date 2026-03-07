from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

tinker = pytest.importorskip("tinker")
renderers = pytest.importorskip("tinker_cookbook.renderers")

from eval_protocol.integrations.tinker_rollout_processor import TinkerRolloutProcessor
from eval_protocol.integrations.tinker_utils import (
    build_tinker_renderer,
    normalize_eval_protocol_messages,
    resolve_renderer_name,
    tinker_message_to_eval_protocol_message,
)
from eval_protocol.models import EvaluateResult, EvaluationRow, ExecutionMetadata, Message
from eval_protocol.pytest.types import RolloutProcessorConfig


def test_normalize_eval_protocol_messages_preserves_multimodal_tool_metadata():
    messages = [
        Message.model_validate(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "board"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/lake.png"}},
                ],
            }
        ),
        Message.model_validate(
            {
                "role": "assistant",
                "reasoning_content": "think",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lake_move", "arguments": '{"action":"RIGHT"}'},
                    }
                ],
            }
        ),
        Message.model_validate(
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "lake_move",
                "content": "next board",
            }
        ),
    ]

    normalized = normalize_eval_protocol_messages(messages)

    assert normalized[0]["content"] == [
        {"type": "text", "text": "board"},
        {"type": "image", "image": "https://example.com/lake.png"},
    ]
    assert normalized[1]["content"] == [
        {"type": "thinking", "thinking": "think"},
        {"type": "text", "text": ""},
    ]
    assert normalized[1]["tool_calls"][0].function.name == "lake_move"
    assert normalized[1]["tool_calls"][0].function.arguments == '{"action": "RIGHT"}'
    assert normalized[2]["tool_call_id"] == "call_1"
    assert normalized[2]["name"] == "lake_move"


def test_build_tinker_renderer_uses_image_processor_for_vl_models(monkeypatch):
    calls: list[tuple[str, object | None]] = []

    def fake_get_image_processor(model_name):
        assert model_name == "Qwen/Qwen3-VL-30B-A3B-Instruct"
        return "image-processor"

    def fake_get_renderer(name, tokenizer, image_processor=None):
        calls.append((name, image_processor))
        return "renderer"

    monkeypatch.setattr("eval_protocol.integrations.tinker_utils.get_image_processor", fake_get_image_processor)
    monkeypatch.setattr("eval_protocol.integrations.tinker_utils.renderers.get_renderer", fake_get_renderer)

    tokenizer, renderer = build_tinker_renderer(
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        renderer_name="qwen3_vl_instruct",
        tokenizer="tok",
    )

    assert tokenizer == "tok"
    assert renderer == "renderer"
    assert calls == [("qwen3_vl_instruct", "image-processor")]


def test_resolve_renderer_name_prefers_kimi_k25():
    assert resolve_renderer_name("moonshotai/Kimi-K2.5") == "kimi_k25"


def test_tinker_message_to_eval_protocol_message_preserves_reasoning_and_tool_calls():
    tool_call = renderers.ToolCall(
        function=renderers.ToolCall.FunctionBody(
            name="lake_move",
            arguments='{"action":"DOWN"}',
        ),
        id="call_2",
    )
    message = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "inspect board"},
            {"type": "text", "text": "done"},
        ],
        "tool_calls": [tool_call],
    }

    converted = tinker_message_to_eval_protocol_message(message)

    assert converted.role == "assistant"
    assert converted.content == "done"
    assert converted.reasoning_content == "inspect board"
    assert converted.tool_calls[0].function.name == "lake_move"
    assert converted.tool_calls[0].function.arguments == '{"action":"DOWN"}'


@pytest.mark.asyncio
async def test_tinker_rollout_processor_round_trips_structured_messages(monkeypatch):
    class FakeRenderer:
        def __init__(self):
            self.conversations = []

        def build_generation_prompt(self, messages):
            self.conversations.append(messages)
            return tinker.ModelInput.from_ints([1, 2, 3])

        def get_stop_sequences(self):
            return [99]

        def parse_response(self, _tokens):
            return (
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "plan"},
                        {"type": "text", "text": ""},
                    ],
                    "tool_calls": [
                        renderers.ToolCall(
                            function=renderers.ToolCall.FunctionBody(
                                name="lake_move",
                                arguments='{"action":"RIGHT"}',
                            ),
                            id="call_3",
                        )
                    ],
                },
                True,
            )

    class FakeSamplingClient:
        async def sample_async(self, prompt, num_samples, sampling_params):
            assert isinstance(prompt, tinker.ModelInput)
            assert num_samples == 1
            assert sampling_params.stop == [99]
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=[42, 43])])

    processor = TinkerRolloutProcessor(
        sampling_client=FakeSamplingClient(),
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        renderer_name="qwen3_vl_instruct",
    )
    monkeypatch.setattr(
        "eval_protocol.integrations.tinker_rollout_processor.default_logger.log",
        lambda row: None,
    )
    fake_renderer = FakeRenderer()
    processor.renderer = fake_renderer
    processor.tokenizer = object()

    row = EvaluationRow(
        row_id="row-1",
        messages=[
            Message.model_validate(
                {
                    "role": "system",
                    "content": "You are playing FrozenLake.",
                }
            ),
            Message.model_validate(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "state"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/lake.png"}},
                    ],
                }
            ),
            Message.model_validate(
                {
                    "role": "tool",
                    "tool_call_id": "call_prev",
                    "name": "lake_move",
                    "content": "old board",
                }
            ),
        ],
        execution_metadata=ExecutionMetadata(),
        evaluation_result=EvaluateResult(score=0.0),
    )
    config = RolloutProcessorConfig(
        completion_params={"max_tokens": 32, "temperature": 0.0},
        semaphore=asyncio.Semaphore(1),
        mcp_config_path="",
        steps=1,
        logger=None,
        kwargs={},
    )

    tasks = processor([row], config)
    [processed_row] = await asyncio.gather(*tasks)

    convo = fake_renderer.conversations[0]
    assert convo[1]["content"] == [
        {"type": "text", "text": "state"},
        {"type": "image", "image": "https://example.com/lake.png"},
    ]
    assert convo[2]["tool_call_id"] == "call_prev"
    assert convo[2]["name"] == "lake_move"

    assistant = processed_row.messages[-1]
    assert assistant.role == "assistant"
    assert assistant.content == ""
    assert assistant.reasoning_content == "plan"
    assert assistant.tool_calls[0].function.name == "lake_move"
    assert assistant.tool_calls[0].function.arguments == '{"action":"RIGHT"}'
