import asyncio

import pytest

from eval_protocol.integrations.fireworks_v1_completions_client import FireworksV1CompletionsClient


def test_plaintext_fallback_disabled_raises_on_non_json():
    client = FireworksV1CompletionsClient(
        model_id="accounts/fireworks/models/qwen3-0p6b",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
        allow_plaintext_action_fallback=False,
    )
    with pytest.raises(ValueError):
        client._parse_tool_call_with_optional_fallback("move RIGHT next")
    asyncio.run(client.close())


def test_plaintext_fallback_extracts_action_when_enabled():
    client = FireworksV1CompletionsClient(
        model_id="accounts/fireworks/models/qwen3-0p6b",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
        allow_plaintext_action_fallback=True,
    )
    parsed = client._parse_tool_call_with_optional_fallback("The best move is RIGHT.")
    assert parsed.arguments["action"] == "RIGHT"
    asyncio.run(client.close())


def test_plaintext_fallback_raises_when_no_action_found():
    client = FireworksV1CompletionsClient(
        model_id="accounts/fireworks/models/qwen3-0p6b",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
        allow_plaintext_action_fallback=True,
    )
    with pytest.raises(ValueError):
        client._parse_tool_call_with_optional_fallback("I cannot decide from this state.")
    asyncio.run(client.close())


def test_parse_assistant_output_preserves_non_tool_content(monkeypatch):
    client = FireworksV1CompletionsClient(
        model_id="accounts/fireworks/models/qwen3-0p6b",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
    )
    monkeypatch.setattr(client, "_parse_tool_call_with_vllm_parser", lambda **kwargs: None)
    parsed = client._parse_assistant_output(
        completion_text='<think>\n\n</think>\n{"tool_calls":[{"name":"lake_move","arguments":{"action":"RIGHT"}}]}',
        completion_token_ids=[1, 2, 3],
        tools=[{"type": "function", "function": {"name": "lake_move"}}],
    )
    assert parsed["parsed_tool_call"].arguments == {"action": "RIGHT"}
    assert parsed["assistant_content"] == "<think>\n\n</think>"
    assert parsed["non_tool_content"] == "<think>\n\n</think>"
    assert parsed["parser"] == "json_schema"
    asyncio.run(client.close())


def test_parse_assistant_output_uses_vllm_parser_when_available(monkeypatch):
    client = FireworksV1CompletionsClient(
        model_id="accounts/fireworks/models/qwen3-0p6b",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
    )

    class _Parsed:
        arguments = {"action": "DOWN"}

    monkeypatch.setattr(
        client,
        "_parse_tool_call_with_vllm_parser",
        lambda **kwargs: {"parsed_tool_call": _Parsed(), "assistant_content": "thought", "parser": "vllm:qwen3xml"},
    )
    parsed = client._parse_assistant_output(
        completion_text='{"tool_calls":[{"name":"lake_move","arguments":{"action":"DOWN"}}]}',
        completion_token_ids=[1, 2, 3],
        tools=[{"type": "function", "function": {"name": "lake_move"}}],
    )
    assert parsed["assistant_content"] == "thought"
    assert parsed["non_tool_content"] == "thought"
    assert parsed["parser"] == "vllm:qwen3xml"
    assert parsed["parsed_tool_call"].arguments == {"action": "DOWN"}
    asyncio.run(client.close())


def test_build_prompt_token_ids_retries_without_tools(monkeypatch):
    client = FireworksV1CompletionsClient(
        model_id="accounts/fireworks/models/qwen3-0p6b",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
    )

    class FakeTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(kwargs)
            if "tools" in kwargs:
                raise RuntimeError("tools unsupported")
            return [11, 22, 33]

        def encode(self, text, add_special_tokens=False):  # pragma: no cover
            return [99]

    fake_tokenizer = FakeTokenizer()
    monkeypatch.setattr(client, "_get_tokenizer", lambda: fake_tokenizer)
    token_ids = client._build_prompt_token_ids(
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "lake_move"}}],
    )
    assert token_ids == [11, 22, 33]
    assert len(fake_tokenizer.calls) == 2
    asyncio.run(client.close())


def test_build_prompt_token_ids_handles_dict_input_ids(monkeypatch):
    client = FireworksV1CompletionsClient(
        model_id="accounts/fireworks/models/qwen3-0p6b",
        tokenizer_name_or_path="Qwen/Qwen3-0.6B",
    )

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [[101, 102, 103]]}

        def encode(self, text, add_special_tokens=False):  # pragma: no cover
            return [99]

    monkeypatch.setattr(client, "_get_tokenizer", lambda: FakeTokenizer())
    token_ids = client._build_prompt_token_ids(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
    )
    assert token_ids == [101, 102, 103]
    asyncio.run(client.close())
