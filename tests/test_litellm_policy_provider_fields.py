import types

import pytest

import eval_protocol.mcp.execution.policy as policy_mod
from eval_protocol.mcp.execution.policy import LiteLLMPolicy


@pytest.mark.asyncio
async def test_litellm_policy_surfaces_provider_specific_reasoning_details(monkeypatch):
    """
    Ensure that provider_specific_fields from the LiteLLM message object are
    preserved on the returned message dict from LiteLLMPolicy._make_llm_call.
    """

    # Define a fake ModelResponse base class and patch the module's ModelResponse
    class FakeModelResponseBase: ...

    policy_mod.ModelResponse = FakeModelResponseBase

    async def fake_acompletion(*args, **kwargs):
        # This mimics the LiteLLM Message object shape we rely on in policy._make_llm_call
        message_obj = types.SimpleNamespace(
            role="assistant",
            content="",
            tool_calls=[
                types.SimpleNamespace(
                    id="tool_get_reservation_details_123",
                    type="function",
                    function=types.SimpleNamespace(
                        name="get_reservation_details",
                        arguments='{"reservation_id":"EHGLP3"}',
                    ),
                )
            ],
            provider_specific_fields={
                "reasoning_details": [{"id": "tool_get_reservation_details_123", "type": "reasoning.encrypted"}],
                "custom_field": "keep_me",
            },
        )

        class FakeModelResponse(FakeModelResponseBase):
            def __init__(self) -> None:
                self.choices = [
                    types.SimpleNamespace(
                        finish_reason="tool_calls",
                        index=0,
                        message=message_obj,
                    )
                ]
                self.usage = types.SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                )

        return FakeModelResponse()

    # Patch acompletion so we don't hit the network
    monkeypatch.setattr(policy_mod, "acompletion", fake_acompletion)

    # Use a concrete policy instance; base_url/model_id values don't matter for this unit test
    policy = LiteLLMPolicy(model_id="openrouter/google/gemini-3-pro-preview", use_caching=False)

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tool_get_reservation_details_123",
                    "type": "function",
                    "function": {"name": "get_reservation_details", "arguments": '{"reservation_id":"EHGLP3"}'},
                }
            ],
        }
    ]

    # No tools are needed for this test – we only care about the returned message shape
    result = await policy._make_llm_call(messages, tools=[])

    assert "choices" in result
    assert len(result["choices"]) == 1
    msg = result["choices"][0]["message"]

    # Core fields should be present
    assert msg["role"] == "assistant"
    assert isinstance(msg.get("tool_calls"), list)

    # provider_specific_fields should be preserved on the message
    ps = msg.get("provider_specific_fields")
    assert isinstance(ps, dict)
    assert ps["reasoning_details"] == [{"id": "tool_get_reservation_details_123", "type": "reasoning.encrypted"}]
    # Non-core provider_specific_fields should also be preserved
    assert ps.get("custom_field") == "keep_me"
