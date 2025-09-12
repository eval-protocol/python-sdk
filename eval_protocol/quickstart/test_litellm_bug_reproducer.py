"""
Minimal reproducer for LiteLLM event loop binding bug.

This reproduces the issue where LiteLLM's LoggingWorker gets bound to the first
event loop but tries to operate from subsequent event loops in pytest parameterized tests.

Only reproduces in CI environments (GitHub Actions), not locally.
"""

import pytest
import litellm


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        # "gpt-4o-mini",
        # "gpt-3.5-turbo",
        "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
        "fireworks_ai/accounts/fireworks/models/gpt-oss-20b",
    ],
)
async def test_multiple_models(model):
    response = await litellm.acompletion(model=model, messages=[{"role": "user", "content": "Hello"}])
    assert response.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]
