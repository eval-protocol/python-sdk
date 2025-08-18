from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from werkzeug.wrappers import Response

import eval_protocol as ep


# Sync tests for the ep.make() function
def test_mcp_env_make_appends_trailing_slash():
    """
    Verify that ep.make() appends a trailing slash to the MCP server URL if it's missing.
    This prevents 307 redirects that can break HTTP clients.
    """
    base_url = "http://localhost:8000/mcp"
    corrected_url = "http://localhost:8000/mcp/"

    envs = ep.make(base_url, n=1, seeds=[42])

    assert len(envs.sessions) == 1
    assert envs.sessions[0].base_url == corrected_url


def test_mcp_env_make_keeps_existing_trailing_slash():
    """
    Verify that ep.make() does not add an extra slash if one is already present.
    """
    base_url = "http://localhost:8000/mcp/"

    envs = ep.make(base_url, n=1, seeds=[42])

    assert len(envs.sessions) == 1
    # The session's base_url should remain unchanged
    assert envs.sessions[0].base_url == base_url


# Async test for the underlying HTTP client behavior
@pytest.mark.asyncio
async def test_post_request_is_preserved_on_307_redirect(httpserver):
    """
    Validates that the underlying HTTP client preserves the POST method
    when following a 307 redirect, as a guard against client behavior regressions.

    This confirms the behavior of the `httpx` client.
    """
    # Configure a 307 redirect from /mcp-old to /mcp/
    httpserver.expect_request("/mcp-old", method="POST").respond_with_response(
        Response(
            status=307,
            headers={"Location": httpserver.url_for("/mcp/")},
        )
    )

    # The final endpoint must receive a POST request after the redirect
    httpserver.expect_request("/mcp/", method="POST").respond_with_json({"status": "ok"})

    redirecting_url = httpserver.url_for("/mcp-old")

    # Use httpx.AsyncClient directly to test the redirect behavior.
    # We must explicitly enable redirects.
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # We manually use the client to send a POST request to the redirecting URL.
        response = await client.post(redirecting_url, content=b"test_data")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    # pytest-httpserver automatically asserts that both expected requests were received.
