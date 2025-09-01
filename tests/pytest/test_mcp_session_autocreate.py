"""
Regression test: ensure MCP-Gym auto-creates a session on first tool call
without requiring a prior initial state fetch, and returns JSON.
"""

import time
from multiprocessing import Process

import httpx
import pytest

from eval_protocol.mcp.client.connection import MCPConnectionManager
from eval_protocol.types import MCPSession


def _run_airline_server():
    import os

    os.environ["PORT"] = "9780"
    from eval_protocol.mcp_servers.tau2.tau2_mcp import AirlineDomainMcp

    server = AirlineDomainMcp(seed=None)
    server.run(transport="streamable-http")


@pytest.mark.asyncio
async def test_tool_call_returns_json_without_prior_initial_state():
    proc = Process(target=_run_airline_server, daemon=True)
    proc.start()

    try:
        base_url = "http://127.0.0.1:9780/mcp"
        client = httpx.Client(timeout=1.0)
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                r = client.get(base_url)
                if r.status_code in (200, 307, 406):
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            pytest.fail("Server did not start on port 9780 in time")

        session = MCPSession(base_url=base_url, session_id="test-autocreate", seed=None, model_id="test-model")

        mgr = MCPConnectionManager()
        await mgr.initialize_session(session)
        await mgr.discover_tools(session)

        observation, reward, done, info = await mgr.call_tool(session, "list_all_airports", {})

        assert isinstance(observation, dict), f"Expected JSON dict, got: {type(observation)} {observation}"
        assert observation.get("error") != "invalid_json_response"

        await mgr.reset_session(session)
        await mgr.close_session(session)
    finally:
        proc.terminate()
        proc.join(timeout=5)
