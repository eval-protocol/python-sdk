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

    python_version = os.environ.get("PYTHON_VERSION", "3.10").replace(".", "")
    port = str(9780 + int(python_version[-1:]))
    os.environ["PORT"] = port
    from eval_protocol.mcp_servers.tau2.tau2_mcp import AirlineDomainMcp

    server = AirlineDomainMcp(seed=None)
    server.run(transport="streamable-http")


@pytest.mark.asyncio
async def test_tool_call_returns_json_without_prior_initial_state():
    import os

    proc = Process(target=_run_airline_server, daemon=True)
    proc.start()

    try:
        python_version = os.environ.get("PYTHON_VERSION", "3.10").replace(".", "")
        port = str(9780 + int(python_version[-1:]))

        base_url = f"http://127.0.0.1:{port}/mcp"
        client = httpx.Client(timeout=1.0)
        start_time = time.time()
        deadline = start_time + 20
        ready_time = None
        while time.time() < deadline:
            try:
                r = client.get(base_url)
                if r.status_code in (200, 307, 406):
                    ready_time = time.time()
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            pytest.fail("Server did not start on port 9780 in time")

        assert ready_time is not None, "Server did not return a successful status before exiting loop"
        assert ready_time - start_time < 20, f"Server took too long to respond: {ready_time - start_time:.2f}s"

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
