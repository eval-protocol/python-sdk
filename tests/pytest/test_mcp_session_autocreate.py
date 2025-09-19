"""
Regression test: ensure MCP-Gym auto-creates a session on first tool call
without requiring a prior initial state fetch, and returns JSON.
"""

import time
from multiprocessing import Process, Queue

import httpx
import pytest

from eval_protocol.mcp.client.connection import MCPConnectionManager
from eval_protocol.types import MCPSession


def _run_airline_server(port_queue):
    import os

    # Use different ports based on Python version to avoid conflicts in parallel CI runs
    python_version = os.environ.get("PYTHON_VERSION", "3.10").replace(".", "")
    port = str(9780 + int(python_version[-2:]))  # 9780, 9781, 9782
    os.environ["PORT"] = port

    port_queue.put(int(port))  # Send the port back to the test
    from eval_protocol.mcp_servers.tau2.tau2_mcp import AirlineDomainMcp

    server = AirlineDomainMcp(seed=None)
    server.run(transport="streamable-http")


@pytest.mark.asyncio
async def test_tool_call_returns_json_without_prior_initial_state():
    port_queue = Queue()
    proc = Process(target=_run_airline_server, args=(port_queue,), daemon=True)
    proc.start()

    try:
        # Get the dynamically assigned port
        port = port_queue.get(timeout=10)
        base_url = f"http://127.0.0.1:{port}/mcp"
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
            pytest.fail(f"Server did not start on port {port} in time")

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
