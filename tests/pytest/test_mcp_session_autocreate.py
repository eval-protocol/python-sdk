"""
Regression tests for the airline MCP server.

The tests in this module ensure we can exercise key behaviours from the
AirlineDomainMcp server. They also act as regression coverage for performance
fixes that impact readiness probes.
"""

import importlib.util
from pathlib import Path
import time
from multiprocessing import Process

import httpx
import pytest

from eval_protocol.mcp.client.connection import MCPConnectionManager
from eval_protocol.types import MCPSession


def _load_airline_environment_module():
    module_name = "airline_environment_for_test"
    module_path = (
        Path(__file__).resolve().parents[2]
        / "eval_protocol"
        / "mcp_servers"
        / "tau2"
        / "airplane_environment"
        / "airline_environment.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_airline_environment_reset_uses_cached_db(monkeypatch):
    """AirlineEnvironment should only hit disk the first time it's reset."""

    airline_module = _load_airline_environment_module()

    from vendor.tau2.environment import db as db_module

    load_file_calls = 0
    original_load_file = db_module.load_file

    def counting_load_file(path: str, *args, **kwargs):
        nonlocal load_file_calls
        load_file_calls += 1
        return original_load_file(path, *args, **kwargs)

    monkeypatch.setattr(db_module, "load_file", counting_load_file)

    env = airline_module.AirlineEnvironment()

    env.reset()
    assert load_file_calls == 1
    assert env.db.users, "Expected seeded users in the airline database"

    user_id, user = next(iter(env.db.users.items()))
    original_first_name = user.name.first_name
    env.db.users[user_id].name.first_name = "Changed"

    env.reset()
    assert load_file_calls == 1
    assert env.db.users[user_id].name.first_name == original_first_name


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
