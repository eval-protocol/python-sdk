#!/usr/bin/env python3
"""
Simple MCP Server for Testing get_initial_state Concurrency
Simulates the exact pattern: envs.reset() -> get_initial_state -> slow HTTP endpoint
"""

import asyncio
import os
import time

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

# Create a simple MCP server
mcp = FastMCP(name="TestServer")


@mcp.custom_route("/control/initial_state", methods=["GET"])
async def get_initial_state_endpoint(request: Request) -> JSONResponse:
    """
    Simulate the get_initial_state endpoint that's slow.
    This mimics the pattern in your McpGym code.
    """
    print(f"🔍 get_initial_state called at {time.time()}")

    # Simulate the slow operation (like environment initialization)
    time.sleep(1)  # 1 second delay to test concurrency

    # Return a dummy initial state
    return JSONResponse({"observation": "dummy_initial_state", "session_id": "test_session", "timestamp": time.time()})


@mcp.tool
def dummy_tool() -> str:
    """Dummy tool for MCP compatibility."""
    return "dummy"


def main():
    """Run the test server."""
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting get_initial_state test server on port {port}")
    print(f"📡 Endpoint: http://localhost:{port}/control/initial_state")

    # Use FastMCP 2.0 run method with streamable-http transport
    mcp.run(transport="http", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
