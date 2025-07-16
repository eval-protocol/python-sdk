"""
GymProductionServer Framework

This framework provides a base class for creating MCP servers that wrap gymnasium
environments using adapters. It handles:

1. Multi-session and single-session server lifecycle
2. Automatic tool and resource registration
3. Environment management via adapters
4. MCP resource patterns for initial state
5. Standardized tool signatures
6. Session management with proper seed extraction

Usage:
    class MyGameProdServer(GymProductionServer):
        def __init__(self):
            super().__init__("MyGame-v1", MyAdapter())

        def _register_tools(self):
            # Register domain-specific tools

        @staticmethod
        def format_observation(obs, env):
            # Format observations for MCP responses
"""

import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP

from .adapter import EnvironmentAdapter


class GymProductionServer(ABC):
    """
    Multi-session capable MCP server base class.

    Subclasses supply:
    • adapter - EnvironmentAdapter instance
    • _register_tools() - add ergonomic tools
    • format_observation(obs, env) - env-specific view dict
    """

    def __init__(self, name: str, adapter: EnvironmentAdapter):
        """
        Initialize production server.

        Args:
            name: Server name for MCP
            adapter: Environment adapter instance
        """
        self.adapter = adapter

        # Create FastMCP server
        self.mcp = FastMCP(
            name,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8000)),
        )

        # Register resources and tools
        self._register_resources()
        self._register_tools()


    def _register_resources(self):
        """Register standard MCP resources."""

        # REMOVED: game://initial_state MCP resource
        # This was not session-aware and caused all sessions to return identical initial state.
        #
        # Initial state is now provided by session-aware HTTP endpoint in McpGym:
        # - GET /control/initial_state (with mcp-session-id header)
        #
        # The connection manager has been updated to query this HTTP endpoint instead.

        # REMOVED: Control plane MCP resources (control://reward, control://status, control://info)
        # These were not session-aware and caused all sessions to return identical control plane state.
        #
        # Control plane data is now provided by session-aware HTTP endpoints in McpGym:
        # - GET /control/reward (with mcp-session-id header)
        # - GET /control/status (with mcp-session-id header)
        # - GET /control/info (with mcp-session-id header)
        #
        # The rollout system has been updated to query these HTTP endpoints instead.
        pass

    # Abstract methods that subclasses must implement

    @abstractmethod
    def _register_tools(self):
        """Register domain-specific MCP tools."""
        pass

    @staticmethod
    @abstractmethod
    def format_observation(obs: Any, env: Any) -> Dict[str, Any]:
        """Format observation for MCP response."""
        pass

    def run(self, transport: str = "streamable-http", **kwargs):
        """Run the production server."""
        print(f"🚀 {self.mcp.name} Production Server Starting...")
        print(f"📡 Transport: {transport}")
        print("🎯 MCP Pattern: Resources for initial state, tools for actions")
        print("🔗 Initial state resource: game://initial_state")

        # Run the server
        self.mcp.run(transport=transport, **kwargs)