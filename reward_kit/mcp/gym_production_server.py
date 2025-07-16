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

        # Multi-session support
        self.sessions = (
            {}
        )  # session_id -> {"env": env, "obs": obs, "session_data": data}
        self.session_lock = threading.Lock()

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

    def _get_session_id(self, ctx: Context) -> str:
        """Extract session ID from MCP context using proper FastMCP pattern."""
        print(f"🔍 _get_session_id: Starting session ID extraction")
        print(f"🔍 _get_session_id: ctx type: {type(ctx)}")
        print(f"🔍 _get_session_id: hasattr(ctx, 'session'): {hasattr(ctx, 'session')}")

        # Use stable session ID based on client info (following simulation_server.py pattern)
        if hasattr(ctx, "session") and hasattr(ctx.session, "client_params"):
            client_params = ctx.session.client_params
            print(f"🔍 _get_session_id: client_params type: {type(client_params)}")
            print(
                f"🔍 _get_session_id: hasattr(client_params, 'clientInfo'): {hasattr(client_params, 'clientInfo')}"
            )

            if hasattr(client_params, "clientInfo"):
                client_info = client_params.clientInfo
                print(f"🔍 _get_session_id: client_info: {client_info}")
                print(
                    f"🔍 _get_session_id: hasattr(client_info, '_extra'): {hasattr(client_info, '_extra')}"
                )

                if client_info and hasattr(client_info, "_extra"):
                    extra_data = client_info._extra
                    print(f"🔍 _get_session_id: extra_data: {extra_data}")
                    print(f"🔍 _get_session_id: extra_data type: {type(extra_data)}")

                    if extra_data and isinstance(extra_data, dict):
                        # Create a stable session ID based on seed and other config
                        import hashlib
                        import json

                        seed_value = extra_data.get("seed")
                        config_value = extra_data.get("config", {})

                        print(
                            f"🔍 _get_session_id: seed_value: {seed_value} (type: {type(seed_value)})"
                        )
                        print(f"🔍 _get_session_id: config_value: {config_value}")

                        stable_data = {
                            "seed": seed_value,
                            "config": config_value,
                            "name": client_info.name,
                            "version": client_info.version,
                        }

                        print(f"🔍 _get_session_id: stable_data: {stable_data}")
                        stable_str = json.dumps(stable_data, sort_keys=True)
                        session_id = hashlib.md5(stable_str.encode()).hexdigest()
                        print(
                            f"🎯 Generated stable session_id: {session_id} for seed: {seed_value}"
                        )
                        return session_id

        # Fallback for testing or other scenarios
        session_id = f"gym_{id(ctx)}"
        print(f"🎯 Generated fallback session_id: {session_id}")
        return session_id

    def _get_or_create_session(self, ctx: Context) -> Dict[str, Any]:
        """Get or create session data for the given context."""
        session_id = self._get_session_id(ctx)
        print(f"🔍 _get_or_create_session: session_id: {session_id}")

        with self.session_lock:
            if session_id not in self.sessions:
                print(
                    f"🔍 _get_or_create_session: Creating new session for {session_id}"
                )
                # Extract seed from context using proper FastMCP pattern
                seed = None
                config = self.adapter.get_default_config()
                print(f"🔍 _get_or_create_session: default_config: {config}")

                if hasattr(ctx, "session") and hasattr(ctx.session, "client_params"):
                    client_params = ctx.session.client_params
                    if hasattr(client_params, "clientInfo"):
                        client_info = client_params.clientInfo
                        if client_info and hasattr(client_info, "_extra"):
                            extra_data = client_info._extra
                            print(
                                f"🔍 _get_or_create_session: extra_data in session creation: {extra_data}"
                            )
                            if extra_data and isinstance(extra_data, dict):
                                # Extract seed from client info
                                seed = extra_data.get("seed")
                                print(
                                    f"🌱 Extracted seed from client_info: {seed} (type: {type(seed)})"
                                )
                                # Update config with any additional options
                                if "config" in extra_data:
                                    config.update(extra_data["config"])
                                    print(
                                        f"🔍 _get_or_create_session: updated config: {config}"
                                    )

                print(
                    f"🔍 _get_or_create_session: About to create environment with seed: {seed}"
                )

                # Create environment with seed
                if seed is not None:
                    print(
                        f"🔍 _get_or_create_session: Calling create_environment_with_seed({config}, seed={seed})"
                    )
                    env, obs, info = self.adapter.create_environment_with_seed(
                        config, seed=seed
                    )
                    print(
                        f"🔍 _get_or_create_session: create_environment_with_seed returned obs: {obs}, info: {info}"
                    )
                else:
                    print(
                        f"🔍 _get_or_create_session: Calling create_environment({config}) + reset_environment(seed={seed})"
                    )
                    env = self.adapter.create_environment(config)
                    obs, info = self.adapter.reset_environment(env, seed=seed)
                    print(
                        f"🔍 _get_or_create_session: create_environment + reset returned obs: {obs}, info: {info}"
                    )

                # Initialize session state
                self.sessions[session_id] = {
                    "env": env,
                    "obs": obs,
                    "session_data": {},  # Subclasses can store additional data here
                    "session_id": session_id,
                }

                print(
                    f"🎮 Created new session {session_id[:16]}... with seed {seed}, initial obs: {obs}"
                )
            else:
                print(
                    f"🔍 _get_or_create_session: Returning existing session {session_id}"
                )

            return self.sessions[session_id]

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