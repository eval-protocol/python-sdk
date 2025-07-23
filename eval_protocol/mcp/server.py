"""
MCP Environment Server Framework

This provides a generic MCP server that can work with any environment
by using the EnvironmentAdapter interface. This is the framework code
that reward-kit provides to make MCP integration easy.
"""

import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Type

from mcp.server.fastmcp import Context, FastMCP

from .adapter import EnvironmentAdapter


class MCPEnvironmentServer:
    """
    Generic MCP server for environments.

    This class handles all MCP protocol concerns and delegates
    environment-specific logic to the provided EnvironmentAdapter.
    """

    def __init__(
        self,
        adapter: EnvironmentAdapter,
        server_name: str = "Environment-Server",
        default_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MCP environment server.

        Args:
            adapter: EnvironmentAdapter implementation for specific environment
            server_name: Name for the MCP server
            default_config: Default configuration for environment creation
        """
        self.adapter = adapter
        self.default_config = default_config or {}

        # Thread-safe session storage
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.Lock()

        # Create FastMCP server with lifespan management
        self.mcp = FastMCP(server_name, lifespan=self._lifespan)

        # Register tools
        self._register_tools()

    @asynccontextmanager
    async def _lifespan(self, app: FastMCP):
        """Lifespan context manager for the FastMCP app."""
        print(f"🚀 MCP {self.mcp.name} server starting...")
        action_desc = self.adapter.get_action_space_description()
        print(f"🎮 Action space: {action_desc}")

        yield  # Server is running

        # Cleanup on shutdown
        print("🧹 Cleaning up sessions on shutdown...")
        with self.session_lock:
            for session_id, session_data in self.sessions.items():
                env = session_data.get("env")
                if env:
                    try:
                        self.adapter.close_environment(env)
                    except Exception as e:
                        print(
                            f"⚠️ Error closing environment in session {session_id}: {e}"
                        )
            self.sessions.clear()
        print("✅ Server shutdown complete")

    def _get_session_id(self, ctx: Context) -> str:
        """Extract session ID from FastMCP Context."""
        session_obj = ctx.session
        session_id = f"mcp_session_{id(session_obj)}"
        return session_id

    def _get_or_create_session(
        self,
        ctx: Context,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        model_id: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Get or create session state for the current FastMCP session.

        This handles automatic session creation when tools are first called.
        """
        session_id = self._get_session_id(ctx)

        with self.session_lock:
            if session_id not in self.sessions:
                # Merge default config with provided config
                env_config = {**self.default_config}
                if config:
                    env_config.update(config)

                # Create environment via adapter
                env = self.adapter.create_environment(env_config)
                obs, info = self.adapter.reset_environment(env, seed=seed)

                self.sessions[session_id] = {
                    "env": env,
                    "model_id": model_id,
                    "seed": seed,
                    "config": env_config,
                    "created_at": time.time(),
                    "last_used": time.time(),
                    "initial_observation": self.adapter.format_observation(obs),
                    "session_id": session_id,
                }

                print(
                    f"🆕 Created session {session_id[:16]}... with config={env_config}, seed={seed}"
                )

            # Update last used time
            self.sessions[session_id]["last_used"] = time.time()

            return self.sessions[session_id]

    def _register_tools(self):
        """Register MCP tools using the adapter interface."""

        @self.mcp.tool()
        def get_initial_observation(ctx: Context) -> Dict[str, Any]:
            """
            Get the initial observation for the current session.

            This creates a new session automatically if one doesn't exist.

            Returns:
                Dict with initialObservation
            """
            session_data = self._get_or_create_session(ctx)
            return {"initialObservation": session_data["initial_observation"]}

        @self.mcp.tool()
        def step(action: str, ctx: Context) -> Dict[str, Any]:
            """
            Execute a step action in the environment.

            Args:
                action: Action string to execute

            Returns:
                Dict with observation, reward, terminated, truncated, info
            """
            # Get session
            session_data = self._get_or_create_session(ctx)
            env = session_data["env"]
            session_id = session_data["session_id"]

            # Parse action via adapter
            try:
                parsed_action = self.adapter.parse_action(action)
            except Exception as e:
                raise ValueError(f"Invalid action '{action}': {e}")

            # Execute step via adapter
            obs, reward, terminated, truncated, info = self.adapter.step_environment(
                env, parsed_action
            )

            # Update session timestamp
            with self.session_lock:
                session_data["last_used"] = time.time()

            # Format response
            result = {
                "observation": self.adapter.format_observation(obs),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info if info else {},
            }

            # Debug logging
            print(
                f"🎮 {session_id[:16]}...: {action} → obs={result['observation']} reward={reward}"
            )

            if terminated or truncated:
                result_text = "🏆 COMPLETED!" if reward > 0 else "💀 TERMINATED!"
                print(f"🏁 {session_id[:16]}...: Episode ended - {result_text}")

            return result

    def run(self, transport: str = "sse", **kwargs):
        """
        Run the MCP server.

        Args:
            transport: Transport protocol ("stdio", "sse", "streamable-http")
            **kwargs: Additional arguments passed to FastMCP.run()
        """
        print(f"📡 Starting MCP server with {transport} transport")
        print(f"🔧 Tools available: {list(self.mcp._tool_manager._tools.keys())}")
        print()

        # Run the FastMCP server
        self.mcp.run(transport, **kwargs)
