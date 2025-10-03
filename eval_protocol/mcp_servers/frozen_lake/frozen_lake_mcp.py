"""
FrozenLake MCP-Gym Implementation

This module implements the north star vision for MCP-Gym environments,
providing a clean, simple implementation of FrozenLake using the McpGym base class.

Key Features:
- Multi-session support with session-based control plane state
- Data plane: Tool responses contain only observations
- Control plane: Server-side state management keyed by session ID
- Rollout system can query control plane state for termination logic

Example usage:
    from frozen_lake_mcp import FrozenLakeMcp

    server = FrozenLakeMcp(seed=42)
    server.run()
"""

from typing import Any, Dict, Optional

from frozen_lake_adapter import FrozenLakeAdapter
from mcp.server.fastmcp import Context

from eval_protocol.mcp import McpGym


class FrozenLakeMcp(McpGym):
    """
    FrozenLake MCP-Gym environment implementing the north star vision.

    This demonstrates the clean, simple API for MCP-Gym environments:
    - Inherit from McpGym (which inherits from GymProductionServer)
    - Use proper EnvironmentAdapter pattern
    - Register tools with @self.mcp.tool() decorator
    - Compatible with CondaServerProcessManager
    - Multi-session support with session-based control plane state
    """

    def __init__(self, seed: Optional[int] = None, **kwargs):
        """Initialize FrozenLake MCP-Gym environment."""
        adapter = FrozenLakeAdapter()
        super().__init__("FrozenLake-v1", adapter, seed, **kwargs)

        # Multi-session support is now handled by the base class

    # Session management methods are now handled by the base class

    def _register_tools(self):
        """Register domain-specific MCP tools."""

        @self.mcp.tool(
            name="lake_move",
            description="Move on the frozen lake. Actions: LEFT, DOWN, RIGHT, UP. "
            "Returns only observation data; control plane state managed server-side.",
        )
        def lake_move(action: str, ctx: Context) -> Dict[str, Any]:
            """
            Move in the FrozenLake environment.

            Args:
                action: Direction to move (LEFT, DOWN, RIGHT, UP)
                ctx: MCP context (proper FastMCP context)

            Returns:
                Dictionary with observation data ONLY (data plane).
                Control plane state managed server-side per session.
            """
            # Validate action
            if not action or not isinstance(action, str):
                raise ValueError(
                    f"Invalid action parameter: '{action}'. "
                    f"Must be a non-empty string. Valid actions: LEFT, DOWN, RIGHT, UP"
                )

            action = action.strip().upper()

            # Parse action
            try:
                action_int = self.adapter.parse_action(action)
            except ValueError as e:
                raise ValueError(str(e))

            # Get session ID and session data
            session_id = self._get_session_id(ctx)
            session_data = self._get_or_create_session(ctx)

            # Execute environment step using base class method
            observation_data = self._execute_session_environment_step(session_id, action_int)
            observation_data["action"] = action

            # Log move (no control plane data in logs)
            print(f"🎮 Session {session_id[:16]}...: {action} → position {session_data['obs']}")

            return observation_data

    def format_observation(self, obs: int, env: Any) -> Dict[str, Any]:
        """Format observation for MCP response (data plane only)."""
        return {
            "position": int(obs),
            "grid": env.render(),
        }
