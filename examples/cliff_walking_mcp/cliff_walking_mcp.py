"""
Cliff Walking MCP-Gym Implementation

This module implements the north star vision for MCP-Gym environments,
providing a clean, simple implementation of Cliff Walking using the McpGym base class.

Key Features:
- Multi-session support with session-based control plane state
- Data plane: Tool responses contain only observations
- Control plane: Server-side state management keyed by session ID
- Rollout system can query control plane state for termination logic

Example usage:
    from cliff_walking_mcp import CliffWalkingMcp

    server = CliffWalkingMcp(seed=42)
    server.run()
"""

from typing import Any, Dict, Optional

from cliff_walking_adapter import CliffWalkingAdapter
from mcp.server.fastmcp import Context
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv

from reward_kit.mcp import McpGym


class CliffWalkingMcp(McpGym):
    """
    Cliff Walking MCP-Gym environment implementing the north star vision.

    This demonstrates the clean, simple API for MCP-Gym environments:
    - Inherit from McpGym (which inherits from GymProductionServer)
    - Use proper EnvironmentAdapter pattern
    - Register tools with @self.mcp.tool() decorator
    - Compatible with CondaServerProcessManager
    - Multi-session support with session-based control plane state
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize Cliff Walking MCP-Gym environment."""
        adapter = CliffWalkingAdapter()
        super().__init__("CliffWalking-v1", adapter, seed)

        # Multi-session support is now handled by the base class

    # Session management methods are now handled by the base class

    def _register_tools(self):
        """Register domain-specific MCP tools."""

        @self.mcp.tool(
            name="cliff_move",
            description="Move on the cliff walking. Actions: UP, RIGHT, DOWN, LEFT. "
            "Returns only observation data; control plane state managed server-side.",
        )
        def cliff_move(action: str, ctx: Context) -> Dict[str, Any]:
            """
            Move in the Cliff Walking environment.

            Args:
                action: Direction to move (UP, RIGHT, DOWN, LEFT)
                ctx: MCP context (proper FastMCP context)

            Returns:
                Dictionary with observation data ONLY (data plane).
                Control plane state managed server-side per session.
            """
            # Validate action
            if not action or not isinstance(action, str):
                raise ValueError(
                    f"Invalid action parameter: '{action}'. "
                    f"Must be a non-empty string. Valid actions: UP, RIGHT, DOWN, LEFT"
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
            observation_data = self._execute_session_environment_step(
                session_id, action_int
            )
            observation_data["action"] = action

            # Log move (no control plane data in logs)
            print(
                f"🎮 Session {session_id[:16]}...: {action} → position {session_data['obs']}"
            )

            return observation_data

    @staticmethod
    def format_observation(obs: int, env: CliffWalkingEnv) -> Dict[str, Any]:
        """Format observation for MCP response (data plane only)."""
        return {
            "position": int(obs),
            "grid": env.render(),
        }
