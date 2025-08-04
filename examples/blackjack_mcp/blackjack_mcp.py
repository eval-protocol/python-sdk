"""
Blackjack MCP-Gym Implementation

This module implements the north star vision for MCP-Gym environments,
providing a clean, simple implementation of Blackjack using the McpGym base class.

Key Features:
- Multi-session support with session-based control plane state
- Data plane: Tool responses contain only observations
- Control plane: Server-side state management keyed by session ID
- Rollout system can query control plane state for termination logic

Example usage:
    from blackjack_mcp import BlackjackMcp

    server = BlackjackMcp(seed=42)
    server.run()
"""

from typing import Any, Dict, Optional, Tuple

from blackjack_adapter import BlackjackAdapter
from gymnasium.envs.toy_text.blackjack import BlackjackEnv
from mcp.server.fastmcp import Context

from eval_protocol.mcp import McpGym
from eval_protocol.mcp.mcpgym import control_plane_endpoint


class BlackjackMcp(McpGym):
    """
    Blackjack MCP-Gym environment implementing the north star vision.

    This demonstrates the clean, simple API for MCP-Gym environments:
    - Inherit from McpGym (which inherits from GymProductionServer)
    - Use proper EnvironmentAdapter pattern
    - Register tools with @self.mcp.tool() decorator
    - Compatible with CondaServerProcessManager
    - Multi-session support with session-based control plane state
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize Blackjack MCP-Gym environment."""
        adapter = BlackjackAdapter()
        super().__init__("Blackjack-v1", adapter, seed)

        # Multi-session support is now handled by the base class

    # Session management methods are now handled by the base class

    def _register_tools(self):
        """Register domain-specific MCP tools."""

        @self.mcp.tool(
            name="blackjack_move",
            description="Move on the blackjack. Actions: STICK, HIT. "
            "Returns only observation data; control plane state managed server-side.",
        )
        def blackjack_move(action: str, ctx: Context) -> Dict[str, Any]:
            """
            Move in the Blackjack environment.

            Args:
                action: Direction to move (STICK, HIT)
                ctx: MCP context (proper FastMCP context)

            Returns:
                Dictionary with observation data ONLY (data plane).
                Control plane state managed server-side per session.
            """
            # Validate action
            if not action or not isinstance(action, str):
                raise ValueError(
                    f"Invalid action parameter: '{action}'. " f"Must be a non-empty string. Valid actions: STICK, HIT"
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
            print(f"🎮 Session {session_id[:16]}...: {action} → state {session_data['obs']}")

            return observation_data

    def format_observation(self, obs: Tuple[int, int, int], env: BlackjackEnv) -> Dict[str, int]:
        """Format observation for MCP response (data plane only)."""
        return {
            "player_sum": obs[0],
            "dealer_card": obs[1],
            "usable_ace": obs[2],
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the Blackjack MCP-Gym environment
    print("Creating Blackjack MCP-Gym server...")
    server = BlackjackMcp(seed=42)

    print("Server created successfully!")
    print(f"Environment adapter: {server.adapter.__class__.__name__}")
    print("\n🎛️  Multi-session control plane features:")
    print("  - Session-based environment isolation")
    print("  - Server-side control plane state management")
    print("  - get_control_plane_state tool for rollout system")
    print("  - Data plane tools return observations only")

    # Run the server
    print("\nStarting MCP server...")
    server.run()
