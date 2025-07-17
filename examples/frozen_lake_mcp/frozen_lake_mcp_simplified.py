"""
FrozenLake MCP-Gym Implementation - Simplified Version

This module implements the north star vision for MCP-Gym environments,
providing a clean, simple implementation of FrozenLake WITHOUT the adapter pattern.

Key Features:
- Strict data/control plane separation
- Data plane: Tool responses contain only observations
- Control plane: Rewards/termination available via MCP resources (control://reward, control://status)
- Direct environment handling (no adapter layer)
- <50% complexity of adapter-based version

Example usage:
    from frozen_lake_mcp_simplified import FrozenLakeMcpSimplified

    server = FrozenLakeMcpSimplified(seed=42)
    server.run()
"""

import json
from typing import Any, Dict, Optional

from gymnasium.envs.toy_text import FrozenLakeEnv
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from mcp.server.fastmcp import Context, FastMCP


class FrozenLakeMcpSimplified:
    """
    Simplified FrozenLake MCP-Gym environment implementing the north star vision.

    This demonstrates the clean, simple API for MCP-Gym environments WITHOUT adapters:
    - Direct environment handling (no adapter pattern)
    - Built-in action parsing and observation formatting
    - Register tools with @self.mcp.tool() decorator
    - Compatible with CondaServerProcessManager
    - Strict data/control plane separation via MCP resources
    """

    ACTION_NAMES = ["LEFT", "DOWN", "RIGHT", "UP"]

    def __init__(self, seed: Optional[int] = None, grid_size: int = 4):
        """Initialize simplified FrozenLake MCP-Gym environment."""
        self.grid_size = grid_size

        # Create environment directly
        self.env = self._create_environment(seed)
        self.obs, _info = self.env.reset(seed=seed)

        # Initialize control plane state
        self.control_plane_state = {
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {},
            "step_count": 0,
            "total_reward": 0.0,
        }

        # Create FastMCP server
        import os

        self.mcp = FastMCP(
            "FrozenLake-Simplified-v1",
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8000)),
        )

        # Register resources and tools
        self._register_control_plane_resources()
        self._register_standard_resources()
        self._register_tools()

    def _create_environment(self, seed: Optional[int] = None) -> FrozenLakeEnv:
        """Create FrozenLake environment directly."""
        if seed is not None:
            desc = generate_random_map(size=self.grid_size, p=0.8, seed=seed)
        else:
            desc = generate_random_map(size=self.grid_size, p=0.8)

        return FrozenLakeEnv(desc=desc, is_slippery=False, render_mode="ansi")

    def _register_control_plane_resources(self):
        """Register MCP resources for control plane information."""

        @self.mcp.resource("control://reward")
        def current_reward() -> str:
            """Provide current reward information via MCP resource."""
            return json.dumps(
                {
                    "reward": self.control_plane_state["reward"],
                    "step_count": self.control_plane_state["step_count"],
                }
            )

        @self.mcp.resource("control://status")
        def current_status() -> str:
            """Provide current episode status via MCP resource."""
            return json.dumps(
                {
                    "terminated": self.control_plane_state["terminated"],
                    "truncated": self.control_plane_state["truncated"],
                    "step_count": self.control_plane_state["step_count"],
                    "total_reward": self.control_plane_state["total_reward"],
                }
            )

        @self.mcp.resource("control://info")
        def current_info() -> str:
            """Provide current environment info via MCP resource."""
            return json.dumps(self.control_plane_state["info"])

    def _register_standard_resources(self):
        """Register standard MCP resources."""

        @self.mcp.resource("game://initial_state")
        def initial_state() -> str:
            """Provide initial game state as MCP resource."""
            return json.dumps(self._format_observation(self.obs))

    def _register_tools(self):
        """Register domain-specific MCP tools."""

        @self.mcp.tool(
            name="lake_move",
            description="Move on the frozen lake. Actions: LEFT, DOWN, RIGHT, UP. "
            "Check control://reward and control://status resources for rewards and termination.",
        )
        def lake_move(action: str, ctx: Context) -> Dict[str, Any]:
            """
            Move in the FrozenLake environment.

            Args:
                action: Direction to move (LEFT, DOWN, RIGHT, UP)
                ctx: MCP context (proper FastMCP context)

            Returns:
                Dictionary with observation data ONLY (data plane).
                Rewards and termination info available via control plane resources.
            """
            # Validate action
            if not action or not isinstance(action, str):
                raise ValueError(
                    f"Invalid action parameter: '{action}'. "
                    f"Must be a non-empty string. Valid actions: LEFT, DOWN, RIGHT, UP"
                )

            action = action.strip().upper()

            # Parse action (directly without adapter)
            if action not in self.ACTION_NAMES:
                raise ValueError(
                    f"Invalid action '{action}'. Valid actions: {self.ACTION_NAMES}"
                )
            action_int = self.ACTION_NAMES.index(action)

            # Execute environment step with control plane separation
            observation_data = self._execute_environment_step(action_int)

            # Add the action to the response for context
            observation_data["action"] = action

            # Log the result (control plane state is already logged in _execute_environment_step)
            if (
                self.control_plane_state["terminated"]
                or self.control_plane_state["truncated"]
            ):
                status = (
                    "🏆 GOAL!" if self.control_plane_state["reward"] > 0 else "💀 HOLE!"
                )
                print(f"🎮 Game ended: {status}")
            else:
                print(f"🎮 {action} → position {self.obs}")

            # Return ONLY data plane information (no rewards/termination)
            return observation_data

    def _execute_environment_step(self, action_int: int) -> Dict[str, Any]:
        """
        Execute environment step and update control plane (directly without adapter).

        Args:
            action_int: Parsed action integer

        Returns:
            Data plane response (observation only, no rewards)
        """
        # Execute environment step directly
        obs, reward, terminated, truncated, info = self.env.step(action_int)

        # Update global observation state
        self.obs = obs

        # Update control plane (separate from data plane)
        self._update_control_plane(reward, terminated, truncated, info)

        # Return ONLY data plane information (no rewards/termination)
        return self._format_observation(obs)

    def _update_control_plane(
        self, reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]
    ):
        """
        Update control plane state after environment step.

        Args:
            reward: Reward from environment step
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Info dictionary from environment
        """
        self.control_plane_state["reward"] = reward
        self.control_plane_state["terminated"] = terminated
        self.control_plane_state["truncated"] = truncated
        self.control_plane_state["info"] = info
        self.control_plane_state["step_count"] += 1
        self.control_plane_state["total_reward"] += reward

        # Log control plane update (for debugging)
        print(
            f"🎛️  Control plane updated: reward={reward}, terminated={terminated}, step={self.control_plane_state['step_count']}"
        )

    def _format_observation(self, obs: int) -> Dict[str, Any]:
        """Format observation for MCP response (data plane only, directly without adapter)."""
        return {
            "position": int(obs),
            "grid": self.env.render(),
        }

    def run(self, transport: str = "streamable-http", **kwargs):
        """Run the MCP server."""
        print(f"🚀 {self.mcp.name} Simplified Server Starting...")
        print(f"📡 Transport: {transport}")
        print("🎯 MCP Pattern: Resources for initial state, tools for actions")
        print("🔗 Initial state resource: game://initial_state")
        print("\n🎛️  Control plane resources available:")
        print("  - control://reward (current reward and step count)")
        print("  - control://status (termination status and total reward)")
        print("  - control://info (environment info)")
        print(f"\n📊 Environment: {self.grid_size}x{self.grid_size} FrozenLake")
        print(f"📍 Initial position: {self.obs}")

        # Run the server
        self.mcp.run(transport=transport, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test the simplified FrozenLake MCP-Gym environment
    print("Creating simplified FrozenLake MCP-Gym server...")
    server = FrozenLakeMcpSimplified(seed=42)

    print("Server created successfully!")
    print(f"Initial observation: {server.obs}")
    print("🎯 NO ADAPTER PATTERN - direct environment handling")

    # Run the server
    print("\nStarting MCP server...")
    server.run()
