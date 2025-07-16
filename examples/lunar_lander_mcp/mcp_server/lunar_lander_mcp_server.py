#!/usr/bin/env python3
"""
LunarLander MCP Server - Visual Environment Example

This MCP server provides access to the LunarLander environment from Gymnasium,
featuring visual rendering capabilities and image responses.

Features:
- **VISUAL ENVIRONMENT**: Supports rendered frames as base64 images
- **UNIFIED PATTERN**: Uses GymProductionServer base class
- Initial state provided via MCP resources including rendered frame
- Tools used for actions (lander_action), resources provide state/visual data
- Dynamic visual feedback showing lander position and physics
- Proper landing detection and reward handling
- OpenAI-compatible tool calling interface

MCP Integration:
- Initial state available through "game://initial_state" resource
- Visual frames available through "game://current_frame" resource
- Tools provide actions, resources provide state/configuration/visual data

Requirements:
- swig (for box2d compilation)
- gymnasium[box2d]
- pygame (for rendering)
"""

import argparse
import os
from typing import Any, Dict

from lunar_lander_adapter import LunarLanderAdapter
from mcp.server.fastmcp import Context

from reward_kit.mcp import GymProductionServer

# TODO: FAST FOLLOW. refactor this entire file to use McpGym, leaving logic below incorrect for now.

class LunarLanderProdServer(GymProductionServer):
    """LunarLander production server with visual rendering support."""

    def __init__(self, seed: int = None):
        super().__init__("LunarLander-v3", LunarLanderAdapter())
        if seed is not None:
            # The key change to make the environment reproducible
            self.env.reset(seed=seed)

    def _register_tools(self):
        """Register domain-specific tools."""

        @self.mcp.tool(
            name="lander_action",
            description="Control the lunar lander. Actions: NOTHING, FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT",
        )
        def lander_action(action: str, ctx: Context) -> Dict[str, Any]:
            """
            Control the lunar lander.

            Args:
                action: Action to take (NOTHING, FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT)

            Returns:
                Game state with position, velocity, rendered frame, reward, and completion status
            """
            # Validate action
            if not action or not isinstance(action, str):
                raise ValueError(
                    f"Invalid action parameter: '{action}'. "
                    f"Must be a non-empty string. Valid actions: NOTHING, FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT"
                )

            action = action.strip().upper()

            # Parse and execute action
            try:
                action_int = self.adapter.parse_action(action)
            except ValueError as e:
                raise ValueError(str(e))

            # Execute action
            obs, reward, terminated, truncated, info = self.adapter.step_environment(
                self.env, action_int
            )

            # Update global state
            self.obs = obs

            # Get status and render frame
            status = self.adapter.get_landing_status(obs, reward, terminated, truncated)
            rendered_frame = self.adapter.render_frame(self.env)

            # Log the result
            print(f"🚀 {action} → {status} (reward: {reward:.2f})")

            # Return complete state including visual frame
            result = {
                **self._render(obs),
                "action": action,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "status": status,
            }

            # Add rendered frame if available
            if rendered_frame:
                result["rendered_frame"] = rendered_frame

            # Add info if available
            if info:
                result["info"] = info

            return result

    def _register_resources(self):
        """Register MCP resources including visual frame."""
        super()._register_resources()

        @self.mcp.resource("game://current_frame")
        def current_frame() -> str:
            """Get the current rendered frame of the lunar lander."""
            if self.env is None:
                return "Environment not initialized"

            rendered_frame = self.adapter.render_frame(self.env)
            if rendered_frame:
                return f"Current lunar lander frame: {rendered_frame}"
            else:
                return "Frame rendering not available"

        @self.mcp.resource("game://action_space")
        def action_space() -> str:
            """Get information about available actions."""
            action_info = self.adapter.get_action_space_info()
            actions_desc = "\n".join(
                [f"{k}: {v}" for k, v in action_info["actions"].items()]
            )
            return f"LunarLander Action Space:\n{actions_desc}"

        @self.mcp.resource("game://observation_space")
        def observation_space() -> str:
            """Get information about the observation space."""
            obs_info = self.adapter.get_observation_space_info()
            obs_desc = "\n".join(
                [f"Index {k}: {v}" for k, v in obs_info["description"].items()]
            )
            return f"LunarLander Observation Space (8D vector):\n{obs_desc}"

    @staticmethod
    def format_observation(obs: Any, env: Any) -> Dict[str, Any]:
        """Format observation for MCP response."""
        # Use adapter to format the observation
        adapter = LunarLanderAdapter()
        formatted = adapter.format_observation(obs)

        # Add rendered frame
        rendered_frame = adapter.render_frame(env)
        if rendered_frame:
            formatted["rendered_frame"] = rendered_frame

        return formatted


def main():
    """Run the LunarLander MCP server."""
    parser = argparse.ArgumentParser(description="LunarLander MCP Server")
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
        help="Transport protocol to use",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP transport"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for the environment"
    )

    args = parser.parse_args()

    # Set environment variable for HTTP port
    if args.transport == "streamable-http":
        os.environ["PORT"] = str(args.port)

    # Create and run server
    server = LunarLanderProdServer(seed=args.seed)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()