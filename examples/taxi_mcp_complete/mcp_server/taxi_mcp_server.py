#!/usr/bin/env python3
"""
Taxi MCP Server - Production Implementation

This is the Taxi MCP server using the unified GymProductionServer framework.
Uses the new unified API as documented in the proposal.

Features:
- **UNIFIED PATTERN**: Uses GymProductionServer base class
- Initial state provided via MCP resources
- Tools used only for actions (taxi_move), not for getting initial state
- Dynamic grid layout showing taxi and passenger positions
- State decoding and human-readable descriptions
- OpenAI-compatible tool calling interface

MCP Integration:
- Initial state available through "game://initial_state" resource
- Tools provide actions, resources provide state/configuration data
"""

import argparse
import os
from typing import Any, Dict

from mcp.server.fastmcp import Context
from taxi_adapter import TaxiAdapter

from eval_protocol.mcp import GymProductionServer

# TODO: FAST FOLLOW. refactor this entire file to use McpGym, leaving logic below incorrect for now.

class TaxiProdServer(GymProductionServer):
    """Taxi production server using unified framework."""

    def __init__(self):
        super().__init__("Taxi-v3", TaxiAdapter())

    def _register_tools(self):
        """Register domain-specific tools."""

        @self.mcp.tool(
            name="taxi_move",
            description="Move and act in the Taxi game. Actions: SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF",
        )
        def taxi_move(action: str, ctx: Context) -> Dict[str, Any]:
            """
            Move or act in the Taxi game.

            Args:
                action: Action to take (SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF)

            Returns:
                Game state with position, reward, and completion status
            """
            # Extract seed from client info and reinitialize if needed
            self.extract_seed_from_context(ctx)
            # note for later, should be: session_data = self._get_or_create_session(ctx)

            # Validate action
            if not action or not isinstance(action, str):
                raise ValueError(
                    f"Invalid action parameter: '{action}'. "
                    f"Must be a non-empty string. Valid actions: {self.adapter.ACTION_NAMES}"
                )

            action = action.strip().upper()

            # Parse and execute action
            try:
                action_int = self.adapter.parse_action(action)
            except ValueError as e:
                raise ValueError(str(e))

            # Execute move
            obs, reward, terminated, truncated, info = self.adapter.step_environment(
                self.env, action_int
            )

            # Update global state
            self.obs = obs

            # Log the result
            if terminated or truncated:
                status = "🎉 SUCCESS!" if reward > 0 else "💀 FAILED!"
                print(f"🚕 Game ended: {status}")
            else:
                print(f"🚕 {action} → state {obs}")

            # Format response with taxi-specific data
            result = {
                **self._render(obs),
                "action": action,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }

            # Add decoded state information
            decoded = self.adapter.decode_state(obs)
            result["decoded_state"] = decoded
            result["state_description"] = self.adapter.get_state_description(obs)

            # Convert action mask to human-readable legal actions if available
            if info and "action_mask" in info:
                legal_actions = []
                blocked_actions = []
                for i, allowed in enumerate(info["action_mask"]):
                    action_name = self.adapter.ACTION_NAMES[i]
                    if allowed:
                        legal_actions.append(action_name)
                    else:
                        blocked_actions.append(action_name)
                result["legal_actions"] = legal_actions
                result["blocked_actions"] = blocked_actions

            # Add info if available
            if info:
                # Convert numpy arrays to lists for JSON serialization
                clean_info = {}
                for key, value in info.items():
                    if hasattr(value, "tolist"):  # numpy array
                        clean_info[key] = value.tolist()
                    else:
                        clean_info[key] = value
                result["info"] = clean_info

            return result

    @staticmethod
    def format_observation(obs: int, env: Any) -> Dict[str, Any]:
        """Format observation for MCP response."""
        # Create adapter instance to decode state
        adapter = TaxiAdapter()
        decoded = adapter.decode_state(obs)

        # Get grid layout if available
        grid_layout = "Position: " + str(obs)
        if hasattr(env, "desc") and env.desc is not None:
            grid_layout = adapter._get_visual_grid(obs, env)

        return {
            "state": int(obs),
            "grid_layout": grid_layout,
            "state_description": adapter.get_state_description(obs),
            "decoded_state": decoded,
        }


def main():
    """Run the Taxi MCP server."""
    parser = argparse.ArgumentParser(description="Taxi MCP Server")
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
        help="Transport protocol to use",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP transport"
    )

    args = parser.parse_args()

    # Set environment variable for HTTP port
    if args.transport == "streamable-http":
        os.environ["PORT"] = str(args.port)

    # Create and run server
    server = TaxiProdServer()
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()