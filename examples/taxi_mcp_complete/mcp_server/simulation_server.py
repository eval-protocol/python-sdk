"""
Taxi Simulation MCP Server - Refactored Version

This simulation server implements Taxi using the unified SimulationServerBase
framework, following the proper MCP patterns.

Usage:
    python simulation_server_new.py --port 8000 --host 0.0.0.0
"""

import argparse
import json
from typing import Any, Dict

from taxi_adapter import TaxiAdapter
from taxi_mcp_server import TaxiProdServer

from eval_protocol.mcp import SimulationServerBase
from eval_protocol.mcp.simulation_server import simulation_resource, simulation_tool


class TaxiSimServer(TaxiAdapter, SimulationServerBase):
    """Taxi simulation server using unified framework."""

    def __init__(self):
        # Create production server for validation
        prod_server = TaxiProdServer()

        # Initialize simulation server
        SimulationServerBase.__init__(
            self, "Taxi-Simulation", production_server_app=prod_server.mcp
        )
        TaxiAdapter.__init__(self)

    @simulation_tool
    def taxi_move(self, action: str, *, ctx, session_state) -> Dict[str, Any]:
        """
        Move or act in the Taxi game - simulation version.

        This matches the production server signature exactly.
        """
        # Validate action
        if not action or not isinstance(action, str):
            raise ValueError(
                f"Invalid action parameter: '{action}'. "
                f"Must be a non-empty string. Valid actions: {self.ACTION_NAMES}"
            )

        action = action.strip().upper()

        # Parse and execute action
        try:
            action_int = self.parse_action(action)
        except ValueError as e:
            raise ValueError(str(e))

        # Execute step
        env = session_state["env"]
        obs, reward, terminated, truncated, info = self.step_environment(
            env, action_int
        )

        # Update session state
        session_state["steps"] += 1
        session_state["total_reward"] = session_state.get("total_reward", 0.0) + reward

        # Format response to match production server exactly
        result = {
            "state": int(self.format_observation(obs)),
            "grid_layout": self._get_visual_grid(obs, env),
            "state_description": self.get_state_description(obs),
            "action": action,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

        # Add decoded state information
        decoded = self.decode_state(obs)
        result["decoded_state"] = decoded

        # Convert action mask to human-readable legal actions if available
        if info and "action_mask" in info:
            legal_actions = []
            blocked_actions = []
            for i, allowed in enumerate(info["action_mask"]):
                action_name = self.ACTION_NAMES[i]
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

    @simulation_resource("game://initial_state")
    def initial_state(self, *, ctx, session_state) -> str:
        """Get initial state resource for simulation."""
        env = session_state["env"]
        initial_observation = session_state["initial_observation"]

        initial_state = {
            "state": int(initial_observation),
            "grid_layout": self._get_visual_grid(initial_observation, env),
            "state_description": self.get_state_description(initial_observation),
            "moves": 0,
            "terminated": False,
            "truncated": False,
            "reward": 0.0,
            "decoded_state": self.decode_state(initial_observation),
            "info": {
                "grid_size": "5x5",
                "initial_position": int(initial_observation),
                "seed": session_state.get("seed", 42),
            },
        }
        return json.dumps(initial_state)


def main():
    """Main entry point for Taxi simulation server."""
    parser = argparse.ArgumentParser(description="Taxi Simulation Server")
    parser.add_argument("--port", default=8001, type=int, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")

    args = parser.parse_args()

    print(f"🚀 Starting Taxi Simulation Server")
    print(f"🌐 Host: {args.host}")
    print(f"🌐 Port: {args.port}")
    print("🎯 Framework: Unified SimulationServerBase")
    print()

    # Set port environment variable
    import os

    os.environ["FASTMCP_PORT"] = str(args.port)

    # Create and run server
    server = TaxiSimServer()
    server.run()


if __name__ == "__main__":
    main()
