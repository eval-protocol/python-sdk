#!/usr/bin/env python3
"""
Airline Adapter for τ²-Bench Integration

This adapter handles the specific mechanics of the τ²-Bench airline domain
including reservation management, flight search, and policy enforcement.
"""

import json
from typing import Any, Dict, Optional, Tuple, List

from reward_kit.mcp.adapter import EnvironmentAdapter
from airplane_environment.airline_environment import AirlineEnvironment

# TODO: Open question: 
# I know we said we shouldn't use adapter when we can define the environment, but mcpgym.py relies on self.adapter.step_environment() and so on. Should that be refactored? I can make that change + remove tau2_adapter.py in a fast follow.

class AirlineAdapter(EnvironmentAdapter):
    """τ²-Bench Airline domain adapter"""

    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create and configure the τ²-Bench airline environment."""
        env_config = self.get_default_config()
        if config:
            env_config.update(config)
        
        env = AirlineEnvironment(config=env_config)
        return env
    
    def create_environment_with_seed(self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None) -> Tuple[Any, int, Dict[str, Any]]:
        """Create environment instance with a deterministic seed and return env, seed, info."""
        env = self.create_environment(config)
        obs, info = env.reset(seed=seed)

        return env, obs, info

    def reset_environment(
        self, env: Any, seed: Optional[int] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state."""
        return env.reset(seed=seed)
    
    def step_environment(
        self, env: Any, action: Dict[str, Any]
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the airline environment."""        
        try:
            return env.step(action)        
        except Exception as e:
            print(f"Error in step_environment: {e}")
            raise e

    def close_environment(self, env: Any) -> None:
        """Clean up environment resources."""
        env.close()

    def parse_action(self, action_str: str) -> Dict[str, Any]:
        """Parse action string from MCP tool call into environment action."""
        return json.loads(action_str)

    def format_observation(self, obs: Any) -> Dict[str, Any]:
        """Format observation for MCP tool responses."""
        return obs

    def get_action_space_description(self) -> Dict[str, Any]:
        """Get description of valid actions for airline environment."""
        return {
            "type": "airline_tools",
            "description": "Airline booking and management tools",
            "available_actions": [
                "get_user_details",
                "get_reservation_details", 
                "cancel_reservation",
                "book_reservation",
                "search_direct_flight",
                "search_onestop_flight",
                "update_reservation_flights",
                "update_reservation_passengers", 
                "update_reservation_baggages",
                "list_all_airports",
                "get_flight_status",
                "send_certificate",
                "transfer_to_human_agents",
                "calculate"
            ]
        }

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for airline environment."""
        return {
            "domain": "airline",
            "max_turns": 20,
        }
