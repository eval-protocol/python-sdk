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


class AirlineAdapter(EnvironmentAdapter):
    """τ²-Bench Airline domain adapter"""

    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create and configure the τ²-Bench airline environment."""
        env_config = self.get_default_config()
        if config:
            env_config.update(config)
        
        env = AirlineEnvironment(**env_config)
        return env

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
        # TODO: Implement this
        return obs

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for airline environment."""
        return {
            "domain": "airline",
            "max_turns": 20,
        }
