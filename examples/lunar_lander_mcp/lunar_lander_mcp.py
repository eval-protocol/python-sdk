#!/usr/bin/env python3
"""
LunarLander MCP-Gym Implementation

This module implements the north star vision for MCP-Gym environments,
providing a clean, simple implementation of LunarLander using the McpGym base class.

Key Features:
- Multi-session support with session-based control plane state
- Data plane: Tool responses contain only observations and visual frames
- Control plane: Server-side state management keyed by session ID
- Rollout system can query control plane state for termination logic
- Visual rendering with base64 encoded frames

Example usage:
    from lunar_lander_mcp import LunarLanderMcp

    server = LunarLanderMcp(seed=42)
    server.run()
"""

import argparse
import os
from re import A
from typing import Any, Dict, Optional, List
import json

from lunar_lander_adapter import LunarLanderAdapter
from mcp.server.fastmcp import Context

from reward_kit.mcp import McpGym
from reward_kit.mcp.mcpgym import control_plane_endpoint


class LunarLanderMcp(McpGym):
    """LunarLander production server with visual rendering support."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize LunarLander MCP-Gym environment."""
        self.adapter = LunarLanderAdapter()
        super().__init__("LunarLander-v3", self.adapter, seed)

        # Multi-session support is now handled by the base class

    def _register_tools(self):
        """Register domain-specific MCP tools."""

        @self.mcp.tool(
            name="lander_action",
            description="Control the lunar lander with discrete actions. "
            "Valid actions: NOTHING, FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT. "
            "NOTHING: Do nothing, FIRE_LEFT: Fire left orientation engine, "
            "FIRE_MAIN: Fire main engine, FIRE_RIGHT: Fire right orientation engine.",
        )
        def lander_action(action: str, ctx: Context) -> Dict[str, Any]:
            """
            Execute a discrete action for the lunar lander.

            Args:
                action: Action to execute (NOTHING, FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT)
                ctx: MCP context

            Returns:
                Environment observation after executing the action
            """
            # Validate parameter
            if not isinstance(action, str):
                raise ValueError(
                    f"Invalid action type: '{type(action)}'. Must be a string."
                )

            # Parse action using adapter
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

            return observation_data


    def format_observation(self, obs: Any, env: Any) -> Dict[str, Any]:
        """Format observation for MCP response (data plane only)."""
        # Use the existing adapter instance instead of creating a new one
        formatted = self.adapter.format_observation(obs)

        # Add rendered frame (this is part of data plane - visual observation)
        rendered_frame = self.adapter.render_frame(env)
        
        if rendered_frame:
            formatted["image_url"] = {
                "url": rendered_frame  # Note: OpenAI format allows data URI, i.e. data:image/png;base64,<base64_data>
            }
        return formatted
