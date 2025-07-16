"""
MCP-Gym Framework - North Star Implementation

This module provides the core McpGym base class that implements the north star vision
for universal RL environment integration via MCP protocol.

Key Features:
- Inherits from GymProductionServer for proper MCP integration
- Simple tool registration with @self.mcp.tool() decorator
- Clean separation between data plane (MCP tool calls) and control plane (custom endpoints)
- Compatible with CondaServerProcessManager
- Session-aware control plane endpoints via @control_plane_endpoint decorator
"""

import inspect
import json
import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

from mcp.server.fastmcp import Context
from starlette.requests import Request
from starlette.responses import JSONResponse

from .adapter import EnvironmentAdapter
from .gym_production_server import GymProductionServer

logger = logging.getLogger(__name__)


def control_plane_endpoint(path: str) -> Callable:
    """
    Decorator to register session-aware control plane endpoints.

    Control plane endpoints provide rewards, termination status, and other
    metadata without polluting the tool namespace used by LLMs.

    Args:
        path: URL path for the endpoint (e.g., "/control/reward")

    Example:
        @control_plane_endpoint("/control/reward")
        def get_reward(self, ctx: Context, session_data: Dict[str, Any]) -> Dict[str, Any]:
            control_plane = session_data.get("control_plane", {})
            return {
                "reward": control_plane.get("reward", 0.0),
                "step_count": control_plane.get("step_count", 0)
            }
    """

    def decorator(func: Callable) -> Callable:
        func._is_control_plane_endpoint = True
        func._control_plane_path = path
        return func

    return decorator


class McpGym(GymProductionServer):
    """
    Base class for MCP-Gym environments implementing the north star vision.

    This class provides the universal adapter pattern for RL environments,
    bridging training infrastructure, production MCP standards, and high-quality
    environments through a clean, standardized interface.

    Key Design Principles:
    - Data Plane: JSON tool calls/responses via MCP (state transitions/actions)
    - Control Plane: Rewards/termination signals via MCP resources
    - Environment Implementation: Single-process MCP server per environment
    - Inherits from GymProductionServer for proper MCP protocol handling
    """

    def __init__(
        self, server_name: str, adapter: EnvironmentAdapter, seed: Optional[int] = None
    ):
        """
        Initialize MCP-Gym environment.

        Args:
            server_name: Name for the MCP server
            adapter: Environment adapter instance
            seed: Optional seed for reproducible environments
        """
        super().__init__(server_name, adapter)

        # Control plane endpoints dictionary
        self._control_plane_endpoints: Dict[str, Callable] = {}

        # Initialize control plane state (for backward compatibility - single session)
        self.control_plane_state = {
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {},
            "step_count": 0,
            "total_reward": 0.0,
        }

        # Reset with seed if provided
        self.env, self.obs, _info = self._new_env(seed=seed)

        # Discover and register control plane endpoints
        self._discover_and_register_control_plane_endpoints()

    def _discover_and_register_control_plane_endpoints(self):
        """
        Discover and register control plane endpoints on the subclass instance.

        This scans for methods decorated with @control_plane_endpoint and
        registers them as FastMCP custom routes with session awareness.
        """
        # 1. Discover control plane endpoints on the subclass instance
        discovered_endpoints = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_is_control_plane_endpoint"):
                discovered_endpoints[method.__name__] = method
        self._control_plane_endpoints = discovered_endpoints

        # 2. Register the discovered endpoints as FastMCP custom routes
        for endpoint_name, endpoint_func in discovered_endpoints.items():
            path = endpoint_func._control_plane_path

            # Create session-aware handler for this endpoint
            def create_endpoint_handler(func: Callable):
                async def endpoint_handler(request: Request) -> JSONResponse:
                    try:
                        # Extract session ID from request headers (similar to StreamableHTTP pattern)
                        session_id = request.headers.get("mcp-session-id")
                        if not session_id:
                            return JSONResponse(
                                {"error": "Missing mcp-session-id header"},
                                status_code=400,
                            )

                        # Get or create session data
                        with self.session_lock:
                            session_data = self.sessions.get(session_id)
                            if not session_data:
                                # For initial state endpoint, we need to create the session
                                # based on the session ID and available information
                                if func.__name__ == "get_initial_state_endpoint":
                                    # Create session with extracted seed from session ID
                                    session_data = self._create_session_from_id(
                                        session_id
                                    )
                                else:
                                    return JSONResponse(
                                        {"error": f"Session {session_id} not found"},
                                        status_code=404,
                                    )

                        # Call the endpoint function with session data
                        if inspect.iscoroutinefunction(func):
                            result = await func(session_data=session_data)
                        else:
                            result = func(session_data=session_data)

                        return JSONResponse(result)

                    except Exception as e:
                        return JSONResponse({"error": str(e)}, status_code=500)

                return endpoint_handler

            # Register the custom route
            handler = create_endpoint_handler(endpoint_func)
            self.mcp.custom_route(path, methods=["GET"])(handler)

        if discovered_endpoints:
            logger.info(
                f"✅ Registered {len(discovered_endpoints)} session-aware control plane endpoints"
            )
            for name, endpoint in discovered_endpoints.items():
                logger.info(f"  - {name}: {endpoint._control_plane_path}")
        else:
            logger.info("⚠️  No session-aware control plane endpoints discovered")

    def _create_session_from_id(self, session_id: str) -> Dict[str, Any]:
        """
        Create a session based on session ID when the initial state endpoint is called.

        The session ID is a hash of seed + config, so we can't extract the original values.
        Instead, we'll create a session with default values and let the tool calls handle
        the proper seed extraction from the MCP context.

        Args:
            session_id: Session ID from the client

        Returns:
            Session data dictionary
        """
        # Create environment with default settings
        # The proper seed will be applied when the first tool is called
        config = self.adapter.get_default_config()

        # Create environment without seed initially
        env = self.adapter.create_environment(config)
        obs, info = self.adapter.reset_environment(env, seed=None)

        # Initialize session state
        session_data = {
            "env": env,
            "obs": obs,
            "session_data": {},  # Subclasses can store additional data here
            "session_id": session_id,
        }

        # Store the session
        self.sessions[session_id] = session_data

        print(f"🎮 Created session {session_id[:16]}... for initial state endpoint")

        return session_data

    def _update_control_plane(
        self, reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]
    ):
        """
        Update control plane state after environment step (single session).

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

    def _get_or_create_session_control_plane(self, session_id: str) -> Dict[str, Any]:
        """Get or create control plane state for a specific session."""
        with self.session_lock:
            if session_id not in self.sessions:
                return {}

            session_data = self.sessions[session_id]
            if "control_plane" not in session_data["session_data"]:
                session_data["session_data"]["control_plane"] = {
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                    "info": {},
                    "step_count": 0,
                    "total_reward": 0.0,
                }

            return session_data["session_data"]["control_plane"]

    def _update_session_control_plane(
        self,
        session_id: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ):
        """Update control plane state for a specific session."""
        control_plane = self._get_or_create_session_control_plane(session_id)

        control_plane["reward"] = reward
        control_plane["terminated"] = terminated
        control_plane["truncated"] = truncated
        control_plane["info"] = info
        control_plane["step_count"] += 1
        control_plane["total_reward"] += reward

        # Log control plane update
        print(
            f"🎛️  Session {session_id[:16]}... control plane: reward={reward}, terminated={terminated}, step={control_plane['step_count']}"
        )

    def get_control_plane_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get control plane state for a specific session (for rollout system)."""
        with self.session_lock:
            if session_id in self.sessions:
                return self._get_or_create_session_control_plane(session_id).copy()
            return None

    def _execute_environment_step(self, action_int: int) -> Dict[str, Any]:
        """
        Execute environment step and update control plane (single session).

        Args:
            action_int: Parsed action integer

        Returns:
            Data plane response (observation only, no rewards)
        """
        # Execute environment step
        obs, reward, terminated, truncated, info = self.adapter.step_environment(
            self.env, action_int
        )

        # Update global observation state
        self.obs = obs

        # Update control plane (separate from data plane)
        self._update_control_plane(reward, terminated, truncated, info)

        # Return ONLY data plane information (no rewards/termination)
        return self._render(obs)

    def _execute_session_environment_step(
        self, session_id: str, action_int: int
    ) -> Dict[str, Any]:
        """
        Execute environment step for a specific session and update control plane.

        Args:
            session_id: Session identifier
            action_int: Parsed action integer

        Returns:
            Data plane response (observation only, no rewards)
        """
        session_data = self.sessions[session_id]
        env = session_data["env"]

        # Execute environment step
        obs, reward, terminated, truncated, info = self.adapter.step_environment(
            env, action_int
        )

        # Update session observation state
        session_data["obs"] = obs

        # Update control plane for this session
        self._update_session_control_plane(
            session_id, reward, terminated, truncated, info
        )

        # Return ONLY data plane information (no rewards/termination)
        return self.format_observation(obs, env)

    # ===== SESSION-AWARE CONTROL PLANE ENDPOINTS =====
    # These provide session-specific control plane data via HTTP endpoints
    # instead of global MCP resources, enabling proper multi-session support.

    @control_plane_endpoint("/control/reward")
    def get_reward_endpoint(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get current reward information for this session."""
        control_plane = self._get_session_control_plane_from_data(session_data)
        return {
            "reward": control_plane.get("reward", 0.0),
            "step_count": control_plane.get("step_count", 0),
        }

    @control_plane_endpoint("/control/status")
    def get_status_endpoint(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get current episode status for this session."""
        control_plane = self._get_session_control_plane_from_data(session_data)
        return {
            "terminated": control_plane.get("terminated", False),
            "truncated": control_plane.get("truncated", False),
            "step_count": control_plane.get("step_count", 0),
            "total_reward": control_plane.get("total_reward", 0.0),
        }

    @control_plane_endpoint("/control/info")
    def get_info_endpoint(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get current environment info for this session."""
        control_plane = self._get_session_control_plane_from_data(session_data)
        return control_plane.get("info", {})

    @control_plane_endpoint("/control/initial_state")
    def get_initial_state_endpoint(
        self, session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get initial state for this session."""
        env = session_data.get("env")
        obs = session_data.get("obs")

        if env and obs is not None:
            # Format the observation for this specific session
            return self.format_observation(obs, env)
        else:
            # Fallback if session data is not available
            return {
                "observation": "session_not_initialized",
                "session_id": session_data.get("session_id", "unknown"),
            }

    def _get_session_control_plane_from_data(
        self, session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract control plane state from session data."""
        return session_data.get("session_data", {}).get(
            "control_plane",
            {
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "info": {},
                "step_count": 0,
                "total_reward": 0.0,
            },
        )

    @abstractmethod
    def _register_tools(self):
        """
        Register domain-specific MCP tools.

        Subclasses must implement this method to register their specific tools
        using the @self.mcp.tool() decorator pattern.

        IMPORTANT: Tools should only return data plane information (observations).
        Control plane information (rewards, termination) is available via resources.
        """
        pass

    @staticmethod
    @abstractmethod
    def format_observation(obs: Any, env: Any) -> Dict[str, Any]:
        """
        Format observation for MCP response.

        Args:
            obs: Raw observation from environment
            env: Environment instance

        Returns:
            Formatted observation dictionary (DATA PLANE ONLY)
        """
        pass

    def _new_env(self, seed: Optional[int] = None) -> Tuple[Any, Any, Dict]:
        """Create new environment and return initial state."""
        config = self.adapter.get_default_config()
        
        try:
            env, obs, info = self.adapter.create_environment_with_seed(config, seed=seed)
        except AttributeError:
            env = self.adapter.create_environment(config)
            obs, info = self.adapter.reset_environment(env, seed=seed)
        
        return env, obs, info

    def _render(self, obs) -> Dict[str, Any]:
        """Format observation using subclass implementation."""
        return self.format_observation(obs, self.env)