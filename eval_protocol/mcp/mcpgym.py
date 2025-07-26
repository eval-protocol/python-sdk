"""
MCP-Gym Framework - North Star Implementation

This module provides the core McpGym base class that implements the north star vision
for universal RL environment integration via MCP protocol.

Key Features:
- Unified MCP server with FastMCP integration
- Simple tool registration with @self.mcp.tool() decorator
- Clean separation between data plane (MCP tool calls) and control plane (custom endpoints)
- Compatible with CondaServerProcessManager
- Session-aware control plane endpoints via @control_plane_endpoint decorator
"""

import os
import hashlib
import threading
import inspect
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from .adapter import EnvironmentAdapter

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


class McpGym(ABC):
    """
    Base class for MCP-Gym environments implementing the north star vision.

    This class provides the universal adapter pattern for RL environments,
    bridging training infrastructure, production MCP standards, and high-quality
    environments through a clean, standardized interface.

    Key Design Principles:
    - Data Plane: JSON tool calls/responses via MCP (state transitions/actions)
    - Control Plane: Rewards/termination signals via MCP resources
    - Environment Implementation: Single-process MCP server per environment
    """

    def __init__(self, server_name: str, adapter: EnvironmentAdapter, seed: Optional[int] = None):
        """
        Initialize the MCP-Gym environment.

        Args:
            server_name: Name for the MCP server
            adapter: Environment adapter instance
            seed: Optional seed for reproducible environments
        """
        self.adapter = adapter

        # Create FastMCP server
        self.mcp = FastMCP(
            server_name,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8000)),
        )

        # Multi-session support
        self.sessions = {}  # session_id -> {"env": env, "obs": obs, "session_data": data}
        self.session_lock = threading.Lock()

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

        # Register tools and control plane endpoints
        self._register_tools()
        self._discover_and_register_control_plane_endpoints()

    def _get_session_id(self, ctx: Context) -> str:
        """
        Extract session ID from MCP context using proper FastMCP pattern.

        Creates stable session IDs based on client info (seed + config + client details)
        for consistent session management across reconnections.
        """
        print(f"🔍 _get_session_id: Starting session ID extraction")
        print(f"🔍 _get_session_id: ctx type: {type(ctx)}")
        print(f"🔍 _get_session_id: hasattr(ctx, 'session'): {hasattr(ctx, 'session')}")

        # Use stable session ID based on client info (following simulation_server.py pattern)
        if hasattr(ctx, "session") and hasattr(ctx.session, "client_params"):
            client_params = ctx.session.client_params
            print(f"🔍 _get_session_id: client_params type: {type(client_params)}")
            print(f"🔍 _get_session_id: hasattr(client_params, 'clientInfo'): {hasattr(client_params, 'clientInfo')}")

            if hasattr(client_params, "clientInfo"):
                client_info = client_params.clientInfo
                print(f"🔍 _get_session_id: client_info: {client_info}")
                print(f"🔍 _get_session_id: hasattr(client_info, '_extra'): {hasattr(client_info, '_extra')}")

                if client_info and hasattr(client_info, "_extra"):
                    extra_data = client_info._extra
                    print(f"🔍 _get_session_id: extra_data: {extra_data}")
                    print(f"🔍 _get_session_id: extra_data type: {type(extra_data)}")

                    if extra_data and isinstance(extra_data, dict):
                        # Create a stable session ID based on seed and other config
                        seed_value = extra_data.get("seed")
                        config_value = extra_data.get("config", {})
                        dataset_row_id_value = extra_data.get("dataset_row_id")
                        model_id_value = extra_data.get("model_id")

                        print(f"🔍 _get_session_id: seed_value: {seed_value} (type: {type(seed_value)})")
                        print(f"🔍 _get_session_id: config_value: {config_value}")

                        stable_data = {
                            "seed": seed_value,
                            "config": config_value,
                            "dataset_row_id": dataset_row_id_value,
                            "model_id": model_id_value,
                            "name": client_info.name,
                            "version": client_info.version,
                        }

                        print(f"🔍 _get_session_id: stable_data: {stable_data}")
                        stable_str = json.dumps(stable_data, sort_keys=True)
                        session_id = hashlib.md5(stable_str.encode()).hexdigest()
                        print(f"🎯 Generated stable session_id: {session_id} for seed: {seed_value}")
                        return session_id

        # Fallback for testing or other scenarios
        session_id = f"gym_{id(ctx)}"
        print(f"🎯 Generated fallback session_id: {session_id}")
        return session_id

    def _get_or_create_session(self, ctx: Context) -> Dict[str, Any]:
        """
        Get or create session data for the given context.

        This method handles comprehensive session creation with seed extraction
        from MCP context and proper environment initialization.
        """
        session_id = self._get_session_id(ctx)
        print(f"🔍 _get_or_create_session: session_id: {session_id}")

        with self.session_lock:
            if session_id not in self.sessions:
                print(f"🔍 _get_or_create_session: Creating new session for {session_id}")
                # Extract seed from context using proper FastMCP pattern
                seed = None
                config = self._get_default_config()
                print(f"🔍 _get_or_create_session: default_config: {config}")

                if hasattr(ctx, "session") and hasattr(ctx.session, "client_params"):
                    client_params = ctx.session.client_params
                    if hasattr(client_params, "clientInfo"):
                        client_info = client_params.clientInfo
                        if client_info and hasattr(client_info, "_extra"):
                            extra_data = client_info._extra
                            print(f"🔍 _get_or_create_session: extra_data in session creation: {extra_data}")
                            if extra_data and isinstance(extra_data, dict):
                                # Extract seed from client info
                                seed = extra_data.get("seed")
                                print(f"🌱 Extracted seed from client_info: {seed} (type: {type(seed)})")
                                # Update config with any additional options
                                if "config" in extra_data:
                                    config.update(extra_data["config"])
                                    print(f"🔍 _get_or_create_session: updated config: {config}")

                print(f"🔍 _get_or_create_session: About to create environment with seed: {seed}")

                env, obs, info = self._new_env(seed=seed)
                print(f"🔍 _get_or_create_session: environment created with obs: {obs}, info: {info}")

                # Initialize session state
                self.sessions[session_id] = {
                    "env": env,
                    "obs": obs,
                    "session_data": {},  # Subclasses can store additional data here
                    "session_id": session_id,
                }

                print(f"🎮 Created new session {session_id[:16]}... with seed {seed}, initial obs: {obs}")
            else:
                print(f"🔍 _get_or_create_session: Returning existing session {session_id}")

            return self.sessions[session_id]

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
                                    env, obs, info = self._new_env(seed=None)
                                    # Initialize session state with extracted seed from session ID
                                    session_data = {
                                        "env": env,
                                        "obs": obs,
                                        "session_data": {},  # Subclasses can store additional data here
                                        "session_id": session_id,
                                    }
                                    # Store the session
                                    self.sessions[session_id] = session_data
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
            logger.info(f"✅ Registered {len(discovered_endpoints)} session-aware control plane endpoints")
            for name, endpoint in discovered_endpoints.items():
                logger.info(f"  - {name}: {endpoint._control_plane_path}")
        else:
            logger.info("⚠️  No session-aware control plane endpoints discovered")

    def _update_control_plane(self, reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]):
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
        obs, reward, terminated, truncated, info = self.adapter.step_environment(self.env, action_int)

        # Update global observation state
        self.obs = obs

        # Update control plane (separate from data plane)
        self._update_control_plane(reward, terminated, truncated, info)

        # Return ONLY data plane information (no rewards/termination)
        return self._render(obs)

    def _execute_session_environment_step(self, session_id: str, action: Any) -> Dict[str, Any]:
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
        obs, reward, terminated, truncated, info = self.adapter.step_environment(env, action)

        # Update session observation state
        session_data["obs"] = obs

        # Update control plane for this session
        self._update_session_control_plane(session_id, reward, terminated, truncated, info)

        # Return ONLY data plane information (no rewards/termination)
        return self.format_observation(obs, env)

    def _new_env(self, seed: Optional[int] = None) -> Tuple[Any, Any, Dict]:
        """Create new environment and return initial state."""
        config = self.adapter.get_default_config()

        if seed:
            env, obs, info = self.adapter.create_environment_with_seed(config, seed=seed)
        else:
            env = self.adapter.create_environment(config)
            obs, info = self.adapter.reset_environment(env, seed=seed)

        return env, obs, info

    def _render(self, obs) -> Dict[str, Any]:
        """Format observation using subclass implementation."""
        return self.format_observation(obs, self.env)

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration from adapter.

        Wrapper method to handle potential adapter interface issues.
        """
        try:
            return self.adapter.get_default_config()
        except AttributeError:
            # Fallback for adapters that don't implement get_default_config
            return {}

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
    def get_initial_state_endpoint(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get initial state for this session."""
        env = session_data.get("env")
        obs = session_data.get("obs")

        if env and obs is not None:
            try:
                formatted_obs = self.format_observation(obs, env)
                return formatted_obs
            except Exception as e:
                logger.error(f"❌ Error in format_observation: {e}")
                return {
                    "error": f"Failed to format observation: {str(e)}",
                    "observation_type": str(type(obs)),
                    "session_id": session_data.get("session_id", "unknown"),
                }
        else:
            # Fallback if session data is not available
            return {
                "observation": "session_not_initialized",
                "session_id": session_data.get("session_id", "unknown"),
            }

    def _get_session_control_plane_from_data(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
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

    def format_observation(self, obs: Any, env: Any) -> Dict[str, Any]:
        """
        Format observation for MCP response.

        Args:
            obs: Raw observation from environment
            env: Environment instance

        Returns:
            Formatted observation dictionary (DATA PLANE ONLY)
            
        Implementation Note:
            You can use self._to_json_serializable(obs) as a starting point
            for most standard serialization needs.
        """
        serialized_obs = self._to_json_serializable(obs)
        
        # If it's already a dict, return as-is, otherwise wrap it
        if isinstance(serialized_obs, dict):
            return serialized_obs
        else:
            return {"observation": serialized_obs}

    def run(self, transport: str = "streamable-http", **kwargs):
        """
        Run the unified MCP-Gym server.

        Args:
            transport: MCP transport protocol ("stdio", "sse", "streamable-http")
            **kwargs: Additional arguments passed to FastMCP.run()
        """
        print(f"🚀 {self.mcp.name} MCP-Gym Server Starting...")
        print(f"📡 Transport: {transport}")
        print("🎯 MCP Pattern: HTTP endpoints for control plane, tools for data plane")
        print("🔗 Session-aware control plane endpoints:")

        # List registered control plane endpoints
        for endpoint_name, endpoint_func in self._control_plane_endpoints.items():
            print(f"  - {endpoint_name}: {endpoint_func._control_plane_path}")

        if not self._control_plane_endpoints:
            print("  - No control plane endpoints registered")

        print()

        # Run the unified server
        self.mcp.run(transport=transport, **kwargs)

    def _to_json_serializable(self, obj: Any) -> Any:
        """Convert any object to JSON-serializable format.
        
        Handles Pydantic models, dataclasses, lists, dicts, and primitive types.
        This is a utility method that can be used by format_observation implementations.
        """
        from pydantic import BaseModel
        import dataclasses
        from datetime import datetime, date
        from enum import Enum
        
        # Handle None and primitive types
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
            
        # Handle datetime objects
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
            
        # Handle enums
        elif isinstance(obj, Enum):
            return obj.value
            
        # Handle Pydantic models (covers tau2 objects and many others)
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
            
        # Handle dataclasses
        elif dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
            
        # Handle dictionaries
        elif isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
            
        # Handle lists and tuples
        elif isinstance(obj, (list, tuple)):
            return [self._to_json_serializable(item) for item in obj]
            
        # Handle sets (convert to list)
        elif isinstance(obj, set):
            return [self._to_json_serializable(item) for item in obj]
            
        # Handle objects with __dict__ (fallback)
        elif hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    try:
                        result[key] = self._to_json_serializable(value)
                    except Exception:
                        # If conversion fails, store as string
                        result[key] = str(value)
            return result
            
        # Final fallback - convert to string
        else:
            return str(obj)
