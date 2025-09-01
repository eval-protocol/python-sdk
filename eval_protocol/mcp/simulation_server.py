"""
MCP Simulation Server Framework

This framework enforces the correct separation between production and simulation servers.
It ensures that:
1. No session management tools are exposed to models
2. Session initialization happens via client_info (MCP spec)
3. Only domain game tools are exposed
4. Simulation logic is handled internally using proper MCP session management

Usage:
    class MyGameSimulation(SimulationServerBase):
        def create_environment(self, config): ...
        def reset_environment(self, env, seed): ...
        # etc.

    server = MyGameSimulation("MyGame")
    server.run()
"""

import asyncio
import contextlib
import functools
import inspect
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Iterable, cast
from pydantic import AnyUrl

import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ToolMismatchError(Exception):
    """Raised when simulation and production tools do not match."""

    pass


class SignatureMismatchError(Exception):
    """Raised when a tool's signature does not match the production version."""

    pass


def simulation_tool(func: Callable) -> Callable:
    """
    Decorator to mark methods as simulation tools.
    These tools will be exposed to the MCP client and validated against production.
    """
    func._is_simulation_tool = True
    return func


def simulation_resource(uri_pattern: str) -> Callable:
    """
    Decorator to mark methods as MCP resources in simulation servers.

    Unlike production resources, simulation resources have access to session context
    and can provide session-specific initial states based on initialization options.
    """

    def decorator(func: Callable) -> Callable:
        func._is_resource = True
        func._resource_uri = uri_pattern
        return func

    return decorator


class SimulationServerBase(ABC):
    """
    Base class for simulation MCP servers using proper StreamableHTTPSessionManager.

    This framework enforces correct separation by:
    - Using StreamableHTTPSessionManager for proper session management
    - Extracting seeds from client_info during session initialization
    - Only exposing domain-specific game tools
    - Preventing session management tool pollution
    - Supporting MCP resources for initial state following proper MCP patterns
    """

    def __init__(
        self,
        server_name: str,
        production_server_app=None,
    ):
        """
        Initialize simulation server framework.

        Args:
            server_name: Name for the MCP server.
            production_server_app: The production server app instance for validation (optional).
        """
        self.server_name = server_name
        self.production_server_app = production_server_app
        self._domain_tools: Dict[str, Callable] = {}
        self._domain_resources: Dict[str, Callable] = {}

        # Create low-level MCP server
        self.app = Server(server_name)

        # Session state storage for simulation environments
        self.session_environments: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.Lock()

        # Discover and register domain tools and resources
        self._discover_and_register_tools()
        self._discover_and_register_resources()
        self._register_session_handlers()

    def _get_session_id_from_context(self, ctx) -> str:
        """Extract session ID from MCP request context."""
        # Use a stable session ID based on the client info
        # Since we know the client_info is consistent for a given session,
        # we can use a hash of the client_info to create a stable session ID
        if hasattr(ctx, "session") and hasattr(ctx.session, "client_params"):
            client_params = ctx.session.client_params
            if hasattr(client_params, "clientInfo"):
                client_info = client_params.clientInfo
                if client_info and hasattr(client_info, "_extra"):
                    extra_data = client_info._extra
                    if extra_data and isinstance(extra_data, dict):
                        # Create a stable session ID based on seed and other config
                        import hashlib
                        import json

                        stable_data = {
                            "seed": extra_data.get("seed"),
                            "config": extra_data.get("config", {}),
                            "name": client_info.name,
                            "version": client_info.version,
                        }
                        stable_str = json.dumps(stable_data, sort_keys=True)
                        session_id = hashlib.md5(stable_str.encode()).hexdigest()
                        logger.debug(f"Generated stable session_id from client_info: {session_id}")
                        return session_id

        # Fallback for testing or other scenarios
        session_id = f"sim_{id(ctx)}"
        logger.debug(f"Generated fallback session_id: {session_id}")
        return session_id

    def _get_or_create_session_env(self, ctx) -> Dict[str, Any]:
        """
        Get or create session environment.

        This extracts the seed from client_info and creates a session-specific environment.
        """
        session_id = self._get_session_id_from_context(ctx)

        with self.session_lock:
            if session_id not in self.session_environments:
                # Extract seed from client info if available
                config = self.get_default_config()
                seed = None

                # Extract client info and seed
                if hasattr(ctx, "session") and hasattr(ctx.session, "client_params"):
                    client_params = ctx.session.client_params
                    if hasattr(client_params, "clientInfo"):
                        client_info = client_params.clientInfo
                        if client_info and hasattr(client_info, "_extra"):
                            extra_data = client_info._extra
                            if extra_data and isinstance(extra_data, dict):
                                # Extract seed from client info
                                seed = extra_data.get("seed")
                                logger.info(f"🎯 Extracted seed from client_info: {seed}")
                                # Update config with any additional options
                                if "config" in extra_data:
                                    config.update(extra_data["config"])

                # Create environment with seed - use create_environment_with_seed if available
                # This is important for environments like FrozenLake that need the seed during creation
                if hasattr(self, "create_environment_with_seed") and callable(
                    getattr(self, "create_environment_with_seed")
                ):
                    env, obs, info = self.create_environment_with_seed(config, seed=seed)
                else:
                    env = self.create_environment(config)
                    obs, info = self.reset_environment(env, seed=seed)

                self.session_environments[session_id] = {
                    "env": env,
                    "config": config,
                    "seed": seed,
                    "created_at": time.time(),
                    "initial_observation": self.format_observation(obs),
                    "session_id": session_id,
                    "steps": 0,
                    "total_reward": 0.0,
                    "last_used": time.time(),
                }
                logger.info(f"🆕 Simulation session created: {session_id[:16]}... (seed={seed})")

            self.session_environments[session_id]["last_used"] = time.time()
            return self.session_environments[session_id]

    def _discover_and_register_tools(self):
        """
        Discover and register tools marked with @simulation_tool.
        """
        # 1. Discover tools on the subclass instance
        discovered_tools = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_is_simulation_tool"):
                discovered_tools[method.__name__] = method
        self._domain_tools = discovered_tools

        # 2. Register the discovered tools with the MCP server
        if discovered_tools:

            @self.app.call_tool()
            async def call_tool(name: str, arguments: dict):
                # Get the current request context
                ctx = self.app.request_context
                session_state = self._get_or_create_session_env(ctx)

                # Find the matching tool function
                if name in self._domain_tools:
                    tool_func = self._domain_tools[name]

                    # Check if the tool function is async or sync
                    if inspect.iscoroutinefunction(tool_func):
                        result = await tool_func(ctx=ctx, session_state=session_state, **arguments)
                    else:
                        # For sync functions, call them directly
                        result = tool_func(ctx=ctx, session_state=session_state, **arguments)

                    # Return list of ContentBlock for low-level server
                    from mcp.types import TextContent

                    result_str = json.dumps(result) if not isinstance(result, str) else result
                    return [TextContent(type="text", text=result_str)]
                else:
                    raise ValueError(f"Unknown tool: {name}")

            @self.app.list_tools()
            async def list_tools():
                """List all available tools."""
                from mcp.types import Tool

                tools = []
                for tool_name, tool_func in self._domain_tools.items():
                    # Extract docstring as description
                    description = tool_func.__doc__ or f"Execute {tool_name} action"

                    # Create a basic input schema - could be enhanced by inspecting function signature
                    input_schema = {"type": "object", "properties": {}, "required": []}

                    tools.append(
                        Tool(
                            name=tool_name,
                            description=description,
                            inputSchema=input_schema,
                        )
                    )

                return tools

            logger.info(f"✅ Registered {len(discovered_tools)} domain tools")

    def _discover_and_register_resources(self):
        """
        Discover and register resources on the subclass instance.
        """
        # 1. Discover resources on the subclass instance
        discovered_resources = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_is_resource"):
                discovered_resources[method.__name__] = method
        self._domain_resources = discovered_resources

        # 2. Register the discovered resources with the MCP server
        if discovered_resources:

            @self.app.read_resource()
            async def read_resource(uri: AnyUrl):
                # Get the current request context
                ctx = self.app.request_context

                # Find the matching resource function by URI pattern
                for resource_name, resource_func in self._domain_resources.items():
                    resource_uri_pattern = getattr(resource_func, "_resource_uri", f"/{resource_name}")
                    # Convert URI to string for pattern matching
                    uri_str = str(uri)
                    # Simple pattern matching - could be enhanced for complex patterns
                    if uri_str == resource_uri_pattern or uri_str.endswith(resource_uri_pattern):
                        # Create session state for this resource call
                        session_state = self._get_or_create_session_env(ctx)

                        # Check if the resource function is async or sync
                        if inspect.iscoroutinefunction(resource_func):
                            result = await resource_func(ctx=ctx, session_state=session_state)
                        else:
                            # For sync functions, call them directly
                            result = resource_func(ctx=ctx, session_state=session_state)

                        # Ensure we return the proper format for the low-level server
                        if isinstance(result, str):
                            return result
                        else:
                            return json.dumps(result)

                raise ValueError(f"Unknown resource: {uri}")

            @self.app.list_resources()
            async def list_resources():
                """List all available resources."""
                from mcp.types import Resource

                resources = []
                for resource_name, resource_func in self._domain_resources.items():
                    # Extract docstring as description
                    description = resource_func.__doc__ or f"Resource {resource_name}"

                    # Some callables may not have the attribute; guard for type checkers.
                    # Resource expects AnyUrl; pass as str and allow coercion by pydantic.
                    uri_value: str = str(getattr(resource_func, "_resource_uri", f"/{resource_name}"))
                    resources.append(
                        Resource(
                            uri=cast(AnyUrl, uri_value),
                            name=resource_name,
                            description=description,
                            mimeType="application/json",
                        )
                    )

                return resources

            logger.info(f"✅ Registered {len(discovered_resources)} domain resources")

    def _register_session_handlers(self):
        """Register session initialization and cleanup handlers."""

        @self.app.set_logging_level()
        async def set_logging_level(level: str) -> None:
            """Handle logging level requests."""
            # Validate and set logging level; ignore invalid values gracefully
            try:
                numeric_level = getattr(logging, level.upper())
                if isinstance(numeric_level, int):
                    logger.setLevel(numeric_level)
            except Exception:
                pass

        # NOTE: The low-level Server doesn't have built-in session lifecycle hooks
        # We'll need to capture client_info during the first request in each session
        # This is a limitation of using the low-level server directly

    # Abstract methods that subclasses MUST implement

    @abstractmethod
    def create_environment(self, config: Dict[str, Any]) -> Any:
        """Create environment instance."""
        pass

    @abstractmethod
    def reset_environment(self, env: Any, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def step_environment(self, env: Any, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute step in environment."""
        pass

    @abstractmethod
    def close_environment(self, env: Any) -> None:
        """Clean up environment resources."""
        pass

    @abstractmethod
    def parse_action(self, action_str: str) -> Any:
        """Parse action string to environment action."""
        pass

    @abstractmethod
    def format_observation(self, observation: Any) -> Any:
        """Format observation for JSON serialization."""
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default environment configuration."""
        pass

    # Optional hook: some environments need seed at creation time
    def create_environment_with_seed(
        self, config: Dict[str, Any], *, seed: Optional[int] = None
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Create environment with a seed when required; default falls back to create+reset.

        Subclasses can override when the environment requires the seed at construction time.
        Returns a tuple of (env, initial_observation, info).
        """
        env = self.create_environment(config)
        obs, info = self.reset_environment(env, seed=seed)
        return env, obs, info

    def run(self, port: int = 8000, host: str = "127.0.0.1", **kwargs):
        """
        Run the simulation server using StreamableHTTPSessionManager.

        Args:
            port: Port to listen on
            host: Host to bind to
            **kwargs: Additional arguments for uvicorn
        """
        print("📡 Starting simulation server with StreamableHTTPSessionManager")
        print(f"🎮 Domain tools: {list(self._domain_tools.keys())}")
        print(f"📦 Domain resources: {list(self._domain_resources.keys())}")
        if self.production_server_app:
            print("✅ Tool signatures validated against production server.")
        print("🚫 No session management tools exposed (framework enforced)")
        print()

        # Create the session manager with our app
        session_manager = StreamableHTTPSessionManager(
            app=self.app,
        )

        # ASGI handler for streamable HTTP connections
        async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
            await session_manager.handle_request(scope, receive, send)

        @contextlib.asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncIterator[None]:
            """Context manager for managing session manager lifecycle."""
            async with session_manager.run():
                logger.info(f"🚀 {self.server_name} started with StreamableHTTP session manager!")
                try:
                    yield
                finally:
                    logger.info("🧹 Simulation server shutting down...")
                    # Clean up session environments
                    with self.session_lock:
                        for (
                            session_id,
                            session_data,
                        ) in self.session_environments.items():
                            env = session_data.get("env")
                            if env:
                                try:
                                    self.close_environment(env)
                                except Exception as e:
                                    logger.warning(f"⚠️ Error closing environment in session {session_id}: {e}")
                        self.session_environments.clear()
                    logger.info("✅ Simulation server shutdown complete")

        # Create an ASGI application using the transport
        starlette_app = Starlette(
            debug=kwargs.get("debug", False),
            routes=[
                Mount("/mcp", app=handle_streamable_http),
            ],
            lifespan=lifespan,
        )

        # Run the server
        uvicorn.run(
            starlette_app,
            host=host,
            port=port,
            log_level=kwargs.get("log_level", "info"),
            **{k: v for k, v in kwargs.items() if k not in ["debug", "log_level"]},
        )
