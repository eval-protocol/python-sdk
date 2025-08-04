import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from typing import Optional

import click
import uvicorn
import yaml
from mcp.server.streamable_http_manager import (  # MCP SDK component
    StreamableHTTPSessionManager,
)
from starlette.applications import Starlette
from starlette.routing import Mount, Route  # Import Mount

from eval_protocol.mcp_agent.config import AppConfig
from eval_protocol.mcp_agent.intermediary_server import RewardKitIntermediaryServer

logger = logging.getLogger(__name__)

# Global server instance to be managed by signal handlers
# This will now be the Uvicorn server instance.
_uvicorn_server_instance_ref: Optional[uvicorn.Server] = None  # Keep a global ref if needed for signals
# Keep a reference to our MCP server for lifespan management
_mcp_server_instance_ref: Optional[RewardKitIntermediaryServer] = None
# _session_manager_ref is not needed globally if lifespan_wrapper handles it.


# Custom app_lifespan is no longer needed if StreamableHTTPSessionManager.lifespan_wrapper is used.


async def main_async(config_path: str, host: str, port: int):
    """
    Asynchronous main function to load config, set up the ASGI application,
    and run it with Uvicorn.
    """
    global _uvicorn_server_instance_ref, _mcp_server_instance_ref  # _session_manager_ref removed from globals
    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
        app_config = AppConfig(**raw_config)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        return
    except Exception as e:
        logger.error(f"Error loading or validating AppConfig from {config_path}: {e}")
        return

    # Configure logging early
    server_root_log_level_str = app_config.log_level.upper()
    server_root_log_level = getattr(logging, server_root_log_level_str, logging.INFO)

    logging.basicConfig(
        level=server_root_log_level,  # Root logger for the server process
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # Added datefmt for consistency
    )
    logger.info(f"Configuration loaded from {config_path}. Server root log level set to {server_root_log_level_str}.")

    # Ensure eval_protocol.mcp_agent namespace respects this level
    rk_mcp_agent_logger = logging.getLogger("eval_protocol.mcp_agent")
    rk_mcp_agent_logger.setLevel(server_root_log_level)

    # Be very explicit for the intermediary_server logger as well
    intermediary_server_logger = logging.getLogger("eval_protocol.mcp_agent.intermediary_server")
    intermediary_server_logger.setLevel(server_root_log_level)
    # Also ensure its handlers respect this level
    for handler in intermediary_server_logger.handlers:
        handler.setLevel(server_root_log_level)
    # If it's propagating to the 'eval_protocol.mcp_agent' parent, ensure that parent's handlers are also correct.
    # The parent rk_mcp_agent_logger already had its level set.

    # Quiet down other noisy libraries for the server unless server itself is in DEBUG mode
    if server_root_log_level > logging.DEBUG:  # e.g. if INFO or WARNING
        libraries_to_quiet = [
            "httpx",
            "mcp",
            "uvicorn",
            "starlette",
            "asyncio",
            "hpack",
            "httpcore",
        ]
        for lib_name in libraries_to_quiet:
            logging.getLogger(lib_name).setLevel(logging.WARNING)

    logger.info(
        f"Log level for 'eval_protocol.mcp_agent' namespace set to {logging.getLevelName(logging.getLogger('eval_protocol.mcp_agent').getEffectiveLevel())}"
    )

    # 1. Instantiate RewardKitIntermediaryServer
    _mcp_server_instance_ref = RewardKitIntermediaryServer(
        app_config=app_config
    )  # Store globally for lifespan_wrapper

    # 2. Instantiate StreamableHTTPSessionManager
    # Pass the internal _mcp_server (the MCPServer instance) from our FastMCP subclass
    session_manager = StreamableHTTPSessionManager(
        app=_mcp_server_instance_ref._mcp_server,
        event_store=None,
        json_response=True,  # Changed to True
    )

    # 3. Create Starlette app, using session_manager.lifespan_wrapper
    # This wrapper should handle the startup/shutdown of both the session_manager's task group
    # and the underlying _mcp_server_instance_ref.
    routes = [
        Mount("/mcp", app=session_manager.handle_request),
    ]

    # The lifespan_wrapper approach was incorrect as the method doesn't exist.
    # We will now use a custom lifespan for the MCPServer and run Uvicorn
    # within the context of session_manager.run() if it's an async context manager.

    @asynccontextmanager
    async def mcp_server_lifespan_only(app_for_lifespan: Starlette):
        # This lifespan only manages the _mcp_server_instance_ref
        if _mcp_server_instance_ref:
            logger.info("MCP Server Lifespan: Starting up RewardKitIntermediaryServer...")
            await _mcp_server_instance_ref.startup()
            logger.info("MCP Server Lifespan: RewardKitIntermediaryServer startup complete.")
        yield
        if _mcp_server_instance_ref:
            logger.info("MCP Server Lifespan: Shutting down RewardKitIntermediaryServer...")
            await _mcp_server_instance_ref.shutdown()
            logger.info("MCP Server Lifespan: RewardKitIntermediaryServer shutdown complete.")

    routes = [
        Mount("/mcp", app=session_manager.handle_request),
    ]
    starlette_app = Starlette(routes=routes, lifespan=mcp_server_lifespan_only)

    # 4. Configure Uvicorn
    config = uvicorn.Config(
        app=starlette_app,  # Starlette app with its own lifespan for MCPServer
        host=host,
        port=port,
        log_level=app_config.log_level.lower(),
        log_config=None,  # Prevent Uvicorn from overriding our basicConfig for app loggers
    )
    uvicorn_server = uvicorn.Server(config)
    _uvicorn_server_instance_ref = uvicorn_server

    logger.info(f"Starting RewardKit Intermediary MCP Server on {host}:{port}/mcp.")

    try:
        if hasattr(session_manager, "run"):
            # Call run() to get the potential context manager
            sm_context_manager = session_manager.run()
            if hasattr(sm_context_manager, "__aenter__") and hasattr(sm_context_manager, "__aexit__"):
                logger.info(
                    "Attempting to run Uvicorn server within context returned by StreamableHTTPSessionManager.run()..."
                )
                async with sm_context_manager:  # type: ignore
                    logger.info("Context from StreamableHTTPSessionManager.run() entered. Serving Uvicorn...")
                    await uvicorn_server.serve()
            else:
                logger.error(
                    "Object returned by StreamableHTTPSessionManager.run() is not an async context manager. Falling back to direct Uvicorn serve."
                )
                await uvicorn_server.serve()
        else:
            logger.error(
                "StreamableHTTPSessionManager does not have a 'run' method. Falling back to direct Uvicorn serve."
            )
            await uvicorn_server.serve()

    except asyncio.CancelledError:
        logger.info("Server operation cancelled (main_async level).")
    except Exception as e:
        logger.error(
            f"An error occurred during server operation (main_async level): {e}",
            exc_info=True,
        )
    finally:
        logger.info("Uvicorn server has shut down (main_async finally).")


# Signal handling is now primarily managed by Uvicorn.
# If we needed custom logic *before* Uvicorn handles signals, it would be more complex.
# For now, relying on Uvicorn's graceful shutdown which triggers the ASGI lifespan.


@click.command()
@click.option(
    "--config",
    "config_path",
    default="mcp_agent_config.yaml",
    help="Path to the YAML configuration file for the MCP agent server.",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--host", default="0.0.0.0", help="Host for the server to listen on.")
@click.option("--port", default=8001, type=int, help="Port for the server to listen on.")
def main_cli(config_path: str, host: str, port: int):
    """
    CLI entry point to run the RewardKit Intermediary MCP Server using Uvicorn.
    """
    try:
        asyncio.run(main_async(config_path, host, port))
    except KeyboardInterrupt:  # This will be caught by Uvicorn first usually
        logger.info("CLI interrupted by KeyboardInterrupt. Uvicorn should handle shutdown.")
    finally:
        logger.info("MCP Agent Server CLI finished.")


if __name__ == "__main__":
    main_cli()
