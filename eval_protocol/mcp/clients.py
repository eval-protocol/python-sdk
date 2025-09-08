import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import aiohttp  # Still needed for type hints if we expose the session, but primary interaction changes
import mcp.types  # Reverted to mcp.types; Explicit import for clarity
from mcp import types as mcp_types  # Reverted to mcp.types; Explicit import for clarity
from mcp.client.session import DEFAULT_CLIENT_INFO, ClientSession
from mcp.client.streamable_http import streamablehttp_client
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class IntermediaryMCPClient:
    """
    Client for interacting with the RewardKitIntermediaryServer using mcp.client components.
    """

    def __init__(self, intermediary_server_url: str):
        if not intermediary_server_url:
            raise ValueError("intermediary_server_url must be provided.")
        self.server_url = intermediary_server_url.rstrip("/")  # Should be like http://localhost:8001/mcp

        self._exit_stack: Optional[AsyncExitStack] = None
        self._mcp_session: Optional[ClientSession] = None

    async def connect(self):
        """Establishes connection and MCP session."""
        # ClientSession does not expose a stable public `is_closed`; consider session presence sufficient
        if self._mcp_session is not None:
            logger.debug("Already connected.")
            return

        self._exit_stack = AsyncExitStack()
        try:
            logger.debug(f"Attempting to connect to Intermediary MCP server at {self.server_url}")
            read_stream, write_stream, http_session_info = await self._exit_stack.enter_async_context(
                streamablehttp_client(self.server_url)
            )
            # http_session_info might contain the underlying aiohttp session if needed, and mcp_session_id
            # logger.debug(f"Streamable HTTP transport established. HTTP session info: {http_session_info}")

            self._mcp_session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream, client_info=DEFAULT_CLIENT_INFO)
            )
            await self._mcp_session.initialize()
            logger.info(f"IntermediaryMCPClient connected and MCP session initialized with {self.server_url}")
        except Exception as e:
            if self._exit_stack:  # pragma: no cover
                await self._exit_stack.aclose()
                self._exit_stack = None
            self._mcp_session = None
            logger.error(
                f"Failed to connect or initialize MCP session with {self.server_url}: {e}",
                exc_info=True,
            )
            raise

    async def close(self):
        """Closes the MCP session and underlying transport."""
        if self._exit_stack:
            logger.debug(f"Closing IntermediaryMCPClient connection to {self.server_url}")
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._mcp_session = None
            logger.info(f"IntermediaryMCPClient connection to {self.server_url} closed.")

    async def _ensure_connected(self):
        # ClientSession doesn't have a public is_closed.
        # We rely on _mcp_session being None or connect() re-establishing.
        # The AsyncExitStack handles actual closure of resources.
        if not self._mcp_session:
            logger.debug("Session not established, attempting to connect...")
            await self.connect()

        # After attempting to connect, if _mcp_session is still None, it means connection failed.
        if not self._mcp_session:
            raise RuntimeError("Failed to establish or re-establish MCP session.")

    async def _call_intermediary_tool(self, tool_name: str, tool_args_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Helper to make a raw tool call to the intermediary server and parse the result.
        The tool_args_payload is the "arguments" field for the intermediary's tool.
        """
        await self._ensure_connected()
        if not self._mcp_session:  # For type checker
            raise RuntimeError("MCP session not available after ensure_connected.")

        logger.debug(f"Calling intermediary tool '{tool_name}' with payload: {tool_args_payload}")

        mcp_response: mcp_types.CallToolResult = await self._mcp_session.call_tool(tool_name, tool_args_payload)

        logger.debug(f"Raw MCP response from intermediary for '{tool_name}': {mcp_response}")

        if mcp_response.isError or not mcp_response.content or not hasattr(mcp_response.content[0], "text"):
            error_message = f"Tool call '{tool_name}' to intermediary failed."
            if mcp_response.isError and mcp_response.content and hasattr(mcp_response.content[0], "text"):
                error_text = getattr(mcp_response.content[0], "text", "")
                error_message += f" Details: {error_text}"
            elif mcp_response.isError:
                error_message += " No detailed error message in content."
            logger.error(error_message)
            try:
                if mcp_response.content and hasattr(mcp_response.content[0], "text"):
                    parsed_error = json.loads(getattr(mcp_response.content[0], "text", ""))
                    if isinstance(parsed_error, dict) and "error" in parsed_error:
                        raise RuntimeError(f"{error_message} Nested error: {parsed_error['error']}")
            except (json.JSONDecodeError, TypeError):
                pass
            raise RuntimeError(error_message)

        try:
            parsed_result = json.loads(getattr(mcp_response.content[0], "text", ""))
            logger.debug(f"Parsed JSON result from intermediary for '{tool_name}': {parsed_result}")
            return parsed_result
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON from intermediary's tool '{tool_name}' response content: {getattr(mcp_response.content[0], 'text', '')}. Error: {e}"
            )
            raise RuntimeError(f"Failed to parse JSON response from intermediary tool '{tool_name}'.")

    async def initialize_session(self, backend_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Initializes a session with the IntermediaryServer, requesting backend instances.
        """
        payload_for_intermediary_tool = {"args": {"backends": backend_requests}}
        return await self._call_intermediary_tool(
            tool_name="initialize_session",
            tool_args_payload=payload_for_intermediary_tool,
        )

    async def call_backend_tool(
        self,
        rk_session_id: str,
        instance_id: str,
        backend_name_ref: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calls a tool on a specific backend instance managed by the IntermediaryServer.
        """
        payload_for_intermediary_tool = {
            "args": {
                "rk_session_id": rk_session_id,
                "instance_id": instance_id,
                "backend_name_ref": backend_name_ref,
                "tool_name": tool_name,
                "tool_args": tool_args,
            }
        }
        return await self._call_intermediary_tool(
            tool_name="call_backend_tool",
            tool_args_payload=payload_for_intermediary_tool,
        )

    async def list_backend_tools(
        self, rk_session_id: str, instance_id: str, backend_name_ref: str
    ) -> mcp_types.ListToolsResult:
        """
        Lists tools available on a specific backend instance via the IntermediaryServer.
        """
        payload_for_intermediary_tool = {
            "args": {
                "rk_session_id": rk_session_id,
                "instance_id": instance_id,
                "backend_name_ref": backend_name_ref,
            }
        }
        # _call_intermediary_tool returns a Dict[str, Any] which is the parsed JSON
        # from the intermediary's response. This dict should be the model_dump of ListToolsResult.
        raw_result_dict = await self._call_intermediary_tool(
            tool_name="list_backend_tools",
            tool_args_payload=payload_for_intermediary_tool,
        )
        # Parse the dictionary back into the Pydantic model
        return mcp_types.ListToolsResult(**raw_result_dict)

    async def cleanup_session(self, rk_session_id: str) -> Dict[str, Any]:
        """
        Cleans up a session on the IntermediaryServer.
        """
        payload_for_intermediary_tool = {"args": {"rk_session_id": rk_session_id}}
        result = await self._call_intermediary_tool(
            tool_name="cleanup_session", tool_args_payload=payload_for_intermediary_tool
        )
        return result

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
