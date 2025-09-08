"""
MCP Connection Management

Handles MCP client connections, session initialization, and resource/tool discovery.
Extracted from mcp_env.py to improve modularity.
"""

import asyncio
import hashlib
import json
import logging
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple, cast

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Implementation

from ...types import MCPSession

logger = logging.getLogger(__name__)


class MCPConnectionManager:
    """Manages MCP client connections and session lifecycle."""

    def __init__(self):
        self._tools_cache: Dict[str, List[Dict]] = {}
        self._tools_cache_lock = asyncio.Lock()

    async def initialize_session(self, session: MCPSession) -> None:
        """
        Initialize a persistent MCP session.

        Args:
            session: The MCPSession to initialize
        """
        if session._mcp_session:
            # If a session exists, close it before creating a new one.
            if session._exit_stack:
                try:
                    await session._exit_stack.aclose()
                except asyncio.CancelledError:
                    # Handle cancellation gracefully (especially important for Python 3.12)
                    logger.debug(f"Session {session.session_id} reinit close was cancelled")
                except Exception as e:
                    logger.warning(f"Error closing existing session {session.session_id} during reinit: {e}")
                finally:
                    session._exit_stack = None
            session._mcp_session = None

        exit_stack = AsyncExitStack()

        # Attach client metadata for the server to consume (session_id, seed, config, etc.).
        # The server inspects a private `_extra` dict on client_info, so we populate it here.
        client_info = Implementation(name="reward-kit", version="1.0.0")
        extra_data: Dict[str, Any] = {"session_id": session.session_id}
        if session.seed is not None:
            extra_data["seed"] = session.seed
        if session.dataset_row and session.dataset_row.environment_context:
            extra_data["config"] = session.dataset_row.environment_context
        if session.dataset_row and session.dataset_row.id:
            extra_data["dataset_row_id"] = session.dataset_row.id
        if session.model_id:
            extra_data["model_id"] = session.model_id

        # Merge with any existing _extra dict instead of overwriting
        existing_extra = getattr(client_info, "_extra", None)
        merged_extra: Dict[str, Any] = {}
        if isinstance(existing_extra, dict):
            merged_extra.update(existing_extra)
        merged_extra.update(extra_data)
        setattr(client_info, "_extra", merged_extra)

        read_stream, write_stream, _ = await exit_stack.enter_async_context(
            streamablehttp_client(session.base_url, terminate_on_close=True)
        )

        mcp_session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream, client_info=client_info)
        )

        await mcp_session.initialize()

        session._mcp_session = mcp_session
        session._exit_stack = exit_stack

        # PRE-WARM: Discover and cache tools immediately after session initialization
        # This prevents concurrent list_tools() calls later
        await self._prewarm_tools_cache(session)

    async def _prewarm_tools_cache(self, session: MCPSession) -> None:
        """
        Pre-warm the tools cache for this session's base URL.
        This prevents concurrent list_tools() calls during discover_tools().
        """
        cache_key = session.base_url

        async with self._tools_cache_lock:
            # Only fetch tools if not already cached for this base_url
            if cache_key not in self._tools_cache:
                logger.debug(f"Pre-warming tools cache for {cache_key}")
                mcp_session_local = session._mcp_session
                if mcp_session_local is None:
                    raise RuntimeError("Session not initialized during prewarm")
                tools_response = await mcp_session_local.list_tools()
                tools = tools_response.tools if hasattr(tools_response, "tools") else []

                tool_schemas = []
                for tool in tools:
                    tool_schema = {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": (tool.inputSchema if hasattr(tool, "inputSchema") else {}),
                    }
                    tool_schemas.append(tool_schema)

                self._tools_cache[cache_key] = tool_schemas
                logger.debug(f"✅ PRE-WARMED {len(tool_schemas)} tools for{cache_key}")

    async def reset_session(self, session: MCPSession) -> None:
        """
        Clean session data in remote mcp server for the given session
        """
        base_url = session.base_url.rstrip("/").removesuffix("/mcp")
        url = f"{base_url}/control/reset_session"

        headers = {"mcp-session-id": session.session_id}
        body = {"seed": session.seed}

        timeout = httpx.Timeout(15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            logger.debug(f"Session {session.session_id}: reset_session -> {resp.json()}")

    async def discover_tools(self, session: MCPSession) -> List[Dict]:
        """
        Discover available tools from an MCP session.
        Now uses pre-warmed cache to avoid concurrent list_tools() calls.

        Args:
            session: The MCPSession to discover tools from

        Returns:
            List of tool schemas
        """
        if not session._mcp_session:
            raise RuntimeError("Session not initialized")

        cache_key = session.base_url

        # Check cache first (should be pre-warmed during initialization)
        async with self._tools_cache_lock:
            if cache_key in self._tools_cache:
                cached_tools = self._tools_cache[cache_key]
                logger.debug(f"Using cached tools for session {session.session_id} ({len(cached_tools)} tools)")
                return cached_tools

        # Fallback: if cache miss (shouldn't happen with pre-warming), fetch directly
        logger.warning(f"Cache miss for {cache_key} - this shouldn't happen with pre-warming")
        mcp_session = session._mcp_session

        tools_response = await mcp_session.list_tools()
        tools = tools_response.tools if hasattr(tools_response, "tools") else []

        # Convert tools to schema format - filter out internal tools
        tool_schemas = []
        for tool in tools:
            # Only expose action tools to the model, not internal state tools
            tool_schema = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": (tool.inputSchema if hasattr(tool, "inputSchema") else {}),
            }
            tool_schemas.append(tool_schema)

        # Cache the result for future use
        async with self._tools_cache_lock:
            self._tools_cache[cache_key] = tool_schemas

        return tool_schemas

    def clear_tools_cache(self, base_url: Optional[str] = None):
        """
        Clear the tools cache for debugging or when server tools change.

        Args:
            base_url: If provided, clear cache only for this URL. If None, clear all.
        """
        if base_url:
            self._tools_cache.pop(base_url, None)
            logger.debug(f"Cleared tools cache for {base_url}")
        else:
            self._tools_cache.clear()
            logger.debug("Cleared all tools cache")

    async def get_initial_state(self, session: MCPSession) -> Any:
        """
        Get initial state from session-aware control plane endpoint.
        Uses HTTP endpoint instead of MCP resources for proper session awareness.

        Args:
            session: The MCPSession to get initial state from

        Returns:
            Initial observation/state
        """
        if not session._mcp_session:
            raise RuntimeError("Session not initialized")

        # Try to get initial state from control plane endpoint first
        initial_observation = None

        try:
            # Extract base URL and session ID from the MCP session
            base_url = session.base_url.rstrip("/").removesuffix("/mcp")
            session_id = session.session_id

            if session_id:
                headers = {"mcp-session-id": session_id}

                # Query initial state endpoint
                try:
                    # Use shorter timeout for playback mode, longer timeout for high-concurrency initialization
                    # (50+ concurrent sessions need more time for initial state setup)
                    timeout = 3.0 if bool(getattr(session, "_is_playback_mode", False)) else 15.0
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        initial_state_response = await client.get(
                            f"{base_url}/control/initial_state",
                            headers=headers,
                            timeout=timeout,
                        )
                        if initial_state_response.status_code == 200:
                            initial_observation = initial_state_response.json()
                            logger.info(
                                f"Session {session.session_id}: ✅ Successfully fetched session-aware initial state from control plane endpoint"
                            )
                        else:
                            logger.warning(
                                f"Control plane initial state endpoint returned {initial_state_response.status_code}"
                            )
                except httpx.TimeoutException:
                    logger.warning(f"Control plane initial state endpoint timed out after {timeout}s")
                except Exception as e:
                    logger.warning(f"Failed to query initial state endpoint: {e}")

        except Exception as e:
            logger.warning(f"Failed to query control plane initial state endpoint: {e}")

        # Fallback to MCP resource if control plane endpoint fails (backward compatibility)
        if initial_observation is None:
            logger.debug(f"Session {session.session_id}: Falling back to MCP resource for initial state")
            initial_observation = await self._get_initial_state_from_mcp_resource(session)

        # Ensure we have some observation
        if initial_observation is None:
            logger.debug(f"Session {session.session_id}: Using default initial state")
            initial_observation = {
                "observation": "default_initial_state",
                "session_id": session.session_id,
            }

        return initial_observation

    async def _get_initial_state_from_mcp_resource(self, session: MCPSession) -> Any:
        """
        Fallback method to get initial state from MCP resources.
        This is kept for backward compatibility but should be replaced by control plane endpoints.
        """
        mcp_session = session._mcp_session
        initial_observation = None

        try:
            # List available resources - this is where initial state should come from
            logger.debug(f"Session {session.session_id}: Discovering MCP resources for initial state...")
            mcp_session_local = session._mcp_session
            if mcp_session_local is None:
                raise RuntimeError("Session not initialized while listing resources")
            resources_response = await mcp_session_local.list_resources()
            resources = resources_response.resources if hasattr(resources_response, "resources") else []
            logger.debug(f"Session {session.session_id}: Found {len(resources)} MCP resources")
            for resource in resources:
                logger.debug(f"Session {session.session_id}: Resource: {resource.name} | URI: {resource.uri}")

            # Try to identify initial state resource based on common patterns
            initial_state_resource = None
            for resource in resources:
                resource_name_lower = resource.name.lower()
                resource_uri_lower = str(resource.uri).lower()  # Convert AnyUrl to string first
                if any(
                    keyword in resource_name_lower or keyword in resource_uri_lower
                    for keyword in ["initial", "state", "observation", "start"]
                ):
                    initial_state_resource = resource
                    logger.debug(
                        f"Session {session.session_id}: ✅ Found initial state resource: {resource.name} | URI: {resource.uri}"
                    )
                    break

            if initial_state_resource:
                # Read the initial state resource
                logger.debug(
                    f"Session {session.session_id}: Reading initial state from resource: {initial_state_resource.uri}"
                )

                mcp_session_for_read = session._mcp_session
                if mcp_session_for_read is None:
                    raise RuntimeError("Session not initialized while reading resource")
                resource_content = await mcp_session_for_read.read_resource(initial_state_resource.uri)

                # Handle the new ResourceContents format
                text_value = getattr(resource_content, "text", None)
                if text_value is not None:
                    try:
                        initial_observation = json.loads(text_value)
                        logger.info(
                            f"Session {session.session_id}: ✅ Successfully parsed JSON initial state with grid_layout: {initial_observation.get('grid_layout', 'N/A')[:20]}..."
                        )
                    except json.JSONDecodeError:
                        initial_observation = {"observation": text_value}
                elif (
                    hasattr(resource_content, "contents")
                    and resource_content.contents
                    and len(resource_content.contents) > 0
                ):
                    # Fallback to old format for backward compatibility
                    content = resource_content.contents[0]
                    content_text = getattr(content, "text", None)
                    if content_text is not None:
                        try:
                            initial_observation = json.loads(content_text)
                        except json.JSONDecodeError:
                            initial_observation = {"observation": content_text}
                    else:
                        initial_observation = {"observation": str(resource_content)}
                else:
                    logger.warning(f"Session {session.session_id}: Resource content is empty or unrecognized format")
                    logger.warning(f"Session {session.session_id}: Unexpected resource format")
                    initial_state_resource = None  # Fall back to other options
            else:
                logger.warning(
                    f"Session {session.session_id}: ❌ No initial state resource found among {len(resources)} resources"
                )
                # Fallback: if no initial state resource, try first available resource
                if resources:
                    first_resource = resources[0]
                    logger.debug(
                        f"Session {session.session_id}: No initial state resource found, using first resource: {first_resource.name}"
                    )
                    logger.debug(
                        f"Session {session.session_id}: About to call mcp_session.read_resource with fallback URI: {first_resource.uri}"
                    )

                    mcp_session_for_fallback_read = session._mcp_session
                    if mcp_session_for_fallback_read is None:
                        raise RuntimeError("Session not initialized while reading fallback resource")
                    resource_content = await mcp_session_for_fallback_read.read_resource(first_resource.uri)

                    logger.debug(
                        f"Session {session.session_id}: fallback read_resource returned type: {type(resource_content)}"
                    )
                    logger.debug(
                        f"Session {session.session_id}: fallback read_resource returned value: {resource_content}"
                    )
                    logger.debug(
                        f"Session {session.session_id}: fallback read_resource dir(): {dir(resource_content)}"
                    )

                    # Handle the new ResourceContents format
                    text_value_2 = getattr(resource_content, "text", None)
                    if text_value_2 is not None:
                        try:
                            initial_observation = json.loads(text_value_2)
                        except json.JSONDecodeError:
                            initial_observation = {"observation": text_value_2}
                    elif (
                        hasattr(resource_content, "contents")
                        and resource_content.contents
                        and len(resource_content.contents) > 0
                    ):
                        # Fallback to old format for backward compatibility
                        content = resource_content.contents[0]
                        content_text_2 = getattr(content, "text", None)
                        if content_text_2 is not None:
                            try:
                                initial_observation = json.loads(content_text_2)
                            except json.JSONDecodeError:
                                initial_observation = {"observation": content_text_2}
                        else:
                            initial_observation = {"observation": str(content)}
                    else:
                        logger.warning(f"Session {session.session_id}: Fallback resource has unexpected format")
                        initial_observation = {"observation": str(resource_content)}
                else:
                    logger.debug(f"Session {session.session_id}: No resources available from MCP server")

        except Exception as e:
            # If resources are not available, fall back to a default observation
            # This maintains backward compatibility with servers that don't expose resources
            logger.warning(f"Session {session.session_id}: Failed to read initial state from MCP resources: {e}")
            logger.warning(f"Session {session.session_id}: Exception type: {type(e)}")
            logger.warning(f"Session {session.session_id}: Exception args: {e.args}")
            import traceback

            logger.warning(f"Session {session.session_id}: Full traceback: {traceback.format_exc()}")
            initial_observation = {
                "observation": "initial_state",
                "message": "Session established",
            }

        return initial_observation

    async def call_tool(self, session: MCPSession, tool_name: str, arguments: Dict) -> Tuple[Any, float, bool, Dict]:
        """
        Execute a tool call via MCP protocol with control plane separation.

        This method implements the control plane separation architecture:
        1. Execute tool call (data plane) - contains only observations
        2. Query control plane resources for reward/termination info
        3. Return combined result maintaining strict plane separation

        Args:
            session: The MCPSession to execute the tool call on
            tool_name: Name of the tool to call
            arguments: Arguments for the tool call

        Returns:
            Tuple of (observation, reward, done, info) with control plane data
        """
        if not session._mcp_session:
            raise RuntimeError("Session not initialized")

        mcp_session = session._mcp_session

        # 1. Execute the tool call via MCP protocol (DATA PLANE)
        tool_result = await mcp_session.call_tool(tool_name, arguments)

        # Extract data plane results (observation only)
        if tool_result.content and len(tool_result.content) > 0:
            content = tool_result.content[0]
            text_value = getattr(content, "text", None)
            if isinstance(text_value, str):
                # Fix: Handle empty or invalid JSON responses gracefully
                if text_value.strip() == "":
                    logger.warning(f"Session {session.session_id}: Empty tool response from {tool_name}")
                    observation = {
                        "observation": "empty_response",
                        "session_id": session.session_id,
                    }
                else:
                    try:
                        observation = json.loads(text_value)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Session {session.session_id}: Invalid JSON from {tool_name}: {text_value}. Error: {e}"
                        )
                        # Create a structured response from the raw text
                        observation = {
                            "observation": text_value,
                            "session_id": session.session_id,
                            "error": "invalid_json_response",
                        }
            else:
                # Handle non-text content
                observation = {
                    "observation": str(content),
                    "session_id": session.session_id,
                }
        else:
            # Handle completely empty tool result
            logger.warning(f"Session {session.session_id}: Tool {tool_name} returned empty result")
            observation = {
                "observation": "no_response",
                "session_id": session.session_id,
            }

        # 2. Query CONTROL PLANE endpoints for reward/termination info
        reward = 0.0
        terminated = False
        truncated = False
        control_plane_info = {}

        try:
            # Extract base URL and session ID from the MCP session
            base_url = session.base_url.rstrip("/").removesuffix("/mcp")
            # Use the session ID from the established MCP session
            session_id = session.session_id

            if session_id:
                headers = {"mcp-session-id": session_id}

                # Query reward endpoint
                try:
                    # Use shorter timeout for better responsiveness
                    timeout = 3.0
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        reward_response = await client.get(
                            f"{base_url}/control/reward",
                            headers=headers,
                            timeout=timeout,
                        )
                        if reward_response.status_code == 200:
                            reward_data = reward_response.json()
                            reward = reward_data.get("reward", 0.0)
                            control_plane_info["reward_source"] = "control_plane_endpoint"
                        else:
                            logger.warning(f"Control plane reward endpoint returned {reward_response.status_code}")
                except httpx.TimeoutException:
                    logger.warning(f"Control plane reward endpoint timed out after {timeout}s")
                except Exception as e:
                    logger.warning(f"Failed to query reward endpoint: {e}")

                # Query status endpoint
                try:
                    timeout = 3.0
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        status_response = await client.get(
                            f"{base_url}/control/status",
                            headers=headers,
                            timeout=timeout,
                        )
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            terminated = status_data.get("terminated", False)
                            truncated = status_data.get("truncated", False)
                            control_plane_info["status_source"] = "control_plane_endpoint"
                        else:
                            logger.warning(f"Control plane status endpoint returned {status_response.status_code}")
                except httpx.TimeoutException:
                    logger.warning(f"Control plane status endpoint timed out after {timeout}s")
                except Exception as e:
                    logger.warning(f"Failed to query status endpoint: {e}")

        except Exception as e:
            logger.warning(f"Failed to query control plane endpoints: {e}")

        # 3. Combine results maintaining strict separation
        done = terminated or truncated

        info = {
            "steps": observation.get("moves", observation.get("steps", 0)),
            "tool_call": tool_name,
            "arguments": arguments,
            "control_plane": control_plane_info,  # Mark control plane data
        }

        # Log control plane separation
        logger.debug(
            f"Session {session.session_id}: Data plane: {list(observation.keys())}, Control plane: reward={reward}, terminated={terminated}"
        )

        return observation, reward, done, info

    async def close_session(self, session: MCPSession) -> None:
        """
        Close an MCP session and clean up resources.

        Args:
            session: The MCPSession to close
        """
        if session._exit_stack:
            try:
                await session._exit_stack.aclose()
            except asyncio.CancelledError:
                # Handle cancellation gracefully (especially important for Python 3.12)
                logger.error(f"Session {session.session_id} close was cancelled")
            except Exception as e:
                # Hitting this error, probably because of use of threads: "Attempted to exit cancel scope in a different task than it was entered in"
                logger.error(f"Error closing session {session.session_id}: {e}")
            finally:
                session._exit_stack = None
                session._mcp_session = None
