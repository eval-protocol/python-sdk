"""
MCP Connection Management

Handles MCP client connections, session initialization, and resource/tool discovery.
Extracted from mcp_env.py to improve modularity.
"""

import asyncio
import hashlib
import json
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from ..types import MCPSession

logger = logging.getLogger(__name__)


class MCPConnectionManager:
    """Manages MCP client connections and session lifecycle."""

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
                    logger.debug(
                        f"Session {session.session_id} reinit close was cancelled"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error closing existing session {session.session_id} during reinit: {e}"
                    )
                finally:
                    session._exit_stack = None
            session._mcp_session = None

        exit_stack = AsyncExitStack()

        client_info = None
        if session.seed is not None or (
            session.dataset_row and session.dataset_row.environment_context
        ):
            from mcp.types import Implementation

            client_info = Implementation(name="reward-kit", version="1.0.0", _extra={})
            if session.seed is not None:
                client_info._extra["seed"] = session.seed
            if session.dataset_row and session.dataset_row.environment_context:
                client_info._extra["config"] = session.dataset_row.environment_context

        read_stream, write_stream, _ = await exit_stack.enter_async_context(
            streamablehttp_client(session.base_url, terminate_on_close=True)
        )

        mcp_session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream, client_info=client_info)
        )

        await mcp_session.initialize()

        session._mcp_session = mcp_session
        session._exit_stack = exit_stack

        # Update session ID to match server's calculation (for control plane sync)
        if client_info and hasattr(client_info, "_extra"):
            extra_data = client_info._extra
            if extra_data and isinstance(extra_data, dict):

                seed_value = extra_data.get("seed")
                config_value = extra_data.get("config", {})

                stable_data = {
                    "seed": seed_value,
                    "config": config_value,
                    "name": client_info.name,
                    "version": client_info.version,
                }

                stable_str = json.dumps(stable_data, sort_keys=True)
                server_session_id = hashlib.md5(stable_str.encode()).hexdigest()

                # Update the session ID to match what the server generated
                session.session_id = server_session_id
                logger.debug(f"Updated session ID to match server: {server_session_id}")

    async def discover_tools(self, session: MCPSession) -> List[Dict]:
        """
        Discover available tools from an MCP session.

        Args:
            session: The MCPSession to discover tools from

        Returns:
            List of tool schemas
        """
        if not session._mcp_session:
            raise RuntimeError("Session not initialized")

        mcp_session = session._mcp_session

        # Get available tools from MCP server
        tools_response = await mcp_session.list_tools()
        tools = tools_response.tools if hasattr(tools_response, "tools") else []

        # Convert tools to schema format - filter out internal tools
        tool_schemas = []
        for tool in tools:
            # Only expose action tools to the model, not internal state tools
            tool_schema = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": (
                    tool.inputSchema if hasattr(tool, "inputSchema") else {}
                ),
            }
            tool_schemas.append(tool_schema)

        return tool_schemas

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
            import httpx

            # Extract base URL and session ID from the MCP session
            base_url = session.base_url.rstrip("/mcp").rstrip("/")
            session_id = session.session_id

            if session_id:
                headers = {"mcp-session-id": session_id}

                # Query initial state endpoint
                try:
                    # Use shorter timeout for playback mode
                    timeout = (
                        3.0
                        if hasattr(session, "_is_playback_mode")
                        and session._is_playback_mode
                        else 5.0
                    )
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
                    logger.warning(
                        f"Control plane initial state endpoint timed out after {timeout}s"
                    )
                except Exception as e:
                    logger.warning(f"Failed to query initial state endpoint: {e}")

        except Exception as e:
            logger.warning(f"Failed to query control plane initial state endpoint: {e}")

        # Fallback to MCP resource if control plane endpoint fails (backward compatibility)
        if initial_observation is None:
            logger.debug(
                f"Session {session.session_id}: Falling back to MCP resource for initial state"
            )
            initial_observation = await self._get_initial_state_from_mcp_resource(
                session
            )

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
            logger.debug(
                f"Session {session.session_id}: Discovering MCP resources for initial state..."
            )
            resources_response = await mcp_session.list_resources()
            resources = (
                resources_response.resources
                if hasattr(resources_response, "resources")
                else []
            )
            logger.debug(
                f"Session {session.session_id}: Found {len(resources)} MCP resources"
            )
            for resource in resources:
                logger.debug(
                    f"Session {session.session_id}: Resource: {resource.name} | URI: {resource.uri}"
                )

            # Try to identify initial state resource based on common patterns
            initial_state_resource = None
            for resource in resources:
                resource_name_lower = resource.name.lower()
                resource_uri_lower = str(
                    resource.uri
                ).lower()  # Convert AnyUrl to string first
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

                resource_content = await mcp_session.read_resource(
                    initial_state_resource.uri
                )

                # Handle the new ResourceContents format
                if hasattr(resource_content, "text"):
                    try:
                        initial_observation = json.loads(resource_content.text)
                        logger.info(
                            f"Session {session.session_id}: ✅ Successfully parsed JSON initial state with grid_layout: {initial_observation.get('grid_layout', 'N/A')[:20]}..."
                        )
                    except json.JSONDecodeError:
                        initial_observation = {"observation": resource_content.text}
                elif (
                    hasattr(resource_content, "contents")
                    and resource_content.contents
                    and len(resource_content.contents) > 0
                ):
                    # Fallback to old format for backward compatibility
                    content = resource_content.contents[0]
                    if hasattr(content, "text"):
                        try:
                            initial_observation = json.loads(content.text)
                        except json.JSONDecodeError:
                            initial_observation = {"observation": content.text}
                    else:
                        initial_observation = {"observation": str(resource_content)}
                else:
                    logger.warning(
                        f"Session {session.session_id}: Resource content is empty or unrecognized format"
                    )
                    logger.warning(
                        f"Session {session.session_id}: Unexpected resource format"
                    )
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

                    resource_content = await mcp_session.read_resource(
                        first_resource.uri
                    )

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
                    if hasattr(resource_content, "text"):
                        try:
                            initial_observation = json.loads(resource_content.text)
                        except json.JSONDecodeError:
                            initial_observation = {"observation": resource_content.text}
                    elif (
                        hasattr(resource_content, "contents")
                        and resource_content.contents
                        and len(resource_content.contents) > 0
                    ):
                        # Fallback to old format for backward compatibility
                        content = resource_content.contents[0]
                        if hasattr(content, "text"):
                            try:
                                initial_observation = json.loads(content.text)
                            except json.JSONDecodeError:
                                initial_observation = {"observation": content.text}
                        else:
                            initial_observation = {"observation": str(content)}
                    else:
                        logger.warning(
                            f"Session {session.session_id}: Fallback resource has unexpected format"
                        )
                        initial_observation = {"observation": str(resource_content)}
                else:
                    logger.debug(
                        f"Session {session.session_id}: No resources available from MCP server"
                    )

        except Exception as e:
            # If resources are not available, fall back to a default observation
            # This maintains backward compatibility with servers that don't expose resources
            logger.warning(
                f"Session {session.session_id}: Failed to read initial state from MCP resources: {e}"
            )
            logger.warning(f"Session {session.session_id}: Exception type: {type(e)}")
            logger.warning(f"Session {session.session_id}: Exception args: {e.args}")
            import traceback

            logger.warning(
                f"Session {session.session_id}: Full traceback: {traceback.format_exc()}"
            )
            initial_observation = {
                "observation": "initial_state",
                "message": "Session established",
            }

        return initial_observation

    async def call_tool(
        self, session: MCPSession, tool_name: str, arguments: Dict
    ) -> Tuple[Any, float, bool, Dict]:
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
            if hasattr(content, "text"):
                # Fix: Handle empty or invalid JSON responses gracefully
                if not content.text or content.text.strip() == "":
                    logger.warning(
                        f"Session {session.session_id}: Empty tool response from {tool_name}"
                    )
                    observation = {
                        "observation": "empty_response",
                        "session_id": session.session_id,
                    }
                else:
                    try:
                        observation = json.loads(content.text)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Session {session.session_id}: Invalid JSON from {tool_name}: {content.text}. Error: {e}"
                        )
                        # Create a structured response from the raw text
                        observation = {
                            "observation": content.text,
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
            logger.warning(
                f"Session {session.session_id}: Tool {tool_name} returned empty result"
            )
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
            # Query control plane endpoints following the new architecture
            import httpx

            # Extract base URL and session ID from the MCP session
            base_url = session.base_url.rstrip("/mcp").rstrip("/")
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
                            control_plane_info["reward_source"] = (
                                "control_plane_endpoint"
                            )
                        else:
                            logger.warning(
                                f"Control plane reward endpoint returned {reward_response.status_code}"
                            )
                except httpx.TimeoutException:
                    logger.warning(
                        f"Control plane reward endpoint timed out after {timeout}s"
                    )
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
                            control_plane_info["status_source"] = (
                                "control_plane_endpoint"
                            )
                        else:
                            logger.warning(
                                f"Control plane status endpoint returned {status_response.status_code}"
                            )
                except httpx.TimeoutException:
                    logger.warning(
                        f"Control plane status endpoint timed out after {timeout}s"
                    )
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
                logger.debug(f"Session {session.session_id} close was cancelled")
            except Exception as e:
                logger.warning(f"Error closing session {session.session_id}: {e}")
            finally:
                session._exit_stack = None
                session._mcp_session = None
