import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
from typing import Optional


class FunctionLike(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Any = None


from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult
from openai.types import FunctionDefinition

from eval_protocol.models import (
    MCPConfigurationServerStdio,
    MCPConfigurationServerUrl,
    MCPMultiClientConfiguration,
)

load_dotenv()  # load environment variables from .env


class MCPMultiClient:
    """
    Implements what clients like Cursor and Claude Desktop do when you configure
    them to use multiple MCP servers. The difference is that it validates
    against a list of environment variables rather than injects them into the
    MCP server process. This is so you can version control your configuration
    without exposing your environment variables to the MCP server process.

    Environment variables should instead be set in a .env file
    """

    def __init__(self, config_path: Optional[str] = None):
        # Initialize session and client objects
        self.sessions: Dict[str, ClientSession] = {}
        self.tools_to_sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str] = None) -> MCPMultiClientConfiguration:
        """Load MCP server configuration from file or use default"""
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                return MCPMultiClientConfiguration(**json.load(f))

        # Default configuration - can be overridden by config file
        return MCPMultiClientConfiguration(mcpServers={})

    def _validate_environment_variables(self, server_name: str, required_env: List[str]) -> None:
        """Validate that required environment variables are set in os.environ"""
        missing_vars = []
        for env_var in required_env:
            if env_var not in os.environ:
                missing_vars.append(env_var)

        if missing_vars:
            raise ValueError(
                f"Server '{server_name}' requires the following environment variables "
                f"to be set in os.environ: {missing_vars}. "
                f"Please set these variables in your environment or .env file."
            )

    def _process_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Process headers by substituting environment variables.

        Supports environment variable substitution in the format:
        - ${ENV_VAR} or $ENV_VAR for environment variables
        - Raw strings are passed through unchanged

        Example:
            {"Authorization": "Bearer ${API_KEY}"}
            -> {"Authorization": "Bearer abc123"} (if API_KEY=abc123)
        """
        import re

        processed_headers = {}
        for key, value in headers.items():
            # Match ${VAR} or $VAR patterns
            def replace_env_var(match):
                var_name = match.group(1) or match.group(2)
                env_value = os.environ.get(var_name)
                if env_value is None:
                    raise ValueError(
                        f"Environment variable '{var_name}' referenced in header '{key}' "
                        f"is not set. Please set it in your environment or .env file."
                    )
                return env_value

            # Replace ${VAR} or $VAR with environment variable value
            processed_value = re.sub(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replace_env_var, value)
            processed_headers[key] = processed_value

        return processed_headers

    async def connect_to_servers(self):
        """Connect to all configured MCP servers"""
        if not self.config.mcpServers:
            print("No MCP servers configured. Please provide a configuration file.")
            return

        for server_name, server_config in self.config.mcpServers.items():
            await self._connect_to_server(server_name, server_config)

    async def _connect_to_server(
        self, server_name: str, server_config: Union[MCPConfigurationServerStdio, MCPConfigurationServerUrl]
    ):
        """Connect to a specific MCP server using its configuration"""
        session: ClientSession

        if isinstance(server_config, MCPConfigurationServerStdio):
            # Handle stdio-based MCP server
            command = server_config.command
            args = server_config.args
            env_config = server_config.env

            if not command:
                raise ValueError(f"Server '{server_name}' must have a 'command' specified")

            # Validate that required environment variables are set
            if env_config:
                self._validate_environment_variables(server_name, env_config)

            # Use the current system environment (os.environ) - convert to plain dict for typing compatibility
            server_params = StdioServerParameters(command=command, args=args, env=dict(os.environ))

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

        elif isinstance(server_config, MCPConfigurationServerUrl):
            # Handle HTTP-based MCP server
            url = server_config.url
            if not url:
                raise ValueError(f"Server '{server_name}' must have a 'url' specified")

            # Build headers only from the authorization field (Responses-style)
            processed_headers: Dict[str, str] = {}
            auth_token = getattr(server_config, "authorization", None)
            if auth_token:
                # Support env substitution in the authorization value as well
                processed_headers = self._process_headers({"Authorization": auth_token})

            # Connect using streamable HTTP client with auth headers
            http_transport = await self.exit_stack.enter_async_context(
                streamablehttp_client(url, headers=processed_headers)
            )
            read_stream, write_stream, get_session_id = http_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        else:
            raise ValueError(f"Unsupported server configuration type: {type(server_config)}")

        await session.initialize()
        self.sessions[server_name] = session

        # List available tools
        response = await session.list_tools()
        tools = response.tools
        for tool in tools:
            if tool.name in self.tools_to_sessions:
                raise ValueError(f"Tool '{tool.name}' already exists")
            self.tools_to_sessions[tool.name] = session
        print(
            f"\nConnected to server '{server_name}' with tools:",
            [tool.name for tool in tools],
        )

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools from all connected servers"""
        all_tools = []
        for server_name, session in self.sessions.items():
            try:
                response = await session.list_tools()
                for tool in response.tools:
                    all_tools.append(
                        {
                            "type": "function",
                            "function": FunctionLike(
                                name=tool.name,
                                description=tool.description,
                                parameters=tool.inputSchema,
                            ),
                        }
                    )
            except Exception as e:
                print(f"Error listing tools from server '{server_name}': {e}")

        return all_tools

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Union[CallToolResult, str]:
        """Call a specific tool by name with arguments"""

        session = self.tools_to_sessions[tool_name]
        try:
            result = await session.call_tool(tool_name, tool_args)
            return result
        except Exception as e:
            return f"Error calling tool {tool_name}: {e}"

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
