"""
MCP Client Connection Management

This module handles MCP client connections, session initialization,
and resource/tool discovery.
"""

from .connection import MCPConnectionManager

__all__ = [
    "MCPConnectionManager",
]
