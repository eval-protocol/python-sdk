"""
Reward-Kit MCP Integration Framework

This module provides utilities for creating MCP servers that integrate
with reward-kit environments and evaluation workflows.

It also provides the refactored MCP environment components for better modularity.
"""

from .adapter import EnvironmentAdapter

# New refactored components
from .client import MCPConnectionManager
from .execution import ExecutionManager, LLMBasePolicy, OpenAIPolicy

# FireworksPolicy is imported conditionally by execution.__init__.py
try:
    from .execution import FireworksPolicy
except ImportError:
    FireworksPolicy = None

from ..types import DatasetRow, MCPSession, MCPToolCall, Trajectory

# North Star MCP-Gym Framework
from .mcpgym import McpGym
from .session import GeneralMCPVectorEnv
from .simulation_server import SimulationServerBase

__all__ = [
    # Legacy MCP server components
    "EnvironmentAdapter",
    "SimulationServerBase",
    # New refactored components
    "MCPConnectionManager",
    "LLMBasePolicy",
    "OpenAIPolicy",
    "ExecutionManager",
    "GeneralMCPVectorEnv",
    "MCPSession",
    "MCPToolCall",
    "DatasetRow",
    "Trajectory",
    # North Star MCP-Gym Framework
    "McpGym",
]

# Only export FireworksPolicy if it's available
if FireworksPolicy is not None:
    __all__.insert(__all__.index("OpenAIPolicy") + 1, "FireworksPolicy")
