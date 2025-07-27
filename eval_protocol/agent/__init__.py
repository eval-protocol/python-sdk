"""
Eval Protocol Agent Evaluation Framework V2 Components.

This package contains the core components for the new, resource-centric
agent evaluation framework, including the ForkableResource ABC, Orchestrator,
and concrete resource implementations.
"""

from .orchestrator import Orchestrator

# Make key components easily importable from eval_protocol.agent
from .resource_abc import ForkableResource
from .resources import (
    DockerResource,
    FileSystemResource,
    PythonStateResource,
    SQLResource,
)
from .tool_registry import ToolRegistry

__all__ = [
    "ForkableResource",
    "Orchestrator",
    "PythonStateResource",
    "SQLResource",
    "FileSystemResource",
    "DockerResource",
    "ToolRegistry",
]
