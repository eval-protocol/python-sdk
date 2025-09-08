"""
Tau2-Bench MCP Server

This module provides MCP server implementations for tau2-bench domains
(airline, mock, retail) along with test data and system prompts.
"""

import importlib.resources
from pathlib import Path


def get_server_script_path() -> str:
    """Get the path to the tau2 MCP server script."""
    try:
        # Try to get from installed package. __package__ can be None during some tooling.
        package = __package__ if __package__ is not None else __name__
        with importlib.resources.as_file(importlib.resources.files(package) / "server.py") as server_path:
            return str(server_path)
    except (ImportError, FileNotFoundError):
        # Fallback for development environment
        return str(Path(__file__).parent / "server.py")


def get_system_prompt(domain: str) -> str:
    """Get system prompt for the specified domain.

    Args:
        domain: Domain name (airline, mock, retail)

    Returns:
        System prompt text
    """
    prompt_filename = f"{domain}_agent_system_prompt.md"

    try:
        # Try to get from installed package
        with importlib.resources.open_text(f"{__package__}.tests.system_prompts", prompt_filename) as f:
            return f.read().strip()
    except (ImportError, FileNotFoundError):
        # Fallback for development environment
        prompt_path = Path(__file__).parent / "tests" / "system_prompts" / prompt_filename
        with open(prompt_path, "r") as f:
            return f.read().strip()


def get_retail_system_prompt() -> str:
    """Get the retail domain system prompt."""
    return get_system_prompt("retail")


# Re-export the main MCP classes for convenience
from .tau2_mcp import AirlineDomainMcp, MockDomainMcp, RetailDomainMcp

__all__ = [
    "get_server_script_path",
    "get_system_prompt",
    "get_retail_system_prompt",
    "AirlineDomainMcp",
    "MockDomainMcp",
    "RetailDomainMcp",
]
