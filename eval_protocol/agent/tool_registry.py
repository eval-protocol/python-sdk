"""
Tool Registry for the Agent Evaluation Framework.
Provides a mechanism to register and manage tools.
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional


class ToolRegistry:
    """
    Registry for tools that can be used by agents.

    Attributes:
        name: Unique identifier for this tool registry
        tools: Dictionary mapping tool names to tool functions
        descriptions: Dictionary mapping tool names to their descriptions
        parameters: Dictionary mapping tool names to their parameter specifications
    """

    def __init__(self, name: str):
        """
        Initialize a new tool registry.

        Args:
            name: Unique identifier for this registry
        """
        self.name = name
        self.tools: Dict[str, Callable] = {}
        self.descriptions: Dict[str, str] = {}
        self.parameters: Dict[str, Dict[str, Any]] = {}

    def tool(self, description: str, parameters: Dict[str, Any]) -> Callable:
        """
        Decorator to register a function as a tool.

        Args:
            description: Human-readable description of the tool
            parameters: Parameter specifications for the tool

        Returns:
            Decorator function that registers the decorated function
        """

        def decorator(func: Callable) -> Callable:
            tool_name = func.__name__
            self.tools[tool_name] = func
            self.descriptions[tool_name] = description
            self.parameters[tool_name] = parameters

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """
        Get a tool function by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            The tool function if found, None otherwise
        """
        return self.tools.get(tool_name)

    def get_tools(self) -> Dict[str, Callable]:
        """
        Get all tools in this registry.

        Returns:
            Dictionary mapping tool names to tool functions
        """
        return self.tools

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Get tool specifications in OpenAI function calling format.

        Returns:
            List of tool specifications compatible with OpenAI API
        """
        tools = []
        for tool_name in self.tools:
            tools.append(
                {
                    "name": tool_name,
                    "description": self.descriptions.get(tool_name, ""),
                    "parameters": {
                        "type": "object",
                        "properties": self.parameters.get(tool_name, {}),
                        "required": list(self.parameters.get(tool_name, {}).keys()),
                    },
                }
            )
        return tools

    def create_fastapi_app(self):
        """
        Create a FastAPI app with endpoints for each tool.

        Returns:
            A FastAPI app instance with tool endpoints
        """
        # This is a stub implementation
        return {"app_type": "FastAPI", "tools": list(self.tools.keys())}
