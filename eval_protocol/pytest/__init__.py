"""
eval_protocol.pytest - Pytest integration for evaluation testing.

This module uses lazy loading to minimize import time.
Heavy dependencies (litellm, torch, etc.) are only loaded when needed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Lazy imports mapping: name -> (module_path, attr_name)
# These are loaded on-demand when accessed
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Rollout processors
    "AgentRolloutProcessor": (".default_agent_rollout_processor", "AgentRolloutProcessor"),
    "MCPGymRolloutProcessor": (".default_mcp_gym_rollout_processor", "MCPGymRolloutProcessor"),
    "NoOpRolloutProcessor": (".default_no_op_rollout_processor", "NoOpRolloutProcessor"),
    "SingleTurnRolloutProcessor": (".default_single_turn_rollout_process", "SingleTurnRolloutProcessor"),
    "RemoteRolloutProcessor": (".remote_rollout_processor", "RemoteRolloutProcessor"),
    "GithubActionRolloutProcessor": (".github_action_rollout_processor", "GithubActionRolloutProcessor"),
    "RolloutProcessor": (".rollout_processor", "RolloutProcessor"),
    # Dataset adapter
    "default_dataset_adapter": (".default_dataset_adapter", "default_dataset_adapter"),
    # Core decorator
    "evaluation_test": (".evaluation_test", "evaluation_test"),
    # Exception handling
    "ExceptionHandlerConfig": (".exception_config", "ExceptionHandlerConfig"),
    "BackoffConfig": (".exception_config", "BackoffConfig"),
    "get_default_exception_handler_config": (".exception_config", "get_default_exception_handler_config"),
    # Post processors
    "RolloutResultPostProcessor": (".rollout_result_post_processor", "RolloutResultPostProcessor"),
    "NoOpRolloutResultPostProcessor": (".rollout_result_post_processor", "NoOpRolloutResultPostProcessor"),
    # Types
    "RolloutProcessorConfig": (".types", "RolloutProcessorConfig"),
}

# Optional imports that may not be available
_OPTIONAL_IMPORTS: dict[str, tuple[str, str]] = {
    "KlavisSandboxRolloutProcessor": (".default_klavis_sandbox_rollout_processor", "KlavisSandboxRolloutProcessor"),
    "PydanticAgentRolloutProcessor": (".default_pydantic_ai_rollout_processor", "PydanticAgentRolloutProcessor"),
    "LangGraphRolloutProcessor": (".default_langchain_rollout_processor", "LangGraphRolloutProcessor"),
}

# Track which optional imports are available (set on first access)
_optional_availability: dict[str, bool] = {}


def __getattr__(name: str):
    """Lazy load attributes on first access."""
    # Handle lazy imports
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package="eval_protocol.pytest")
        value = getattr(module, attr_name)
        # Cache in module namespace for future access
        globals()[name] = value
        return value

    # Handle optional imports
    if name in _OPTIONAL_IMPORTS:
        module_path, attr_name = _OPTIONAL_IMPORTS[name]
        try:
            module = importlib.import_module(module_path, package="eval_protocol.pytest")
            value = getattr(module, attr_name)
            globals()[name] = value
            _optional_availability[name] = True
            return value
        except ImportError:
            _optional_availability[name] = False
            return None

    # Handle availability flags
    if name == "KLAVIS_AVAILABLE":
        if "KlavisSandboxRolloutProcessor" not in _optional_availability:
            # Trigger the import to check availability
            __getattr__("KlavisSandboxRolloutProcessor")
        return _optional_availability.get("KlavisSandboxRolloutProcessor", False)

    if name == "PYDANTIC_AI_AVAILABLE":
        if "PydanticAgentRolloutProcessor" not in _optional_availability:
            __getattr__("PydanticAgentRolloutProcessor")
        return _optional_availability.get("PydanticAgentRolloutProcessor", False)

    if name == "LANGCHAIN_AVAILABLE":
        if "LangGraphRolloutProcessor" not in _optional_availability:
            __getattr__("LangGraphRolloutProcessor")
        return _optional_availability.get("LangGraphRolloutProcessor", False)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available attributes for tab completion."""
    return list(__all__) + ["KLAVIS_AVAILABLE", "PYDANTIC_AI_AVAILABLE", "LANGCHAIN_AVAILABLE"]


__all__ = [
    # Rollout processors
    "AgentRolloutProcessor",
    "MCPGymRolloutProcessor",
    "RolloutProcessor",
    "SingleTurnRolloutProcessor",
    "RemoteRolloutProcessor",
    "GithubActionRolloutProcessor",
    "NoOpRolloutProcessor",
    # Dataset
    "default_dataset_adapter",
    # Types
    "RolloutProcessorConfig",
    # Core
    "evaluation_test",
    # Exception handling
    "ExceptionHandlerConfig",
    "BackoffConfig",
    "get_default_exception_handler_config",
    # Post processors
    "RolloutResultPostProcessor",
    "NoOpRolloutResultPostProcessor",
    # Optional (may be None if dependencies not installed)
    "KlavisSandboxRolloutProcessor",
    "PydanticAgentRolloutProcessor",
    "LangGraphRolloutProcessor",
]


# Type hints for IDE support (not executed at runtime)
if TYPE_CHECKING:
    from .default_agent_rollout_processor import AgentRolloutProcessor as AgentRolloutProcessor
    from .default_dataset_adapter import default_dataset_adapter as default_dataset_adapter
    from .default_mcp_gym_rollout_processor import MCPGymRolloutProcessor as MCPGymRolloutProcessor
    from .default_no_op_rollout_processor import NoOpRolloutProcessor as NoOpRolloutProcessor
    from .default_single_turn_rollout_process import SingleTurnRolloutProcessor as SingleTurnRolloutProcessor
    from .remote_rollout_processor import RemoteRolloutProcessor as RemoteRolloutProcessor
    from .github_action_rollout_processor import GithubActionRolloutProcessor as GithubActionRolloutProcessor
    from .evaluation_test import evaluation_test as evaluation_test
    from .exception_config import (
        ExceptionHandlerConfig as ExceptionHandlerConfig,
        BackoffConfig as BackoffConfig,
        get_default_exception_handler_config as get_default_exception_handler_config,
    )
    from .rollout_processor import RolloutProcessor as RolloutProcessor
    from .rollout_result_post_processor import (
        RolloutResultPostProcessor as RolloutResultPostProcessor,
        NoOpRolloutResultPostProcessor as NoOpRolloutResultPostProcessor,
    )
    from .types import RolloutProcessorConfig as RolloutProcessorConfig
    from .default_klavis_sandbox_rollout_processor import (
        KlavisSandboxRolloutProcessor as KlavisSandboxRolloutProcessor,
    )
    from .default_pydantic_ai_rollout_processor import (
        PydanticAgentRolloutProcessor as PydanticAgentRolloutProcessor,
    )
    from .default_langchain_rollout_processor import (
        LangGraphRolloutProcessor as LangGraphRolloutProcessor,
    )
