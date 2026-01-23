"""
Fireworks Eval Protocol - Simplify reward modeling and evaluation for LLM RL fine-tuning.

A Python library for defining, testing, deploying, and using reward functions
for LLM fine-tuning, including launching full RL jobs on the Fireworks platform.

The library also provides an agent evaluation framework for testing and evaluating
tool-augmented models using self-contained task bundles.
"""

import importlib
import sys
import warnings
from typing import TYPE_CHECKING

import litellm

litellm.disable_add_transform_inline_image_block = True

warnings.filterwarnings("default", category=DeprecationWarning, module="eval_protocol")

# Eager imports for symbols that conflict with module names - ONLY when pytest is running.
# The reward_function.py module exports RewardFunction class, and we also export the
# reward_function decorator from typed_interface. When pytest's AssertionRewritingHook
# imports eval_protocol.reward_function as a module, Python would set
# eval_protocol.reward_function to the module, shadowing our function export.
#
# We detect pytest by checking if _pytest or pytest is already loaded. This avoids
# the ~500ms import overhead for non-test scenarios like the CLI.
_running_under_pytest = "_pytest" in sys.modules or "pytest" in sys.modules
if _running_under_pytest:
    from .reward_function import RewardFunction  # noqa: E402
    from .typed_interface import reward_function  # noqa: E402

# Lazy import mappings: name -> (module_path, attribute_name or None for module import)
_LAZY_IMPORTS = {
    # From .auth
    "get_fireworks_account_id": (".auth", "get_fireworks_account_id"),
    "get_fireworks_api_key": (".auth", "get_fireworks_api_key"),
    # From .common_utils
    "load_jsonl": (".common_utils", "load_jsonl"),
    # From .config
    "RewardKitConfig": (".config", "RewardKitConfig"),
    "get_config": (".config", "get_config"),
    "load_config": (".config", "load_config"),
    # From .mcp_env
    "AnthropicPolicy": (".mcp_env", "AnthropicPolicy"),
    "FireworksPolicy": (".mcp_env", "FireworksPolicy"),
    "LiteLLMPolicy": (".mcp_env", "LiteLLMPolicy"),
    "OpenAIPolicy": (".mcp_env", "OpenAIPolicy"),
    "make": (".mcp_env", "make"),
    "rollout": (".mcp_env", "rollout"),
    "test_mcp": (".mcp_env", "test_mcp"),
    # From .data_loader
    "DynamicDataLoader": (".data_loader", "DynamicDataLoader"),
    "InlineDataLoader": (".data_loader", "InlineDataLoader"),
    # Submodules (accessible as eval_protocol.submodule)
    "mcp": (".mcp", None),
    "rewards": (".rewards", None),
    "models": (".models", None),
    "auth": (".auth", None),
    "config": (".config", None),
    # From .models
    "EvaluateResult": (".models", "EvaluateResult"),
    "Message": (".models", "Message"),
    "MetricResult": (".models", "MetricResult"),
    "EvaluationRow": (".models", "EvaluationRow"),
    "InputMetadata": (".models", "InputMetadata"),
    "Status": (".models", "Status"),
    # From .playback_policy
    "PlaybackPolicyBase": (".playback_policy", "PlaybackPolicyBase"),
    # From .resources
    "create_llm_resource": (".resources", "create_llm_resource"),
    # From .reward_function
    "RewardFunction": (".reward_function", "RewardFunction"),
    # From .typed_interface
    "reward_function": (".typed_interface", "reward_function"),
    # From .quickstart.aha_judge
    "aha_judge": (".quickstart.aha_judge", "aha_judge"),
    # From .utils.evaluation_row_utils
    "multi_turn_assistant_to_ground_truth": (".utils.evaluation_row_utils", "multi_turn_assistant_to_ground_truth"),
    "assistant_to_ground_truth": (".utils.evaluation_row_utils", "assistant_to_ground_truth"),
    "filter_longest_conversation": (".utils.evaluation_row_utils", "filter_longest_conversation"),
    # From .pytest
    "evaluation_test": (".pytest", "evaluation_test"),
    "SingleTurnRolloutProcessor": (".pytest", "SingleTurnRolloutProcessor"),
    "RemoteRolloutProcessor": (".pytest", "RemoteRolloutProcessor"),
    "GithubActionRolloutProcessor": (".pytest", "GithubActionRolloutProcessor"),
    # From .pytest.parameterize
    "DefaultParameterIdGenerator": (".pytest.parameterize", "DefaultParameterIdGenerator"),
    # From .log_utils
    "ElasticsearchDirectHttpHandler": (
        ".log_utils.elasticsearch_direct_http_handler",
        "ElasticsearchDirectHttpHandler",
    ),
    "RolloutIdFilter": (".log_utils.rollout_id_filter", "RolloutIdFilter"),
    "setup_rollout_logging_for_elasticsearch_handler": (
        ".log_utils.util",
        "setup_rollout_logging_for_elasticsearch_handler",
    ),
    "FireworksTracingHttpHandler": (".log_utils.fireworks_tracing_http_handler", "FireworksTracingHttpHandler"),
    "ElasticsearchConfig": (".log_utils.elasticsearch_client", "ElasticsearchConfig"),
    # From .types.remote_rollout_processor
    "InitRequest": (".types.remote_rollout_processor", "InitRequest"),
    "RolloutMetadata": (".types.remote_rollout_processor", "RolloutMetadata"),
    "StatusResponse": (".types.remote_rollout_processor", "StatusResponse"),
    "create_langfuse_config_tags": (".types.remote_rollout_processor", "create_langfuse_config_tags"),
    "DataLoaderConfig": (".types.remote_rollout_processor", "DataLoaderConfig"),
}

# Optional imports that may not be available
_OPTIONAL_IMPORTS = {
    "OpenAIResponsesAdapter": (".adapters", "OpenAIResponsesAdapter"),
    "LangfuseAdapter": (".adapters", "LangfuseAdapter"),
    "create_langfuse_adapter": (".adapters", "create_langfuse_adapter"),
    "BraintrustAdapter": (".adapters", "BraintrustAdapter"),
    "create_braintrust_adapter": (".adapters", "create_braintrust_adapter"),
    "LangSmithAdapter": (".adapters", "LangSmithAdapter"),
    "WeaveAdapter": (".adapters", "WeaveAdapter"),
    "create_app": (".proxy", "create_app"),
    "AuthProvider": (".proxy", "AuthProvider"),
    "AccountInfo": (".proxy", "AccountInfo"),
}


def __getattr__(name: str):
    """Lazy import handler for module-level attributes."""
    # Check regular lazy imports
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package="eval_protocol")
        if attr_name is None:
            # It's a submodule import
            return module
        return getattr(module, attr_name)

    # Check optional imports
    if name in _OPTIONAL_IMPORTS:
        module_path, attr_name = _OPTIONAL_IMPORTS[name]
        try:
            module = importlib.import_module(module_path, package="eval_protocol")
            return getattr(module, attr_name)
        except ImportError:
            # Return None or a placeholder for optional imports
            if name in ("create_app",):

                def create_app(*args, **kwargs):
                    raise ImportError(
                        "Proxy functionality requires additional dependencies. "
                        "Please install with: pip install eval-protocol[proxy]"
                    )

                return create_app
            elif name in ("AuthProvider", "AccountInfo"):

                class _Placeholder:
                    def __init__(self, *args, **kwargs):
                        raise ImportError(
                            "Proxy functionality requires additional dependencies. "
                            "Please install with: pip install eval-protocol[proxy]"
                        )

                return _Placeholder
            return None

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ElasticsearchConfig",
    "ElasticsearchDirectHttpHandler",
    "RolloutIdFilter",
    "setup_rollout_logging_for_elasticsearch_handler",
    "DataLoaderConfig",
    "Status",
    "RemoteRolloutProcessor",
    "GithubActionRolloutProcessor",
    "InputMetadata",
    "EvaluationRow",
    "DefaultParameterIdGenerator",
    "DynamicDataLoader",
    "InlineDataLoader",
    "aha_judge",
    "multi_turn_assistant_to_ground_truth",
    "assistant_to_ground_truth",
    "filter_longest_conversation",
    "evaluation_test",
    "SingleTurnRolloutProcessor",
    "OpenAIResponsesAdapter",
    "LangfuseAdapter",
    "create_langfuse_adapter",
    "BraintrustAdapter",
    "create_braintrust_adapter",
    "LangSmithAdapter",
    "FireworksTracingHttpHandler",
    # Core interfaces
    "Message",
    "MetricResult",
    "EvaluateResult",
    "reward_function",
    "RewardFunction",
    # Authentication
    "get_fireworks_api_key",
    "get_fireworks_account_id",
    # Configuration
    "load_config",
    "get_config",
    "RewardKitConfig",
    # Utilities
    "load_jsonl",
    # MCP Environment API
    "make",
    "rollout",
    "LiteLLMPolicy",
    "AnthropicPolicy",
    "FireworksPolicy",
    "OpenAIPolicy",
    "test_mcp",
    # Playback functionality
    "PlaybackPolicyBase",
    # Resource management
    "create_llm_resource",
    # Submodules
    "rewards",
    "mcp",
    # Remote server types
    "InitRequest",
    "RolloutMetadata",
    "StatusResponse",
    "create_langfuse_config_tags",
    # Proxy
    "create_app",
    "AuthProvider",
    "AccountInfo",
]

# Version is loaded lazily too
_version_info = None


def _get_version():
    global _version_info
    if _version_info is None:
        from . import _version

        _version_info = _version.get_versions()["version"]
    return _version_info


# For TYPE_CHECKING, we provide type hints so IDEs can see the exports
if TYPE_CHECKING:
    from .auth import get_fireworks_account_id, get_fireworks_api_key
    from .common_utils import load_jsonl
    from .config import RewardKitConfig, get_config, load_config
    from .mcp_env import (
        AnthropicPolicy,
        FireworksPolicy,
        LiteLLMPolicy,
        OpenAIPolicy,
        make,
        rollout,
        test_mcp,
    )
    from .data_loader import DynamicDataLoader, InlineDataLoader
    from . import mcp, rewards
    from .models import EvaluateResult, Message, MetricResult, EvaluationRow, InputMetadata, Status
    from .playback_policy import PlaybackPolicyBase
    from .resources import create_llm_resource
    from .reward_function import RewardFunction
    from .typed_interface import reward_function
    from .quickstart.aha_judge import aha_judge
    from .utils.evaluation_row_utils import (
        multi_turn_assistant_to_ground_truth,
        assistant_to_ground_truth,
        filter_longest_conversation,
    )
    from .pytest import (
        evaluation_test,
        SingleTurnRolloutProcessor,
        RemoteRolloutProcessor,
        GithubActionRolloutProcessor,
    )
    from .pytest.parameterize import DefaultParameterIdGenerator
    from .log_utils.elasticsearch_direct_http_handler import ElasticsearchDirectHttpHandler
    from .log_utils.rollout_id_filter import RolloutIdFilter
    from .log_utils.util import setup_rollout_logging_for_elasticsearch_handler
    from .log_utils.fireworks_tracing_http_handler import FireworksTracingHttpHandler
    from .log_utils.elasticsearch_client import ElasticsearchConfig
    from .types.remote_rollout_processor import (
        InitRequest,
        RolloutMetadata,
        StatusResponse,
        create_langfuse_config_tags,
        DataLoaderConfig,
    )
    from .adapters import (
        OpenAIResponsesAdapter,
        LangfuseAdapter,
        create_langfuse_adapter,
        BraintrustAdapter,
        create_braintrust_adapter,
        LangSmithAdapter,
        WeaveAdapter,
    )
    from .proxy import create_app, AuthProvider, AccountInfo


# __version__ property - computed lazily
class _VersionModule:
    @property
    def __version__(self):
        return _get_version()


import sys

_this_module = sys.modules[__name__]
_this_module.__class__ = type("module", (type(_this_module),), {"__version__": property(lambda self: _get_version())})
