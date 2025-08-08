"""
Fireworks Eval Protocol - Simplify reward modeling and evaluation for LLM RL fine-tuning.

A Python library for defining, testing, deploying, and using reward functions
for LLM fine-tuning, including launching full RL jobs on the Fireworks platform.

The library also provides an agent evaluation framework for testing and evaluating
tool-augmented models using self-contained task bundles.
"""

from importlib import import_module
from typing import Any
import warnings

# Lightweight imports (no heavy optional dependencies)
from .integrations.braintrust import reward_fn_to_scorer, scorer_to_reward_fn
from .auth import get_fireworks_account_id, get_fireworks_api_key
from .common_utils import load_jsonl
from .config import RewardKitConfig, get_config, load_config

# Import submodules to make them available via eval_protocol.rewards, etc.
from .models import EvaluateResult, Message, MetricResult
from .playback_policy import PlaybackPolicyBase
from .resources import create_llm_resource
from .reward_function import RewardFunction
from .typed_interface import reward_function

warnings.filterwarnings("default", category=DeprecationWarning, module="eval_protocol")

# Public API (static exports only; dynamic MCP symbols are provided via __getattr__)
__all__ = [
    # Core interfaces
    "Message",
    "MetricResult",
    "EvaluateResult",
    "reward_function",
    "RewardFunction",
    "scorer_to_reward_fn",
    "reward_fn_to_scorer",
    # Authentication
    "get_fireworks_api_key",
    "get_fireworks_account_id",
    # Configuration
    "load_config",
    "get_config",
    "RewardKitConfig",
    # Utilities
    "load_jsonl",
    # Playback functionality
    "PlaybackPolicyBase",
    # Resource management
    "create_llm_resource",
    # Submodules
    "rewards",
    "mcp",
]


def __getattr__(name: str) -> Any:
    """Lazily import heavy MCP environment symbols to speed up package import.

    This defers importing modules that depend on optional or heavy dependencies
    (e.g., vendored tau2, OpenAI clients) until they are actually used.
    """
    if name in {
        "make",
        "rollout",
        "LiteLLMPolicy",
        "AnthropicPolicy",
        "FireworksPolicy",
        "OpenAIPolicy",
        "test_mcp",
    }:
        m = import_module(".mcp_env", __name__)
        return getattr(m, name)
    if name in {"mcp", "rewards"}:
        # Lazy-load subpackages for attribute access like eval_protocol.mcp
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


from . import _version

__version__ = _version.get_versions()["version"]
