"""
Fireworks Eval Protocol - Simplify reward modeling and evaluation for LLM RL fine-tuning.

A Python library for defining, testing, deploying, and using reward functions
for LLM fine-tuning, including launching full RL jobs on the Fireworks platform.

The library also provides an agent evaluation framework for testing and evaluating
tool-augmented models using self-contained task bundles.
"""

import warnings

from .adapters.braintrust import reward_fn_to_scorer, scorer_to_reward_fn
from .auth import get_fireworks_api_key, get_fireworks_account_id
from .common_utils import load_jsonl
from .config import load_config, get_config, RewardKitConfig
from .mcp_env import (
    OpenAIPolicy,
    AnthropicPolicy,
    MCPVectorEnv,
    make,
    rollout,
    test_mcp,
)

# Try to import FireworksPolicy if available
try:
    from .mcp_env import FireworksPolicy

    _FIREWORKS_AVAILABLE = True
except (ImportError, AttributeError):
    _FIREWORKS_AVAILABLE = False
from .models import EvaluateResult, Message, MetricResult
from .playback_policy import PlaybackPolicyBase
from .resources import create_llm_resource
from .reward_function import RewardFunction
from .typed_interface import reward_function

# Import submodules to make them available via eval_protocol.rewards, etc.
from . import rewards
from . import mcp

warnings.filterwarnings("default", category=DeprecationWarning, module="eval_protocol")

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
    # MCP Environment API
    "make",
    "rollout",
    "AnthropicPolicy",
    "FireworksPolicy",
    "OpenAIPolicy",
    "MCPVectorEnv",
    "test_mcp",
    # Playback functionality
    "PlaybackPolicyBase",
    # Resource management
    "create_llm_resource",
    # Submodules
    "rewards",
    "mcp",
]

# Add FireworksPolicy to exports if available
if _FIREWORKS_AVAILABLE:
    __all__.insert(__all__.index("OpenAIPolicy") + 1, "FireworksPolicy")

from . import _version

__version__ = _version.get_versions()["version"]
