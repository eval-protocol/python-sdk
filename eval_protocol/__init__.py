"""
Fireworks Eval Protocol - Simplify reward modeling and evaluation for LLM RL fine-tuning.

A Python library for defining, testing, deploying, and using reward functions
for LLM fine-tuning, including launching full RL jobs on the Fireworks platform.

The library also provides an agent evaluation framework for testing and evaluating
tool-augmented models using self-contained task bundles.
"""

import warnings

from eval_protocol.adapters.braintrust import reward_fn_to_scorer, scorer_to_reward_fn

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

# Try to import FireworksPolicy if available
try:
    from .mcp_env import FireworksPolicy

    _FIREWORKS_AVAILABLE = True
except (ImportError, AttributeError):
    _FIREWORKS_AVAILABLE = False
# Import submodules to make them available via eval_protocol.rewards, etc.
from . import mcp, rewards
from .models import EvaluateResult, Message, MetricResult
from .playback_policy import PlaybackPolicyBase
from .resources import create_llm_resource
from .reward_function import RewardFunction
from .typed_interface import reward_function

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
]

from . import _version

__version__ = _version.get_versions()["version"]
