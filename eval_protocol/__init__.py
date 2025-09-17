"""
Fireworks Eval Protocol - Simplify reward modeling and evaluation for LLM RL fine-tuning.

A Python library for defining, testing, deploying, and using reward functions
for LLM fine-tuning, including launching full RL jobs on the Fireworks platform.

The library also provides an agent evaluation framework for testing and evaluating
tool-augmented models using self-contained task bundles.
"""

import warnings

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
from .models import EvaluateResult, Message, MetricResult, EvaluationRow
from .playback_policy import PlaybackPolicyBase
from .resources import create_llm_resource
from .reward_function import RewardFunction
from .typed_interface import reward_function
from .quickstart import aha_judge, split_multi_turn_rows
from .pytest import evaluation_test, SingleTurnRolloutProcessor

try:
    from .adapters import OpenAIResponsesAdapter

    _OPENAI_RESPONSES_AVAILABLE = True
except ImportError:
    _OPENAI_RESPONSES_AVAILABLE = False

try:
    from .adapters import LangfuseAdapter

    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False

try:
    from .adapters import BraintrustAdapter

    _BRAINTRUST_AVAILABLE = True
except ImportError:
    _BRAINTRUST_AVAILABLE = False

try:
    from .adapters import LangSmithAdapter

    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False

warnings.filterwarnings("default", category=DeprecationWarning, module="eval_protocol")

__all__ = [
    "aha_judge",
    "split_multi_turn_rows",
    "evaluation_test",
    "SingleTurnRolloutProcessor",
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
]

if _OPENAI_RESPONSES_AVAILABLE:
    __all__.append("OpenAIResponsesAdapter")

if _LANGFUSE_AVAILABLE:
    __all__.append("LangfuseAdapter")

if _BRAINTRUST_AVAILABLE:
    __all__.append("BraintrustAdapter")

if _LANGSMITH_AVAILABLE:
    __all__.append("LangSmithAdapter")

from . import _version

__version__ = _version.get_versions()["version"]
