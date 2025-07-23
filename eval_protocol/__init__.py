"""
Fireworks Eval Protocol - Simplify reward modeling and evaluation for LLM RL fine-tuning.

A Python library for defining, testing, deploying, and using reward functions
for LLM fine-tuning, including launching full RL jobs on the Fireworks platform.

The library also provides an agent evaluation framework for testing and evaluating
tool-augmented models using self-contained task bundles.
"""

import warnings

from .adapters.braintrust import reward_fn_to_scorer, scorer_to_reward_fn
from .common_utils import load_jsonl
from .mcp_env import FireworksPolicy, OpenAIPolicy, MCPVectorEnv, make, rollout, test_mcp
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
    # Utilities
    "load_jsonl",
    # MCP Environment API
    "make",
    "rollout",
    "FireworksPolicy",
    "OpenAIPolicy",
    "MCPVectorEnv",
    "test_mcp",
    # Playback functionality
    "PlaybackPolicyBase",
    # Resource management
    "create_llm_resource",
]

from . import _version

__version__ = _version.get_versions()["version"]