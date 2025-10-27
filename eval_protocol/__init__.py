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
from .pytest import evaluation_test, SingleTurnRolloutProcessor, RemoteRolloutProcessor, GithubActionRolloutProcessor
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

try:
    from .adapters import OpenAIResponsesAdapter
except ImportError:
    OpenAIResponsesAdapter = None

try:
    from .adapters import LangfuseAdapter, create_langfuse_adapter
except ImportError:
    LangfuseAdapter = None

try:
    from .adapters import BraintrustAdapter, create_braintrust_adapter
except ImportError:
    BraintrustAdapter = None

try:
    from .adapters import LangSmithAdapter
except ImportError:
    LangSmithAdapter = None


try:
    from .adapters import WeaveAdapter
except ImportError:
    WeaveAdapter = None

try:
    from .proxy import create_app, AuthProvider, AccountInfo  # pyright: ignore[reportAssignmentType]
except ImportError:

    def create_app(*args, **kwargs):
        raise ImportError(
            "Proxy functionality requires additional dependencies. "
            "Please install with: pip install eval-protocol[proxy]"
        )

    class AuthProvider:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Proxy functionality requires additional dependencies. "
                "Please install with: pip install eval-protocol[proxy]"
            )

    class AccountInfo:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Proxy functionality requires additional dependencies. "
                "Please install with: pip install eval-protocol[proxy]"
            )


warnings.filterwarnings("default", category=DeprecationWarning, module="eval_protocol")

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

from . import _version

__version__ = _version.get_versions()["version"]
