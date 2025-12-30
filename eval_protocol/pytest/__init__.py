from .default_agent_rollout_processor import AgentRolloutProcessor
from .default_dataset_adapter import default_dataset_adapter
from .default_mcp_gym_rollout_processor import MCPGymRolloutProcessor
from .default_no_op_rollout_processor import NoOpRolloutProcessor
from .default_single_turn_rollout_process import SingleTurnRolloutProcessor
from .remote_rollout_processor import RemoteRolloutProcessor
from .github_action_rollout_processor import GithubActionRolloutProcessor
from .evaluation_test import evaluation_test
from .exception_config import ExceptionHandlerConfig, BackoffConfig, get_default_exception_handler_config
from .rollout_processor import RolloutProcessor
from .rollout_result_post_processor import RolloutResultPostProcessor, NoOpRolloutResultPostProcessor
from .types import RolloutProcessorConfig

# Conditional import for optional Klavis dependency
try:
    from .default_klavis_sandbox_rollout_processor import KlavisSandboxRolloutProcessor

    KLAVIS_AVAILABLE = True
except ImportError:
    KLAVIS_AVAILABLE = False
    KlavisSandboxRolloutProcessor = None

# Conditional import for optional dependencies
try:
    from .default_pydantic_ai_rollout_processor import PydanticAgentRolloutProcessor

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    PydanticAgentRolloutProcessor = None

# Conditional import for optional LangChain dependency
try:
    from .default_langchain_rollout_processor import LangGraphRolloutProcessor

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LangGraphRolloutProcessor = None

__all__ = [
    "AgentRolloutProcessor",
    "MCPGymRolloutProcessor",
    "RolloutProcessor",
    "SingleTurnRolloutProcessor",
    "RemoteRolloutProcessor",
    "GithubActionRolloutProcessor",
    "NoOpRolloutProcessor",
    "default_dataset_adapter",
    "RolloutProcessorConfig",
    "evaluation_test",
    "ExceptionHandlerConfig",
    "BackoffConfig",
    "get_default_exception_handler_config",
    "RolloutResultPostProcessor",
    "NoOpRolloutResultPostProcessor",
]

# Only add to __all__ if available
if KLAVIS_AVAILABLE:
    __all__.append("KlavisSandboxRolloutProcessor")

# Only add to __all__ if available
if PYDANTIC_AI_AVAILABLE:
    __all__.append("PydanticAgentRolloutProcessor")

if LANGCHAIN_AVAILABLE:
    __all__.append("LangGraphRolloutProcessor")
