"""
Fireworks Eval Protocol - Simplify reward modeling and evaluation for LLM RL fine-tuning.

This package is the canonical import name for the evaluation toolkit. All functionality from `reward_kit` is re-exported here so existing code can gradually migrate.

A Python library for defining, testing, deploying, and using reward functions
for LLM fine-tuning, including launching full RL jobs on the Fireworks platform.

The library also provides an agent evaluation framework for testing and evaluating
tool-augmented models using self-contained task bundles.
"""

# Map eval_protocol submodules to the underlying reward_kit modules
import sys

# Additional convenience imports for common submodules
# Make sure all public symbols are available
# Re-export everything from reward_kit
from reward_kit import *  # noqa: F401,F403
from reward_kit import (
    __all__,
    __version__,
    adapters,
    agent,
    auth,
    cli,
    cli_commands,
    common_utils,
    config,
    datasets,
    evaluation,
    execution,
    gcp_tools,
    generation,
    generic_server,
    integrations,
    mcp,
    mcp_agent,
    models,
    packaging,
    platform_api,
    playback_policy,
    resources,
    reward_function,
    rewards,
    rl_processing,
    server,
    typed_interface,
    utils,
)

_SUBMODULES = [
    "adapters",
    "agent",
    "auth",
    "cli",
    "cli_commands",
    "common_utils",
    "config",
    "datasets",
    "evaluation",
    "execution",
    "gcp_tools",
    "generation",
    "generic_server",
    "integrations",
    "mcp",
    "mcp_agent",
    "models",
    "packaging",
    "platform_api",
    "playback_policy",
    "resources",
    "reward_function",
    "rewards",
    "rl_processing",
    "server",
    "typed_interface",
    "utils",
]

for _name in _SUBMODULES:
    sys.modules[f"{__name__}.{_name}"] = getattr(sys.modules["reward_kit"], _name)
