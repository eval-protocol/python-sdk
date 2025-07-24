"""
Fireworks Reward Kit - Backward compatibility layer.

This module provides backward compatibility for code that imports from reward_kit.
All functionality has been moved to eval_protocol.

This is a compatibility layer that imports and re-exports everything from eval_protocol.
For new code, please import directly from eval_protocol instead.
"""

import warnings
import importlib
import pkgutil
import sys

# Issue a deprecation warning when reward_kit is imported
warnings.warn(
    "reward_kit is deprecated and will be removed in a future version. " "Please import from eval_protocol instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import and re-export everything from eval_protocol
import eval_protocol

# Re-export all public symbols from eval_protocol
from eval_protocol import *  # noqa: F401,F403
from eval_protocol import (
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

# Dynamically create all submodule aliases to point to eval_protocol modules
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
    try:
        module = importlib.import_module(f"eval_protocol.{_name}")
        sys.modules[f"{__name__}.{_name}"] = module
    except ImportError:
        # If a module fails to import, skip it
        pass

# Mirror all nested submodules from eval_protocol so that imports like
# "from reward_kit.rewards.math import something" continue to work
for finder, mod_name, _ in pkgutil.walk_packages(eval_protocol.__path__, eval_protocol.__name__ + "."):
    try:
        module = importlib.import_module(mod_name)
        alias_name = f"{__name__}{mod_name[len('eval_protocol'):]}"
        sys.modules.setdefault(alias_name, module)
    except Exception:
        # If a module fails to import, skip it. These are optional extras.
        pass
