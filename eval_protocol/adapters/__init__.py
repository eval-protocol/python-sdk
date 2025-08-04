"""Data source adapters for Eval Protocol.

This package provides adapters for integrating with various data sources
and converting them to EvaluationRow format for use in evaluation pipelines.

Available adapters:
- LangfuseAdapter: Pull data from Langfuse deployments
- HuggingFaceAdapter: Load datasets from HuggingFace Hub
- Braintrust integration (legacy)
- TRL integration (legacy)
"""

# Conditional imports based on available dependencies
try:
    from .langfuse import LangfuseAdapter, create_langfuse_adapter
    __all__ = ["LangfuseAdapter", "create_langfuse_adapter"]
except ImportError:
    __all__ = []

try:
    from .huggingface import (
        HuggingFaceAdapter, 
        create_huggingface_adapter,
        create_gsm8k_adapter,
        create_math_adapter,
    )
    __all__.extend([
        "HuggingFaceAdapter", 
        "create_huggingface_adapter",
        "create_gsm8k_adapter",
        "create_math_adapter",
    ])
except ImportError:
    pass

# Legacy adapters (always available)
try:
    from .braintrust import reward_fn_to_scorer, scorer_to_reward_fn
    __all__.extend(["scorer_to_reward_fn", "reward_fn_to_scorer"])
except ImportError:
    pass

try:
    from .trl import create_trl_adapter
    __all__.extend(["create_trl_adapter"])
except ImportError:
    pass
