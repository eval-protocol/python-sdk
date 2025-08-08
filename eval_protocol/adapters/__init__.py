"""Data source adapters for Eval Protocol.

This package provides adapters for integrating with various data sources
and converting them to EvaluationRow format for use in evaluation pipelines.

Available adapters:
- LangfuseAdapter: Pull data from Langfuse deployments
- HuggingFaceAdapter: Load datasets from HuggingFace Hub
- Braintrust integration (legacy)
- TRL integration (legacy)
"""

from importlib import import_module
from typing import Any

__all__ = []


def __getattr__(name: str) -> Any:
    # Lazy import optional adapters to avoid import-time side effects and heavy deps
    if name in {"LangfuseAdapter", "create_langfuse_adapter"}:
        m = import_module(".langfuse", __name__)
        return getattr(m, name)
    if name in {"HuggingFaceAdapter", "create_huggingface_adapter", "create_gsm8k_adapter", "create_math_adapter"}:
        m = import_module(".huggingface", __name__)
        return getattr(m, name)
    if name in {"reward_fn_to_scorer", "scorer_to_reward_fn"}:
        m = import_module(".braintrust", __name__)
        return getattr(m, name)
    if name in {"create_trl_adapter"}:
        m = import_module(".trl", __name__)
        return getattr(m, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
