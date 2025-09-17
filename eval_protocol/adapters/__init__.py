"""Data source adapters for Eval Protocol.

This package provides adapters for integrating with various data sources
and converting them to EvaluationRow format for use in evaluation pipelines.

Available adapters:
- BaseAdapter: Abstract base class for all adapters
- LangfuseAdapter: Pull data from Langfuse deployments
- HuggingFaceAdapter: Load datasets from HuggingFace Hub
- BigQueryAdapter: Query data from Google BigQuery
- TRL integration (legacy)
"""

# Always available
from .base import BaseAdapter

__all__ = ["BaseAdapter"]

# Conditional imports based on available dependencies
try:
    from .langfuse import LangfuseAdapter, create_langfuse_adapter

    __all__.extend(["LangfuseAdapter", "create_langfuse_adapter"])
except ImportError:
    pass

try:
    from .huggingface import (
        HuggingFaceAdapter,
        create_gsm8k_adapter,
        create_huggingface_adapter,
        create_math_adapter,
    )

    __all__.extend(
        [
            "HuggingFaceAdapter",
            "create_huggingface_adapter",
            "create_gsm8k_adapter",
            "create_math_adapter",
        ]
    )
except ImportError:
    pass

try:
    from .bigquery import (
        BigQueryAdapter,
        create_bigquery_adapter,
    )

    __all__.extend(
        [
            "BigQueryAdapter",
            "create_bigquery_adapter",
        ]
    )
except ImportError:
    pass

try:
    from .braintrust import BraintrustAdapter, create_braintrust_adapter

    __all__.extend(["BraintrustAdapter", "create_braintrust_adapter"])
except ImportError:
    pass

# Legacy adapters (always available)

try:
    from .trl import create_trl_adapter

    __all__.extend(["create_trl_adapter"])
except ImportError:
    pass

try:
    from .openai_responses import OpenAIResponsesAdapter

    __all__.extend(["OpenAIResponsesAdapter"])
except ImportError:
    pass

try:
    from .langsmith import LangSmithAdapter

    __all__.extend(["LangSmithAdapter"])
except ImportError:
    pass
