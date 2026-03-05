"""Integration helpers for Eval Protocol."""

from .openeval import adapt
from .trl import create_trl_adapter
from .openai_rft import build_python_grader_from_evaluation_test
from .fireworks_v1_completions_client import (
    FireworksV1CompletionsClient,
    ParsedToolCall,
    to_openai_tool_calls,
)

__all__ = [
    "adapt",
    "create_trl_adapter",
    "build_python_grader_from_evaluation_test",
    "FireworksV1CompletionsClient",
    "ParsedToolCall",
    "to_openai_tool_calls",
]
