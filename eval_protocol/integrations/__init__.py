"""Integration helpers for Eval Protocol."""

from .openeval import adapt
from .trl import create_trl_adapter

__all__ = [
    "adapt",
    "create_trl_adapter",
]
