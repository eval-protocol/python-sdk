"""Deprecated adapter wrappers for TRL.

This module forwards imports to :mod:`reward_kit.integrations.trl`.
"""

from ..integrations.trl import create_trl_adapter

__all__ = ["create_trl_adapter"]
