"""Deprecated adapter wrappers for Braintrust.

This module forwards imports to :mod:`reward_kit.integrations.braintrust`.
"""

from ..integrations.braintrust import reward_fn_to_scorer, scorer_to_reward_fn

__all__ = ["scorer_to_reward_fn", "reward_fn_to_scorer"]
