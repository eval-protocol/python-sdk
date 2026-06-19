"""Backward-compat shims for moved adapter deserializers."""

from __future__ import annotations

import warnings

import pytest

from tests.adapters.test_lp_deserializer import GOLDEN_RAW_HEX


def test_lp_deserializer_shim_reexports_and_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        from eval_protocol.adapters import lp_deserializer as shim

    assert any(
        "eval_protocol.adapters.lp_deserializer is deprecated" in str(w.message)
        for w in caught
    )
    raw = bytes.fromhex(GOLDEN_RAW_HEX)
    logprobs, token_ids, metadata = shim.parse_logprobs(raw)
    assert logprobs == [-0.25, -0.5]
    assert token_ids == [7, 8]
    assert metadata["completion_token_count"] == 2


def test_r3_deserializer_shim_reexports_and_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        from eval_protocol.adapters import r3_deserializer as shim

    assert any(
        "eval_protocol.adapters.r3_deserializer is deprecated" in str(w.message)
        for w in caught
    )
    assert shim.MAGIC == b"R3V1"
    assert shim._SelectorMode.ALL == 0
