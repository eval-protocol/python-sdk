"""Tests for prompt token ID payload handling in fireworks_tracing adapter."""

from __future__ import annotations

import base64
import json

import pytest
import zstandard as zstd

pytest.importorskip("mcp")

from eval_protocol.adapters.fireworks_tracing import convert_trace_dict_to_evaluation_row
from eval_protocol.tracing.prompt_token_ids import decode_prompt_token_ids


def _pti_b64(token_ids: list[int]) -> str:
    """Build a gateway pti/v1 payload: base64(zstd(json int array))."""
    raw = json.dumps(token_ids).encode("utf-8")
    return base64.b64encode(zstd.ZstdCompressor().compress(raw)).decode("ascii")


def test_decode_prompt_token_ids_round_trip():
    decoded = decode_prompt_token_ids(_pti_b64([101, 102, 103]))

    assert decoded.value == [101, 102, 103]
    assert decoded.metadata["scope"] == "prompt_only"
    assert decoded.metadata["token_count"] == 3


def test_trace_adapter_attaches_prompt_token_ids_metadata():
    trace = {
        "id": "trace-pti",
        "input": {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        },
        "output": {"role": "assistant", "content": "hello"},
        "payloads": {
            "prompt_token_ids": {
                "data": _pti_b64([201, 202, 203]),
                "manifest": {"PayloadVersion": "pti/v1"},
            },
        },
    }

    row = convert_trace_dict_to_evaluation_row(trace)

    assert row is not None
    extra = row.execution_metadata.extra
    assert extra is not None
    assert extra["prompt_token_ids"] == [201, 202, 203]
    assert extra["prompt_token_ids_metadata"]["token_count"] == 3
