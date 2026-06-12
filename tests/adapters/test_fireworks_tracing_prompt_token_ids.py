"""Tests for prompt token ID payload handling in fireworks_tracing adapter."""

from __future__ import annotations

import base64
import struct

import pytest
import zstandard as zstd

pytest.importorskip("mcp")

from eval_protocol.adapters.fireworks_tracing import convert_trace_dict_to_evaluation_row
from eval_protocol.adapters.pti_deserializer import (
    ENTRY_FORMAT,
    ENTRY_SIZE,
    HEADER_FORMAT,
    MAGIC,
    decompress_and_parse_pti,
)


def _pti_b64(token_ids: list[int]) -> str:
    token_count = len(token_ids)
    body_byte_length = token_count * ENTRY_SIZE
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        1,
        0,
        0,
        token_count,
        body_byte_length,
        0,
    )
    body = b"".join(struct.pack(ENTRY_FORMAT, token_id) for token_id in token_ids)
    compressed = zstd.ZstdCompressor().compress(header + body)
    return base64.b64encode(compressed).decode("ascii")


def test_decompress_and_parse_pti_round_trip():
    token_ids, metadata = decompress_and_parse_pti(_pti_b64([101, 102, 103]))

    assert token_ids == [101, 102, 103]
    assert metadata["scope"] == "prompt_only"
    assert metadata["token_count"] == 3


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
