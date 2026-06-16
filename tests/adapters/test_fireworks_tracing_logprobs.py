"""Tests for logprobs payload handling in fireworks_tracing adapter."""

from __future__ import annotations

import base64
import struct

import pytest
import zstandard as zstd

pytest.importorskip("mcp")

from eval_protocol.adapters.fireworks_tracing import convert_trace_dict_to_evaluation_row
from eval_protocol.tracing.logprobs import (
    ENTRY_FORMAT,
    ENTRY_SIZE,
    HEADER_FORMAT,
    MAGIC,
    MISSING_TOKEN_ID,
)


def _lp_b64(tokens: list[tuple[int, float]]) -> str:
    token_count = len(tokens)
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
    body = b"".join(struct.pack(ENTRY_FORMAT, tid, lp) for tid, lp in tokens)
    raw = header + body
    compressed = zstd.ZstdCompressor().compress(raw)
    return base64.b64encode(compressed).decode("ascii")


def _base_trace(*, with_token_ids: bool = True) -> dict:
    tokens = [(10, -0.1), (11, -0.2)] if with_token_ids else [(MISSING_TOKEN_ID, -0.1), (12, -0.2)]
    return {
        "id": "trace-1",
        "input": {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        },
        "output": {"role": "assistant", "content": "hello"},
        "payloads": {
            "logprobs": {
                "data": _lp_b64(tokens),
                "manifest": {"PayloadVersion": "lp/v1"},
            },
        },
    }


class TestConvertTraceLogprobs:
    def test_attaches_completion_logprobs_and_message_logprobs(self):
        row = convert_trace_dict_to_evaluation_row(_base_trace())
        assert row is not None

        extra = row.execution_metadata.extra
        assert extra is not None
        assert extra["completion_logprobs"] == pytest.approx([-0.1, -0.2])
        assert extra["completion_token_ids"] == [10, 11]

        assistant = row.messages[-1]
        assert assistant.role == "assistant"
        content = assistant.logprobs["content"]
        assert len(content) == len(extra["completion_logprobs"])
        assert content[0]["token_id"] == 10
        assert content[1]["token_id"] == 11
        assert content[0]["logprob"] == pytest.approx(-0.1)
        assert content[1]["logprob"] == pytest.approx(-0.2)

    def test_omits_token_id_keys_when_any_missing(self):
        row = convert_trace_dict_to_evaluation_row(_base_trace(with_token_ids=False))
        assert row is not None

        extra = row.execution_metadata.extra
        assert "completion_logprobs" in extra
        assert "completion_token_ids" not in extra

        content = row.messages[-1].logprobs["content"]
        assert len(content) == 2
        assert all("token_id" not in entry for entry in content)
        assert content[0]["logprob"] == pytest.approx(-0.1)
        assert content[1]["logprob"] == pytest.approx(-0.2)
