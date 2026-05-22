"""Tests for LP/v1 binary deserializer (gateway-compatible)."""

from __future__ import annotations

import base64
import struct

import pytest
import zstandard as zstd

from eval_protocol.adapters.lp_deserializer import (
    ENTRY_FORMAT,
    ENTRY_SIZE,
    HEADER_FORMAT,
    HEADER_SIZE,
    MAGIC,
    MISSING_TOKEN_ID,
    decompress_and_parse_lp,
    parse_logprobs,
)

# Golden raw bytes: two tokens (7, -0.25) and (8, -0.5) — must match gateway serializer.
GOLDEN_RAW_HEX = (
    "4c503031010000000200000010000000000000000000000007000000000080be"
    "08000000000000bf"
)


def _build_raw(tokens: list[tuple[int, float]]) -> bytes:
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
    return header + body


def _compress_b64(raw: bytes) -> str:
    return base64.b64encode(zstd.ZstdCompressor().compress(raw)).decode("ascii")


class TestParseLogprobs:
    def test_golden_bytes_match_gateway(self):
        raw = bytes.fromhex(GOLDEN_RAW_HEX)
        logprobs, token_ids, meta = parse_logprobs(raw)
        assert logprobs == [-0.25, -0.5]
        assert token_ids == [7, 8]
        assert meta["all_token_ids_valid"] is True
        assert meta["token_count"] == 2

    def test_missing_token_id_omits_token_ids_list(self):
        raw = _build_raw([(MISSING_TOKEN_ID, -0.3), (42, -0.4)])
        logprobs, token_ids, meta = parse_logprobs(raw)
        assert logprobs == pytest.approx([-0.3, -0.4])
        assert token_ids is None
        assert meta["all_token_ids_valid"] is False

    def test_decompress_and_parse_round_trip(self):
        raw = bytes.fromhex(GOLDEN_RAW_HEX)
        b64 = _compress_b64(raw)
        logprobs, token_ids, meta = decompress_and_parse_lp(b64)
        assert logprobs == [-0.25, -0.5]
        assert token_ids == [7, 8]
        assert meta["scope"] == "completion_only"

    def test_rejects_bad_magic(self):
        raw = _build_raw([(1, -0.1)])
        bad = b"XXXX" + raw[4:]
        with pytest.raises(ValueError, match="Bad LP/v1 magic"):
            parse_logprobs(bad)
