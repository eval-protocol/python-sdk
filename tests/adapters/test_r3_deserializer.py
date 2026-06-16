"""Tests for R3/v1 binary deserializer."""

from __future__ import annotations

import base64
import math
import struct
from typing import List, Optional

import pytest
import zstandard as zstd

from eval_protocol.tracing.router_replay import (
    HEADER_FORMAT,
    HEADER_SIZE,
    MAGIC,
    _SelectorMode,
    _RoutingDtype,
    _parse_header,
    _read_bitmap_positions,
    decompress_and_parse_r3,
)


def _make_raw_r3(
    *,
    selector_mode: int = _SelectorMode.ALL,
    routing_dtype: int = _RoutingDtype.UINT8,
    total_token_count: int = 4,
    replayed_token_count: int = 4,
    matrix_elem_size: Optional[int] = None,
    replay_start_token: int = 0,
    selector_bytes: bytes = b"",
    matrix_data: Optional[bytes] = None,
) -> bytes:
    """Build a raw (uncompressed) R3/v1 payload for testing.

    ``matrix_elem_size`` is the per-token matrix byte length; when not given
    and no explicit ``matrix_data`` is supplied, defaults to 4 bytes/token
    (a minimal placeholder for tests that don't care about shape).
    """
    if matrix_data is None:
        if matrix_elem_size is None:
            matrix_elem_size = 4
        matrix_data = bytes(range(matrix_elem_size)) * replayed_token_count

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        1,  # version
        selector_mode,
        routing_dtype,
        0x01,  # flags: little-endian
        total_token_count,
        replayed_token_count,
        replay_start_token,
        len(selector_bytes),
        len(matrix_data),
    )
    return header + selector_bytes + matrix_data


def _compress_and_b64(raw: bytes) -> str:
    compressor = zstd.ZstdCompressor()
    compressed = compressor.compress(raw)
    return base64.b64encode(compressed).decode("ascii")


class TestParseHeader:
    def test_valid_header(self):
        raw = _make_raw_r3(total_token_count=10, replayed_token_count=5)
        hdr = _parse_header(raw)
        assert hdr["total_token_count"] == 10
        assert hdr["replayed_token_count"] == 5
        assert hdr["selector_mode"] == _SelectorMode.ALL
        assert hdr["routing_dtype"] == _RoutingDtype.UINT8

    def test_bad_magic(self):
        raw = b"XXXX" + b"\x00" * (HEADER_SIZE - 4)
        with pytest.raises(ValueError, match="Bad R3 magic"):
            _parse_header(raw)

    def test_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            _parse_header(b"\x00" * 10)

    def test_unsupported_version(self):
        raw = struct.pack(
            HEADER_FORMAT,
            MAGIC, 99, 0, 1, 0, 4, 4, 0, 0, 16,
        )
        with pytest.raises(ValueError, match="Unsupported R3 header version"):
            _parse_header(raw)


class TestReadBitmapPositions:
    def test_all_set(self):
        bitmap = bytes([0xFF])
        positions = _read_bitmap_positions(bitmap, 8)
        assert positions == list(range(8))

    def test_none_set(self):
        bitmap = bytes([0x00])
        positions = _read_bitmap_positions(bitmap, 8)
        assert positions == []

    def test_sparse(self):
        # Bit 0 and bit 2 set => positions [0, 2]
        bitmap = bytes([0b00000101])
        positions = _read_bitmap_positions(bitmap, 8)
        assert positions == [0, 2]

    def test_multi_byte(self):
        # 16 tokens: first byte has bits 0,7 set; second byte has bit 1 (token 9) set
        bitmap = bytes([0b10000001, 0b00000010])
        positions = _read_bitmap_positions(bitmap, 16)
        assert positions == [0, 7, 9]


class TestDecompressAndParseR3:
    def test_all_mode_uint8(self):
        matrix_elem_size = 4  # e.g. 2 MoE layers * 2 top-k * 1 byte (uint8)
        total_tokens = 4

        matrices_raw = []
        for i in range(total_tokens):
            matrices_raw.append(bytes([i * 10 + j for j in range(matrix_elem_size)]))
        matrix_data = b"".join(matrices_raw)

        raw = _make_raw_r3(
            total_token_count=total_tokens,
            replayed_token_count=total_tokens,
            matrix_data=matrix_data,
        )
        blob = _compress_and_b64(raw)

        matrices, metadata = decompress_and_parse_r3(blob)

        assert len(matrices) == total_tokens
        assert metadata["routing_dtype"] == "uint8"
        assert metadata["selector_mode"] == "all"
        assert metadata["total_token_count"] == total_tokens
        assert metadata["replayed_token_count"] == total_tokens

        for i in range(total_tokens):
            assert matrices[i] is not None
            decoded = base64.b64decode(matrices[i])
            assert decoded == matrices_raw[i]

    def test_suffix_mode(self):
        matrix_elem_size = 4
        total_tokens = 8
        replayed = 3
        start_token = 5

        matrices_raw = []
        for i in range(replayed):
            matrices_raw.append(bytes([(start_token + i) * 10 + j for j in range(matrix_elem_size)]))
        matrix_data = b"".join(matrices_raw)

        raw = _make_raw_r3(
            selector_mode=_SelectorMode.SUFFIX,
            total_token_count=total_tokens,
            replayed_token_count=replayed,
            replay_start_token=start_token,
            matrix_data=matrix_data,
        )
        blob = _compress_and_b64(raw)

        matrices, metadata = decompress_and_parse_r3(blob)

        assert len(matrices) == total_tokens
        assert metadata["selector_mode"] == "suffix"
        assert metadata["replay_start_token"] == start_token

        # Positions before start_token should be None
        for i in range(start_token):
            assert matrices[i] is None

        # Positions from start_token to start_token+replayed should have data
        for i in range(replayed):
            pos = start_token + i
            assert matrices[pos] is not None
            decoded = base64.b64decode(matrices[pos])
            assert decoded == matrices_raw[i]

    def test_bitmap_mode(self):
        matrix_elem_size = 4
        total_tokens = 8

        # Replay tokens at positions 1, 3, 6
        replayed_positions = [1, 3, 6]
        replayed = len(replayed_positions)

        # Build bitmap
        bitmap = bytearray(math.ceil(total_tokens / 8))
        for pos in replayed_positions:
            bitmap[pos >> 3] |= 1 << (pos & 7)
        selector_bytes = bytes(bitmap)

        matrices_raw = []
        for idx, pos in enumerate(replayed_positions):
            matrices_raw.append(bytes([pos * 10 + j for j in range(matrix_elem_size)]))
        matrix_data = b"".join(matrices_raw)

        raw = _make_raw_r3(
            selector_mode=_SelectorMode.BITMAP,
            total_token_count=total_tokens,
            replayed_token_count=replayed,
            selector_bytes=selector_bytes,
            matrix_data=matrix_data,
        )
        blob = _compress_and_b64(raw)

        matrices, metadata = decompress_and_parse_r3(blob)

        assert len(matrices) == total_tokens
        assert metadata["selector_mode"] == "bitmap"
        assert metadata["replayed_token_count"] == replayed

        for i in range(total_tokens):
            if i in replayed_positions:
                assert matrices[i] is not None
                idx = replayed_positions.index(i)
                decoded = base64.b64decode(matrices[i])
                assert decoded == matrices_raw[idx]
            else:
                assert matrices[i] is None

    def test_uint16_dtype(self):
        matrix_elem_size = 8  # e.g. 2 MoE layers * 2 top-k * 2 bytes (uint16)
        total_tokens = 2

        matrices_raw = []
        for i in range(total_tokens):
            matrices_raw.append(bytes([i * 10 + j for j in range(matrix_elem_size)]))
        matrix_data = b"".join(matrices_raw)

        raw = _make_raw_r3(
            routing_dtype=_RoutingDtype.UINT16,
            total_token_count=total_tokens,
            replayed_token_count=total_tokens,
            matrix_data=matrix_data,
        )
        blob = _compress_and_b64(raw)

        matrices, metadata = decompress_and_parse_r3(blob)

        assert metadata["routing_dtype"] == "uint16"
        assert len(matrices) == total_tokens
        for i in range(total_tokens):
            decoded = base64.b64decode(matrices[i])
            assert decoded == matrices_raw[i]

    def test_zero_replayed_tokens(self):
        raw = _make_raw_r3(
            total_token_count=10,
            replayed_token_count=0,
            matrix_data=b"",
        )
        blob = _compress_and_b64(raw)

        matrices, metadata = decompress_and_parse_r3(blob)

        assert len(matrices) == 10
        assert all(m is None for m in matrices)
        assert metadata["replayed_token_count"] == 0

    def test_unknown_routing_dtype_falls_back_to_str(self):
        """Unknown routing_dtype ints (e.g. a future dtype=3) must not crash
        metadata construction; the dtype is surfaced as its string repr."""
        raw = _make_raw_r3(
            routing_dtype=99,  # not in _RoutingDtype
            total_token_count=2,
            replayed_token_count=2,
            matrix_data=b"\x00" * 8,
        )
        blob = _compress_and_b64(raw)

        _, metadata = decompress_and_parse_r3(blob)
        assert metadata["routing_dtype"] == "99"

    def test_high_compression_ratio_payload(self):
        """Highly compressible payloads (e.g. tokens routing to the same
        experts) can compress much better than 20:1; the deserializer must
        not impose an arbitrary cap on the decompressed size."""
        # 64 KiB of zeros compresses to ~35 bytes (>1000x ratio).
        total_tokens = 1024
        matrix_elem_size = 64  # bytes/token
        matrix_data = b"\x00" * (total_tokens * matrix_elem_size)

        raw = _make_raw_r3(
            total_token_count=total_tokens,
            replayed_token_count=total_tokens,
            matrix_data=matrix_data,
        )
        blob = _compress_and_b64(raw)
        # Sanity: compression really is >> 20x for this case.
        assert len(base64.b64decode(blob)) * 20 < len(raw)

        matrices, metadata = decompress_and_parse_r3(blob)
        assert len(matrices) == total_tokens
        assert metadata["replayed_token_count"] == total_tokens
        for m in matrices:
            assert base64.b64decode(m) == b"\x00" * matrix_elem_size


class TestRoundTrip:
    """Round-trip test using the gateway's serializer and EP's deserializer."""

    def test_round_trip_with_serializer(self):
        """Verify that data serialized by the gateway's r3_serializer can be
        deserialized by EP's r3_deserializer and produce the original per-token
        matrices."""
        import sys
        import os

        # Add the tracing gateway code to the path so we can import the serializer
        serializer_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "mono", "eval-py"
        )
        serializer_dir = os.path.normpath(serializer_dir)

        if not os.path.isdir(serializer_dir):
            pytest.skip(f"Serializer source not available at {serializer_dir}")

        sys.path.insert(0, serializer_dir)
        try:
            from litellm_proxy_config.proxy_core.r3_serializer import (
                serialize_r3,
                compress_and_chunk,
            )
            from litellm_proxy_config.proxy_core.models import RouterReplayData
        except ImportError:
            pytest.skip("r3_serializer or models not importable")
        finally:
            sys.path.pop(0)

        num_moe_layers = 4
        top_k = 8
        total_tokens = 16

        # Build per-token matrices as Optional[bytes], like the gateway produces
        original_matrices: List[Optional[bytes]] = []
        original_b64: List[Optional[str]] = []
        matrix_elem_size = num_moe_layers * top_k  # uint8
        for i in range(total_tokens):
            if i < 4:
                # Prompt tokens: no routing data
                original_matrices.append(None)
                original_b64.append(None)
            else:
                mat = bytes([(i * 7 + j) % 256 for j in range(matrix_elem_size)])
                original_matrices.append(mat)
                original_b64.append(base64.b64encode(mat).decode("ascii"))

        data = RouterReplayData(
            routing_matrices=original_matrices,
            total_token_count=total_tokens,
            routing_dtype="uint8",
        )

        raw_payload = serialize_r3(data)
        chunks = compress_and_chunk(raw_payload, chunk_size=1024 * 1024)
        assembled = b"".join(chunks)
        blob_b64 = base64.b64encode(assembled).decode("ascii")

        # Now deserialize with EP
        matrices, metadata = decompress_and_parse_r3(blob_b64)

        assert len(matrices) == total_tokens
        assert metadata["total_token_count"] == total_tokens

        for i in range(total_tokens):
            if original_b64[i] is None:
                assert matrices[i] is None, f"Token {i} should be None"
            else:
                assert matrices[i] is not None, f"Token {i} should have data"
                assert matrices[i] == original_b64[i], f"Token {i} data mismatch"


class TestConvertTraceDictWithPayloads:
    """Test that convert_trace_dict_to_evaluation_row extracts R3 payloads."""

    def test_trace_with_router_replay_payload(self):
        from eval_protocol.adapters.fireworks_tracing import convert_trace_dict_to_evaluation_row

        matrix_elem_size = 4
        total_tokens = 4

        matrices_raw = []
        for i in range(total_tokens):
            matrices_raw.append(bytes([i * 10 + j for j in range(matrix_elem_size)]))
        matrix_data = b"".join(matrices_raw)

        raw = _make_raw_r3(
            total_token_count=total_tokens,
            replayed_token_count=total_tokens,
            matrix_data=matrix_data,
        )
        blob = _compress_and_b64(raw)

        trace = {
            "id": "test-trace-123",
            "input": {
                "messages": [
                    {"role": "user", "content": "hello"},
                ]
            },
            "output": {
                "choices": [
                    {"message": {"role": "assistant", "content": "hi"}}
                ]
            },
            "tags": ["rollout_id:r1", "run_id:run1"],
            "payloads": {
                "router_replay": {
                    "manifest": {
                        "PayloadVersion": "r3/v1",
                        "Compression": "zstd",
                    },
                    "data": blob,
                }
            },
        }

        row = convert_trace_dict_to_evaluation_row(trace)
        assert row is not None
        assert row.execution_metadata.extra is not None
        assert "routing_matrices" in row.execution_metadata.extra
        assert "routing_metadata" in row.execution_metadata.extra

        rm = row.execution_metadata.extra["routing_matrices"]
        assert len(rm) == total_tokens
        for i in range(total_tokens):
            assert rm[i] is not None
            decoded = base64.b64decode(rm[i])
            assert decoded == matrices_raw[i]

        meta = row.execution_metadata.extra["routing_metadata"]
        assert meta["routing_dtype"] == "uint8"
        assert meta["total_token_count"] == total_tokens

    def test_trace_without_payloads(self):
        from eval_protocol.adapters.fireworks_tracing import convert_trace_dict_to_evaluation_row

        trace = {
            "id": "test-trace-no-payload",
            "input": {
                "messages": [
                    {"role": "user", "content": "hello"},
                ]
            },
            "output": {
                "choices": [
                    {"message": {"role": "assistant", "content": "hi"}}
                ]
            },
            "tags": [],
        }

        row = convert_trace_dict_to_evaluation_row(trace)
        assert row is not None
        assert row.execution_metadata.extra is None

    def test_trace_with_empty_payload_data(self):
        from eval_protocol.adapters.fireworks_tracing import convert_trace_dict_to_evaluation_row

        trace = {
            "id": "test-trace-empty-payload",
            "input": {
                "messages": [
                    {"role": "user", "content": "hello"},
                ]
            },
            "output": {
                "choices": [
                    {"message": {"role": "assistant", "content": "hi"}}
                ]
            },
            "tags": [],
            "payloads": {
                "router_replay": {
                    "manifest": {},
                    "data": "",
                }
            },
        }

        row = convert_trace_dict_to_evaluation_row(trace)
        assert row is not None
        # Empty data string should be skipped (no crash)
        assert row.execution_metadata.extra is None
