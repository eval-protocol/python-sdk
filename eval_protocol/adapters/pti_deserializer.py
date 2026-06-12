"""PTI/v1 binary deserializer for prompt token ID payloads.

Implements the inverse of the tracing gateway's
``prompt_token_ids_serializer.serialize_prompt_token_ids``.
"""

from __future__ import annotations

import base64
import struct
from typing import Any, Dict, List, Tuple

import zstandard as zstd

MAGIC = b"PTI1"
HEADER_VERSION = 1
ENTRY_FORMAT = "<i"
ENTRY_SIZE = struct.calcsize(ENTRY_FORMAT)  # 4 bytes
HEADER_FORMAT = "<4sBBHIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 24 bytes


def _parse_header(raw: bytes) -> Dict[str, Any]:
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"Payload too short for PTI/v1 header: {len(raw)} < {HEADER_SIZE}")

    (
        magic,
        version,
        flags,
        reserved_u16,
        token_count,
        body_byte_length,
        reserved_u64,
    ) = struct.unpack(HEADER_FORMAT, raw[:HEADER_SIZE])

    if magic != MAGIC:
        raise ValueError(f"Bad PTI/v1 magic: {magic!r}")
    if version != HEADER_VERSION:
        raise ValueError(f"Unsupported PTI/v1 header version: {version}")

    return {
        "flags": flags,
        "reserved_u16": reserved_u16,
        "token_count": token_count,
        "body_byte_length": body_byte_length,
        "reserved_u64": reserved_u64,
    }


def parse_prompt_token_ids(raw: bytes) -> Tuple[List[int], Dict[str, Any]]:
    """Parse uncompressed PTI/v1 bytes into prompt token IDs and metadata."""
    header = _parse_header(raw)
    token_count = header["token_count"]
    body_byte_length = header["body_byte_length"]

    if token_count == 0:
        raise ValueError("PTI/v1 token_count must be > 0")
    if body_byte_length != token_count * ENTRY_SIZE:
        raise ValueError(
            f"body_byte_length ({body_byte_length}) != token_count * {ENTRY_SIZE} "
            f"({token_count * ENTRY_SIZE})"
        )

    expected_len = HEADER_SIZE + body_byte_length
    if len(raw) != expected_len:
        raise ValueError(f"PTI/v1 payload length mismatch: {len(raw)} != {expected_len}")

    token_ids: List[int] = []
    offset = HEADER_SIZE
    for _ in range(token_count):
        (token_id,) = struct.unpack(ENTRY_FORMAT, raw[offset : offset + ENTRY_SIZE])
        offset += ENTRY_SIZE
        token_ids.append(token_id)

    metadata: Dict[str, Any] = {
        "scope": "prompt_only",
        "token_count": token_count,
    }
    header.update(metadata)
    return token_ids, header


def decompress_and_parse_pti(data_b64: str) -> Tuple[List[int], Dict[str, Any]]:
    """Decompress and unpack a PTI/v1 prompt token ID payload.

    Args:
        data_b64: Base64-encoded zstd-compressed PTI binary blob from
            ``payloads.prompt_token_ids.data``.

    Returns:
        ``(token_ids, metadata)`` where ``token_ids`` is the prompt token ID
        sequence and ``metadata`` includes ``token_count``.
    """
    compressed = base64.b64decode(data_b64)
    decompressor = zstd.ZstdDecompressor()
    raw = decompressor.decompress(compressed)
    return parse_prompt_token_ids(raw)
