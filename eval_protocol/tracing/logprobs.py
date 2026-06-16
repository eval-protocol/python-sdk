"""LP/v1 binary deserializer for per-token logprobs payloads.

Implements the inverse of the tracing gateway's ``logprobs_serializer.serialize_logprobs``.
See that module for the full header specification.
"""

from __future__ import annotations

import base64
import struct
from typing import Any, Dict, List, Optional, Tuple

import zstandard as zstd

from .types import DecodedPayload, PayloadType

MAGIC = b"LP01"
HEADER_VERSION = 1
MISSING_TOKEN_ID = -1
ENTRY_FORMAT = "<if"
ENTRY_SIZE = struct.calcsize(ENTRY_FORMAT)  # 8 bytes
HEADER_FORMAT = "<4sBBHIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 24 bytes


def _parse_header(raw: bytes) -> Dict[str, Any]:
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"Payload too short for lp/v1 header: {len(raw)} < {HEADER_SIZE}")

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
        raise ValueError(f"Bad LP/v1 magic: {magic!r}")
    if version != HEADER_VERSION:
        raise ValueError(f"Unsupported lp/v1 header version: {version}")

    return {
        "flags": flags,
        "reserved_u16": reserved_u16,
        "token_count": token_count,
        "body_byte_length": body_byte_length,
        "reserved_u64": reserved_u64,
    }


def parse_logprobs(raw: bytes) -> Tuple[List[float], Optional[List[int]], Dict[str, Any]]:
    """Parse uncompressed LP/v1 bytes into logprobs, optional token ids, and metadata."""
    header = _parse_header(raw)
    token_count = header["token_count"]
    body_byte_length = header["body_byte_length"]

    if token_count == 0:
        raise ValueError("LP/v1 token_count must be > 0")
    if body_byte_length != token_count * ENTRY_SIZE:
        raise ValueError(
            f"body_byte_length ({body_byte_length}) != token_count * {ENTRY_SIZE} "
            f"({token_count * ENTRY_SIZE})"
        )

    expected_len = HEADER_SIZE + body_byte_length
    if len(raw) != expected_len:
        raise ValueError(f"LP/v1 payload length mismatch: {len(raw)} != {expected_len}")

    logprobs: List[float] = []
    token_ids: List[int] = []
    all_token_ids_valid = True
    offset = HEADER_SIZE
    for _ in range(token_count):
        wire_id, logprob = struct.unpack(ENTRY_FORMAT, raw[offset : offset + ENTRY_SIZE])
        offset += ENTRY_SIZE
        logprobs.append(logprob)
        if wire_id == MISSING_TOKEN_ID:
            all_token_ids_valid = False
            token_ids.append(wire_id)
        else:
            token_ids.append(wire_id)

    metadata: Dict[str, Any] = {
        "scope": "completion_only",
        "completion_token_count": token_count,
        "all_token_ids_valid": all_token_ids_valid,
    }
    header.update(metadata)
    ids_out: Optional[List[int]] = token_ids if all_token_ids_valid else None
    return logprobs, ids_out, header


def decompress_and_parse_lp(data_b64: str) -> Tuple[List[float], Optional[List[int]], Dict[str, Any]]:
    """Decompress and unpack an LP/v1 payload into completion logprobs and token ids.

    Args:
        data_b64: Base64-encoded zstd-compressed LP binary blob from
            ``payloads.logprobs.data``.

    Returns:
        ``(logprobs, token_ids, metadata)`` where ``logprobs`` is per-completion-token
        scalars, ``token_ids`` is ``None`` if any wire id was ``MISSING_TOKEN_ID``,
        and ``metadata`` includes ``all_token_ids_valid`` and ``completion_token_count``.
    """
    compressed = base64.b64decode(data_b64)
    decompressor = zstd.ZstdDecompressor()
    raw = decompressor.decompress(compressed)
    return parse_logprobs(raw)


def decode_logprobs(data_b64: str) -> DecodedPayload:
    """Decode a gateway ``payloads.logprobs.data`` blob into a ``DecodedPayload``.

    ``value`` is the per-completion-token logprob list; per-token ids (when all
    valid) are available under ``token_ids``.
    """
    logprobs, token_ids, metadata = decompress_and_parse_lp(data_b64)
    return DecodedPayload(
        payload_type=PayloadType.LOGPROBS,
        value=logprobs,
        metadata=metadata,
        token_ids=token_ids,
    )
