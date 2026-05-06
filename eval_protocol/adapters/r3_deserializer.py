"""R3/v1 binary deserializer for router-replay payloads.

Implements the inverse of the packed binary format produced by the tracing
gateway's ``r3_serializer.serialize_r3``.  See that module for the full
header specification.

The main entry point is :func:`decompress_and_parse_r3`, which accepts the
base64-encoded compressed blob returned by the gateway's
``/v1/traces/pointwise?include_payloads=true`` endpoint and produces
per-token routing matrices in the same ``List[Optional[str]]`` format used
by the direct inference path (``DeploymentSampler.sample_with_tokens()``).
"""

from __future__ import annotations

import base64
import logging
import struct
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import zstandard as zstd

logger = logging.getLogger(__name__)

MAGIC = b"R3V1"
HEADER_FORMAT = "<4sBBBBIIIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 32 bytes


class _SelectorMode(IntEnum):
    ALL = 0
    SUFFIX = 1
    BITMAP = 2


class _RoutingDtype(IntEnum):
    UINT8 = 1
    UINT16 = 2


_SELECTOR_MODE_NAMES = {v: v.name.lower() for v in _SelectorMode}
_ROUTING_DTYPE_NAMES = {v: v.name.lower() for v in _RoutingDtype}


def _parse_header(raw: bytes) -> Dict[str, Any]:
    if len(raw) < HEADER_SIZE:
        raise ValueError(
            f"Payload too short for r3/v1 header: {len(raw)} < {HEADER_SIZE}"
        )

    (
        magic,
        version,
        selector_mode,
        routing_dtype,
        flags,
        total_token_count,
        replayed_token_count,
        replay_start_token,
        selector_byte_length,
        matrix_byte_length,
    ) = struct.unpack(HEADER_FORMAT, raw[:HEADER_SIZE])

    if magic != MAGIC:
        raise ValueError(f"Bad R3 magic: {magic!r}")
    if version != 1:
        raise ValueError(f"Unsupported R3 header version: {version}")

    return {
        "selector_mode": selector_mode,
        "routing_dtype": routing_dtype,
        "flags": flags,
        "total_token_count": total_token_count,
        "replayed_token_count": replayed_token_count,
        "replay_start_token": replay_start_token,
        "selector_byte_length": selector_byte_length,
        "matrix_byte_length": matrix_byte_length,
    }


def _read_bitmap_positions(
    selector_bytes: bytes, total_token_count: int
) -> List[int]:
    """Return sorted token indices where the bitmap bit is set."""
    positions: List[int] = []
    for i in range(total_token_count):
        byte_idx = i >> 3
        bit_idx = i & 7
        if byte_idx < len(selector_bytes) and (selector_bytes[byte_idx] >> bit_idx) & 1:
            positions.append(i)
    return positions


def decompress_and_parse_r3(
    data_b64: str,
) -> Tuple[List[Optional[str]], Dict[str, Any]]:
    """Decompress and unpack an R3/v1 payload into per-token routing matrices.

    Args:
        data_b64: Base64-encoded zstd-compressed R3 binary blob, as returned
            by the tracing gateway in ``payloads.router_replay.data``.

    Returns:
        A tuple of ``(routing_matrices, metadata)`` where:

        - ``routing_matrices`` is a ``List[Optional[str]]`` of length
          ``total_token_count``.  Each present position contains a
          base64-encoded routing matrix (matching the format returned by
          the direct inference path); absent positions are ``None``.
        - ``metadata`` is a dict with keys ``routing_dtype``,
          ``selector_mode``, ``total_token_count``, ``replayed_token_count``,
          ``replay_start_token``.
    """
    compressed = base64.b64decode(data_b64)

    # ZstdCompressor.compress() embeds the uncompressed size in the frame
    # header by default, so the library can auto-allocate the output buffer.
    decompressor = zstd.ZstdDecompressor()
    raw = decompressor.decompress(compressed)

    header = _parse_header(raw)

    selector_mode = header["selector_mode"]
    routing_dtype = header["routing_dtype"]
    total_token_count = header["total_token_count"]
    replayed_token_count = header["replayed_token_count"]
    replay_start_token = header["replay_start_token"]
    selector_byte_length = header["selector_byte_length"]
    matrix_byte_length = header["matrix_byte_length"]

    # Per-token matrix byte size is implicit in the payload: all replayed
    # tokens share the same matrix length, so we can recover it from the
    # matrix section total length divided by the replayed-token count.
    if replayed_token_count > 0:
        if matrix_byte_length % replayed_token_count != 0:
            raise ValueError(
                f"matrix_byte_length ({matrix_byte_length}) is not a multiple of "
                f"replayed_token_count ({replayed_token_count}); cannot split "
                "into per-token matrices"
            )
        matrix_elem_size = matrix_byte_length // replayed_token_count
    else:
        matrix_elem_size = 0

    body = raw[HEADER_SIZE:]
    selector_bytes = body[:selector_byte_length]
    matrix_bytes = body[selector_byte_length : selector_byte_length + matrix_byte_length]

    if matrix_elem_size == 0:
        replayed_positions: List[int] = []
    elif selector_mode == _SelectorMode.ALL:
        replayed_positions = list(range(total_token_count))
    elif selector_mode == _SelectorMode.SUFFIX:
        replayed_positions = list(
            range(replay_start_token, replay_start_token + replayed_token_count)
        )
    elif selector_mode == _SelectorMode.BITMAP:
        replayed_positions = _read_bitmap_positions(selector_bytes, total_token_count)
    else:
        raise ValueError(f"Unknown selector_mode: {selector_mode}")

    # Split matrix bytes into per-token chunks and base64-encode each one
    matrices: List[Optional[str]] = [None] * total_token_count
    for idx, pos in enumerate(replayed_positions):
        start = idx * matrix_elem_size
        end = start + matrix_elem_size
        if end > len(matrix_bytes):
            logger.warning(
                "R3 matrix data truncated at token %d (position %d): "
                "expected %d bytes but only %d remaining",
                idx, pos, matrix_elem_size, len(matrix_bytes) - start,
            )
            break
        matrices[pos] = base64.b64encode(matrix_bytes[start:end]).decode("ascii")

    metadata: Dict[str, Any] = {
        "routing_dtype": _ROUTING_DTYPE_NAMES.get(routing_dtype, str(routing_dtype)),
        "selector_mode": _SELECTOR_MODE_NAMES.get(selector_mode, str(selector_mode)),
        "total_token_count": total_token_count,
        "replayed_token_count": replayed_token_count,
        "replay_start_token": replay_start_token,
    }

    return matrices, metadata
