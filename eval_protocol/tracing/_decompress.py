"""Shared helper for the Fireworks tracing-gateway payload decoders."""

from __future__ import annotations

import base64

import zstandard as zstd


def decompress_b64(data_b64: str) -> bytes:
    """Base64-decode then zstd-decompress a gateway ``payloads.*.data`` blob.

    The gateway stores every payload as ``base64(zstd(raw_bytes))``; this is the
    common first step every decoder shares before interpreting ``raw_bytes``.
    """
    compressed = base64.b64decode(data_b64)
    return zstd.ZstdDecompressor().decompress(compressed)
