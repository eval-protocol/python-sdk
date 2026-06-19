"""Deprecated compatibility shim for ``eval_protocol.tracing.router_replay``.

Import from ``eval_protocol.tracing.router_replay`` (or ``decode_payloads`` from
``eval_protocol.tracing``) instead. This module re-exports the R3/v1 helpers
that lived here before the tracing package refactor.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "eval_protocol.adapters.r3_deserializer is deprecated; "
    "import from eval_protocol.tracing.router_replay instead.",
    DeprecationWarning,
    stacklevel=2,
)

from eval_protocol.tracing.router_replay import (  # noqa: E402
    BITS_PER_BYTE,
    HEADER_FORMAT,
    HEADER_SIZE,
    MAGIC,
    _RoutingDtype,
    _SelectorMode,
    _parse_header,
    _read_bitmap_positions,
    decompress_and_parse_r3,
)

__all__ = [
    "BITS_PER_BYTE",
    "HEADER_FORMAT",
    "HEADER_SIZE",
    "MAGIC",
    "_RoutingDtype",
    "_SelectorMode",
    "_parse_header",
    "_read_bitmap_positions",
    "decompress_and_parse_r3",
]
