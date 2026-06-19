"""Deprecated compatibility shim for ``eval_protocol.tracing.logprobs``.

Import from ``eval_protocol.tracing.logprobs`` (or ``decode_payloads`` from
``eval_protocol.tracing``) instead. This module re-exports the LP/v1 helpers
that lived here before the tracing package refactor.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "eval_protocol.adapters.lp_deserializer is deprecated; "
    "import from eval_protocol.tracing.logprobs instead.",
    DeprecationWarning,
    stacklevel=2,
)

from eval_protocol.tracing.logprobs import (  # noqa: E402
    ENTRY_FORMAT,
    ENTRY_SIZE,
    HEADER_FORMAT,
    HEADER_SIZE,
    HEADER_VERSION,
    MAGIC,
    MISSING_TOKEN_ID,
    decompress_and_parse_lp,
    parse_logprobs,
)

__all__ = [
    "ENTRY_FORMAT",
    "ENTRY_SIZE",
    "HEADER_FORMAT",
    "HEADER_SIZE",
    "HEADER_VERSION",
    "MAGIC",
    "MISSING_TOKEN_ID",
    "decompress_and_parse_lp",
    "parse_logprobs",
]
