"""Decode Fireworks tracing-gateway payloads.

Standalone, dependency-light helpers (stdlib + ``zstandard`` only) for turning
the binary/JSON ``payloads`` returned by the Fireworks tracing gateway
(``GET /traces?include_payloads=true``) into Python values. No EvaluationRow or
rollout machinery required -- usable on its own.

Typical use::

    from eval_protocol.tracing import decode_payloads, PayloadType

    decoded = decode_payloads(trace["payloads"])
    decoded[PayloadType.PROMPT_TOKEN_IDS].value   # List[int]
    decoded[PayloadType.LOGPROBS].value           # List[float]
    decoded[PayloadType.ROUTER_REPLAY].value      # List[Optional[str]]

See ``README.md`` in this package for details.
"""

from __future__ import annotations

from .registry import decode_payload, decode_payloads, decode_trace
from .types import DecodedPayload, PayloadType

__all__ = [
    "PayloadType",
    "DecodedPayload",
    "decode_payloads",
    "decode_payload",
    "decode_trace",
]
