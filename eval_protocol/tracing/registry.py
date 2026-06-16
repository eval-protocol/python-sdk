"""Decoder registry + master decode for tracing-gateway payloads.

Adding a new payload type is a single entry in ``PAYLOAD_DECODERS`` (plus its
decoder module). Callers use the master :func:`decode_payloads` /
:func:`decode_trace` and never stitch per-type decoders together.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from .logprobs import decode_logprobs
from .prompt_token_ids import decode_prompt_token_ids
from .router_replay import decode_router_replay
from .types import DecodedPayload, PayloadType

logger = logging.getLogger(__name__)

# Callback invoked when a single payload fails to decode: (payload_type, exc).
OnError = Callable[[PayloadType, Exception], None]

PAYLOAD_DECODERS: Dict[PayloadType, Callable[[str], DecodedPayload]] = {
    PayloadType.PROMPT_TOKEN_IDS: decode_prompt_token_ids,
    PayloadType.LOGPROBS: decode_logprobs,
    PayloadType.ROUTER_REPLAY: decode_router_replay,
}


def decode_payload(payload_type: PayloadType | str, data_b64: str) -> DecodedPayload:
    """Decode a single payload by type.

    ``payload_type`` accepts a ``PayloadType`` or its string value, so external
    callers can pass either.
    """
    ptype = PayloadType(payload_type)
    decoder = PAYLOAD_DECODERS.get(ptype)
    if decoder is None:
        raise ValueError(f"No decoder registered for payload type: {ptype!r}")
    return decoder(data_b64)


def decode_payloads(
    payloads: Dict[str, Any],
    *,
    on_error: Optional[OnError] = None,
) -> Dict[PayloadType, DecodedPayload]:
    """Master decode: run every registered decoder over a gateway ``payloads`` dict.

    Args:
        payloads: The ``payloads`` object from a gateway trace (i.e.
            ``trace["payloads"]``), mapping payload-type name -> ``{"manifest", "data"}``.
        on_error: Optional callback ``(payload_type, exc)`` invoked when a present
            payload fails to decode. Defaults to logging a warning. A failure in
            one payload never blocks the others.

    Returns:
        ``{PayloadType: DecodedPayload}`` for every payload that is present and
        decodes successfully. Only known ``PayloadType`` members are considered,
        so unknown payload types from a newer gateway are ignored rather than
        raising.
    """
    if not isinstance(payloads, dict):
        return {}

    decoded: Dict[PayloadType, DecodedPayload] = {}
    for ptype, decoder in PAYLOAD_DECODERS.items():
        entry = payloads.get(ptype)
        if not isinstance(entry, dict) or not entry.get("data"):
            continue
        try:
            decoded[ptype] = decoder(entry["data"])
        except Exception as exc:  # noqa: BLE001 - isolate per-payload failures
            if on_error is not None:
                on_error(ptype, exc)
            else:
                logger.warning("Failed to decode %s payload: %s", ptype.value, exc)
    return decoded


def decode_trace(
    trace: Dict[str, Any],
    *,
    on_error: Optional[OnError] = None,
) -> Dict[PayloadType, DecodedPayload]:
    """Convenience wrapper around :func:`decode_payloads` for a raw trace dict."""
    payloads = trace.get("payloads") if isinstance(trace, dict) else None
    return decode_payloads(payloads or {}, on_error=on_error)
