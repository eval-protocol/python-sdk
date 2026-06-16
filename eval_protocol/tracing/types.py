"""Shared types for the Fireworks tracing-gateway payload decoders."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class PayloadType(str, Enum):
    """Known out-of-band trace payload types emitted by the tracing gateway.

    Canonical source of truth for payload-type names on the EP side. Mirrors the
    gateway's ``rft_tracing.schemas.PayloadType`` but is defined locally so this
    package has no dependency on the gateway/mono codebase. Being a ``str`` enum,
    members compare and hash equal to their string value, so they can be used
    directly against the gateway's string-keyed ``payloads`` JSON.
    """

    PROMPT_TOKEN_IDS = "prompt_token_ids"
    LOGPROBS = "logprobs"
    ROUTER_REPLAY = "router_replay"


@dataclass(frozen=True)
class DecodedPayload:
    """A single decoded gateway payload.

    ``value`` shape depends on ``payload_type``:
      - ``PROMPT_TOKEN_IDS`` -> ``List[int]``
      - ``LOGPROBS``         -> ``List[float]`` (per completion token); the
        optional per-token ids are in ``token_ids``
      - ``ROUTER_REPLAY``    -> ``List[Optional[str]]`` (per-token base64 routing
        matrices, ``None`` where absent)
    """

    payload_type: PayloadType
    value: Any
    metadata: Dict[str, Any]
    # LOGPROBS only: per-completion-token ids, or None if any were missing.
    token_ids: Optional[List[int]] = None
