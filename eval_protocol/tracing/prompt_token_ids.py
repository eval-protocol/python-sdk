"""``pti/v1`` decoder for prompt token ID payloads.

Inverse of the tracing gateway's ``serialize_prompt_token_ids``: the gateway
stores prompt token IDs as ``base64(zstd(json.dumps(token_ids)))`` -- a compact
JSON int array, no bespoke binary header.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Tuple

import zstandard as zstd

from .types import DecodedPayload, PayloadType


def parse_prompt_token_ids(raw: bytes) -> Tuple[List[int], Dict[str, Any]]:
    """Parse uncompressed ``pti/v1`` bytes (a JSON int array) into ids + metadata."""
    token_ids = json.loads(raw)
    metadata: Dict[str, Any] = {"scope": "prompt_only", "token_count": len(token_ids)}
    return token_ids, metadata


def decode_prompt_token_ids(data_b64: str) -> DecodedPayload:
    """Decode a gateway ``payloads.prompt_token_ids.data`` blob."""
    raw = zstd.ZstdDecompressor().decompress(base64.b64decode(data_b64))
    token_ids, metadata = parse_prompt_token_ids(raw)
    return DecodedPayload(
        payload_type=PayloadType.PROMPT_TOKEN_IDS,
        value=token_ids,
        metadata=metadata,
    )
