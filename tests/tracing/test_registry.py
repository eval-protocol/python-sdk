"""Tests for the standalone tracing-gateway payload decoder registry."""

from __future__ import annotations

import base64
import json
import struct

import zstandard as zstd

from eval_protocol.tracing import (
    DecodedPayload,
    PayloadType,
    decode_payload,
    decode_payloads,
    decode_trace,
)
from eval_protocol.tracing import logprobs as lp_mod
from eval_protocol.tracing import router_replay as r3_mod


def _b64_zstd(raw: bytes) -> str:
    return base64.b64encode(zstd.ZstdCompressor().compress(raw)).decode("ascii")


def _pti_data(token_ids: list[int]) -> str:
    return _b64_zstd(json.dumps(token_ids).encode("utf-8"))


def _lp_data(tokens: list[tuple[int, float]]) -> str:
    body = b"".join(struct.pack(lp_mod.ENTRY_FORMAT, tid, lp) for tid, lp in tokens)
    header = struct.pack(
        lp_mod.HEADER_FORMAT, lp_mod.MAGIC, 1, 0, 0, len(tokens), len(body), 0
    )
    return _b64_zstd(header + body)


def _r3_data_all_mode(matrices: list[bytes]) -> str:
    matrix_data = b"".join(matrices)
    header = struct.pack(
        r3_mod.HEADER_FORMAT,
        r3_mod.MAGIC,
        1,  # version
        r3_mod._SelectorMode.ALL,
        r3_mod._RoutingDtype.UINT8,
        0x01,  # flags
        len(matrices),  # total_token_count
        len(matrices),  # replayed_token_count
        0,  # replay_start_token
        0,  # selector_byte_length
        len(matrix_data),
    )
    return _b64_zstd(header + matrix_data)


def _all_payloads() -> dict:
    return {
        "prompt_token_ids": {"manifest": {"PayloadVersion": "pti/v1"}, "data": _pti_data([1, 2, 3])},
        "logprobs": {"manifest": {"PayloadVersion": "lp/v1"}, "data": _lp_data([(7, -0.25), (8, -0.5)])},
        "router_replay": {
            "manifest": {"PayloadVersion": "r3/v1"},
            "data": _r3_data_all_mode([b"\x01\x02\x03\x04", b"\x05\x06\x07\x08"]),
        },
    }


def test_decode_payloads_all_types():
    decoded = decode_payloads(_all_payloads())

    assert set(decoded) == {
        PayloadType.PROMPT_TOKEN_IDS,
        PayloadType.LOGPROBS,
        PayloadType.ROUTER_REPLAY,
    }
    assert all(isinstance(dp, DecodedPayload) for dp in decoded.values())

    assert decoded[PayloadType.PROMPT_TOKEN_IDS].value == [1, 2, 3]

    lp = decoded[PayloadType.LOGPROBS]
    assert lp.value == [-0.25, -0.5]
    assert lp.token_ids == [7, 8]

    r3 = decoded[PayloadType.ROUTER_REPLAY]
    assert len(r3.value) == 2
    assert base64.b64decode(r3.value[0]) == b"\x01\x02\x03\x04"


def test_decode_payload_accepts_str_and_enum():
    data = _pti_data([10, 20])
    via_enum = decode_payload(PayloadType.PROMPT_TOKEN_IDS, data)
    via_str = decode_payload("prompt_token_ids", data)
    assert via_enum.value == via_str.value == [10, 20]


def test_decode_trace_reaches_into_payloads():
    trace = {"id": "t1", "payloads": {"prompt_token_ids": {"data": _pti_data([5, 6])}}}
    decoded = decode_trace(trace)
    assert decoded[PayloadType.PROMPT_TOKEN_IDS].value == [5, 6]


def test_unknown_and_empty_types_are_skipped():
    payloads = {
        "some_future_type": {"data": "ignored"},  # unknown -> ignored
        "logprobs": {"data": ""},  # present but empty -> skipped
        "prompt_token_ids": {"data": _pti_data([9])},
    }
    decoded = decode_payloads(payloads)
    assert set(decoded) == {PayloadType.PROMPT_TOKEN_IDS}


def test_on_error_fires_on_bad_data():
    errors = []
    payloads = {"prompt_token_ids": {"data": "not-valid-base64-zstd-json!!"}}

    decoded = decode_payloads(payloads, on_error=lambda pt, e: errors.append((pt, e)))

    assert decoded == {}
    assert len(errors) == 1
    assert errors[0][0] == PayloadType.PROMPT_TOKEN_IDS


def test_decode_payloads_non_dict_returns_empty():
    assert decode_payloads(None) == {}
    assert decode_trace({"id": "no-payloads"}) == {}
