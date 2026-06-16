# `eval_protocol.tracing` — Fireworks tracing-gateway payload decoders

Standalone helpers for decoding the out-of-band **payloads** the Fireworks
tracing gateway stores alongside a trace (prompt token IDs, completion logprobs,
router-replay routing matrices).

This package is intentionally self-contained: it depends only on the stdlib and
`zstandard`. It does **not** import `EvaluationRow`, rollout processors, or any
other Eval Protocol machinery, so you can use it even if you are not using EP for
rollouts — just point at it for extracting gateway payloads.

## What is a "payload"?

When you read a trace with payloads included:

```
GET {gateway}/v1/traces?rollout_id=...&include_payloads=true
```

each trace carries a `payloads` object like:

```json
{
  "payloads": {
    "prompt_token_ids": {
      "manifest": { "PayloadVersion": "pti/v1", "...": "..." },
      "data": "<base64 of zstd-compressed bytes>"
    },
    "logprobs":      { "manifest": { "PayloadVersion": "lp/v1" }, "data": "..." },
    "router_replay": { "manifest": { "PayloadVersion": "r3/v1" }, "data": "..." }
  }
}
```

The `data` field is `base64(zstd(raw_bytes))`. Each payload type has its own
`raw_bytes` encoding (`pti/v1` is a JSON int array; `lp/v1` and `r3/v1` are packed
binary). This package hides all of that.

## Usage

Decode everything at once (the common case):

```python
from eval_protocol.tracing import decode_payloads, PayloadType

decoded = decode_payloads(trace["payloads"])

if PayloadType.PROMPT_TOKEN_IDS in decoded:
    token_ids = decoded[PayloadType.PROMPT_TOKEN_IDS].value      # List[int]

if PayloadType.LOGPROBS in decoded:
    lp = decoded[PayloadType.LOGPROBS]
    logprobs = lp.value                                          # List[float]
    token_ids = lp.token_ids                                     # Optional[List[int]]

if PayloadType.ROUTER_REPLAY in decoded:
    matrices = decoded[PayloadType.ROUTER_REPLAY].value          # List[Optional[str]]
```

If you have the whole trace dict, `decode_trace(trace)` reaches into
`trace["payloads"]` for you.

Decode a single payload:

```python
from eval_protocol.tracing import decode_payload, PayloadType

dp = decode_payload(PayloadType.PROMPT_TOKEN_IDS, trace["payloads"]["prompt_token_ids"]["data"])
dp.value  # List[int]
```

### Error handling

`decode_payloads` isolates per-payload failures: if one payload fails to decode,
the others are still returned. Pass `on_error=callback(payload_type, exc)` to
control logging (defaults to a warning):

```python
decode_payloads(payloads, on_error=lambda pt, e: print(f"{pt} failed: {e}"))
```

## Return type

`decode_payloads` / `decode_trace` return `Dict[PayloadType, DecodedPayload]`.

`DecodedPayload` fields:

| field          | meaning                                                            |
|----------------|-------------------------------------------------------------------|
| `payload_type` | `PayloadType` enum member                                         |
| `value`        | decoded value (type depends on `payload_type`, see below)         |
| `metadata`     | decoded header/manifest metadata (token counts, scope, etc.)      |
| `token_ids`    | `Optional[List[int]]` — LOGPROBS per-token ids (else `None`)      |

`value` by type:

| `PayloadType`       | `value`                  | notes                                        |
|---------------------|--------------------------|----------------------------------------------|
| `PROMPT_TOKEN_IDS`  | `List[int]`              | prompt token ids                             |
| `LOGPROBS`          | `List[float]`            | per completion token; ids in `token_ids` (or `None`) |
| `ROUTER_REPLAY`     | `List[Optional[str]]`    | per-token base64 routing matrices; `None` where absent |

## Adding a new payload type

1. Add a member to `PayloadType` in `types.py`.
2. Add a `decode_<name>(data_b64) -> DecodedPayload` function in a new module.
3. Register it in `PAYLOAD_DECODERS` in `registry.py`.

`decode_payloads` picks it up automatically.
