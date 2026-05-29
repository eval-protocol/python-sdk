"""Minimal e2e test for logprobs trace payloads via RemoteRolloutProcessor.

Spins up the reference remote server locally, which makes the LLM call
through litellm-gateway-dev. RemoteRolloutProcessor polls the dev gateway
and fetches traces with include_payloads=True.

Run with:
    cd eval-protocol-python-sdk
    FIREWORKS_API_KEY="$FIREWORKS_DEV_API_KEY" \\
      pytest tests/manual/test_logprobs_e2e.py -v -s

Requires gateway+consumer dev deploy with logprobs payload support and deployment:
    accounts/pyroworks-dev/deployments/malaysia2-careful-paprika
"""

import os
import socket
import subprocess
import sys
import time
from typing import List

import pytest
import requests

from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, EvaluateResult, Message, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor

DEPLOYMENT = "accounts/pyroworks-dev/deployments/malaysia2-careful-paprika"
GATEWAY_DEV_URL = "https://litellm-gateway-dev-j4kzagdteq-uc.a.run.app"
FIREWORKS_DEV_INFERENCE_BASE = "https://dev.api.fireworks.ai/inference/v1"


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


SERVER_PORT = _find_available_port()


def _wait_for_server(port: int, timeout: int = 30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(f"http://127.0.0.1:{port}")
            return
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    raise TimeoutError(f"Remote server did not start within {timeout}s")


@pytest.fixture
def remote_server_module(request):
    return getattr(request, "param", "tests.remote_server.remote_server")


@pytest.fixture(autouse=True)
def _remote_server(remote_server_module):
    env = os.environ.copy()
    env["FW_TRACING_GATEWAY_BASE_URL"] = GATEWAY_DEV_URL
    api_key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("FIREWORKS_DEV_API_KEY")
    if api_key:
        env["FIREWORKS_API_KEY"] = api_key
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            remote_server_module,
            "--host",
            "127.0.0.1",
            "--port",
            str(SERVER_PORT),
        ],
        env=env,
    )
    _wait_for_server(SERVER_PORT)
    yield
    proc.terminate()
    proc.wait()


def input_rows() -> List[EvaluationRow]:
    return [
        EvaluationRow(messages=[Message(role="user", content="What is 2+2?")]),
    ]


def two_turn_input_rows() -> List[EvaluationRow]:
    return [
        EvaluationRow(messages=[Message(role="user", content="What is 2+2?")]),
    ]


def _logprobs_content(message: Message) -> list:
    if not message.logprobs:
        return []
    return message.logprobs.get("content") or []


@pytest.mark.parametrize(
    "completion_params",
    [
        {
            "model": DEPLOYMENT,
            "logprobs": True,
            "base_url": FIREWORKS_DEV_INFERENCE_BASE,
        }
    ],
)
@evaluation_test(
    data_loaders=DynamicDataLoader(generators=[input_rows]),
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url=f"http://127.0.0.1:{SERVER_PORT}",
        model_base_url=GATEWAY_DEV_URL,
        include_payloads=True,
        timeout_seconds=180,
    ),
)
async def test_logprobs_present(row: EvaluationRow) -> EvaluationRow:
    """Verify completion logprobs and Message.logprobs after remote rollout."""

    has_response = len(row.messages) > 1
    assistant_msg = row.messages[-1] if has_response else None

    extra = row.execution_metadata.extra or {}
    completion_logprobs = extra.get("completion_logprobs") or []
    has_completion_logprobs = len(completion_logprobs) > 0

    message_content = None
    if assistant_msg and assistant_msg.logprobs:
        message_content = assistant_msg.logprobs.get("content") or []

    has_message_logprobs = message_content is not None and len(message_content) > 0
    lengths_match = (
        has_completion_logprobs
        and has_message_logprobs
        and len(message_content) == len(completion_logprobs)
    )

    if has_completion_logprobs:
        print(
            f"\n  Logprobs OK: {len(completion_logprobs)} completion tokens"
            f" | message.content len={len(message_content) if message_content else 0}"
        )
    else:
        print(f"\n  No logprobs in extra={extra}")

    score = 1.0 if (has_response and has_completion_logprobs and lengths_match) else 0.0
    reason_parts = []
    if not has_response:
        reason_parts.append("no assistant response")
    if not has_completion_logprobs:
        reason_parts.append("no completion_logprobs in execution_metadata.extra")
    if not lengths_match:
        reason_parts.append(
            f"message.logprobs content length ({len(message_content or [])}) "
            f"!= completion_logprobs ({len(completion_logprobs)})"
        )

    reason = "All checks passed" if score == 1.0 else "; ".join(reason_parts)

    row.evaluation_result = EvaluateResult(
        score=score,
        reason=reason,
        metrics={
            "has_response": MetricResult(
                score=float(has_response),
                is_score_valid=True,
                reason="got response" if has_response else "no response",
            ),
            "has_completion_logprobs": MetricResult(
                score=float(has_completion_logprobs),
                is_score_valid=True,
                reason="present" if has_completion_logprobs else "missing",
            ),
            "logprobs_lengths_match": MetricResult(
                score=float(lengths_match),
                is_score_valid=True,
                reason="match" if lengths_match else "mismatch",
            ),
        },
    )

    assert has_response, f"Expected assistant response. Messages: {row.messages}"
    assert has_completion_logprobs, (
        f"Expected completion_logprobs in extra but got: {row.execution_metadata.extra}"
    )
    assert lengths_match, (
        "Expected len(message.logprobs['content']) == len(completion_logprobs); "
        f"got {len(message_content or [])} vs {len(completion_logprobs)}"
    )

    return row


@pytest.mark.parametrize(
    "remote_server_module",
    ["tests.remote_server.remote_server_two_turn_logprobs"],
    indirect=True,
)
@pytest.mark.parametrize(
    "completion_params",
    [
        {
            "model": DEPLOYMENT,
            "logprobs": True,
            "base_url": FIREWORKS_DEV_INFERENCE_BASE,
        }
    ],
)
@evaluation_test(
    data_loaders=DynamicDataLoader(generators=[two_turn_input_rows]),
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url=f"http://127.0.0.1:{SERVER_PORT}",
        model_base_url=GATEWAY_DEV_URL,
        include_payloads=True,
        timeout_seconds=180,
    ),
)
async def test_two_turn_logprobs_present(row: EvaluationRow) -> EvaluationRow:
    """Verify each assistant turn in a two-turn remote rollout has logprobs."""

    roles = [message.role for message in row.messages]
    assistant_messages = row.get_assistant_messages()
    logprob_lengths = [len(_logprobs_content(message)) for message in assistant_messages]

    has_two_turn_shape = roles == ["user", "assistant", "user", "assistant"]
    has_two_assistant_turns = len(assistant_messages) == 2
    all_turns_have_logprobs = has_two_assistant_turns and all(length > 0 for length in logprob_lengths)

    extra = row.execution_metadata.extra or {}
    final_completion_logprobs = extra.get("completion_logprobs") or []
    assistant_turn_payloads = extra.get("assistant_turn_payloads") or []
    final_lengths_match = (
        has_two_assistant_turns
        and len(final_completion_logprobs) > 0
        and len(final_completion_logprobs) == logprob_lengths[-1]
    )
    has_payloads_for_each_turn = len(assistant_turn_payloads) == len(assistant_messages)
    turn_payload_lengths_match = has_payloads_for_each_turn and all(
        payload.get("assistant_turn_index") == idx
        and len(payload.get("completion_logprobs") or []) == logprob_lengths[idx]
        for idx, payload in enumerate(assistant_turn_payloads)
    )

    if all_turns_have_logprobs:
        print(f"\n  Two-turn logprobs OK: assistant token counts={logprob_lengths}")
    else:
        print(f"\n  Missing two-turn logprobs: roles={roles} token_counts={logprob_lengths}")

    all_ok = (
        has_two_turn_shape
        and all_turns_have_logprobs
        and final_lengths_match
        and turn_payload_lengths_match
    )
    reason_parts = []
    if not has_two_turn_shape:
        reason_parts.append(f"expected user/assistant/user/assistant roles but got {roles}")
    if not has_two_assistant_turns:
        reason_parts.append(f"expected 2 assistant turns but got {len(assistant_messages)}")
    if has_two_assistant_turns and not all_turns_have_logprobs:
        reason_parts.append(f"missing assistant logprobs; token_counts={logprob_lengths}")
    if not final_lengths_match:
        reason_parts.append(
            "final assistant message logprobs length "
            f"({logprob_lengths[-1] if logprob_lengths else 0}) "
            f"!= completion_logprobs ({len(final_completion_logprobs)})"
        )
    if not has_payloads_for_each_turn:
        reason_parts.append(f"expected per-turn payloads for each assistant turn but got {assistant_turn_payloads}")
    if has_payloads_for_each_turn and not turn_payload_lengths_match:
        reason_parts.append(f"per-turn payload lengths do not match message logprobs: {assistant_turn_payloads}")

    row.evaluation_result = EvaluateResult(
        score=1.0 if all_ok else 0.0,
        reason="All checks passed" if all_ok else "; ".join(reason_parts),
        metrics={
            "has_two_turn_shape": MetricResult(
                score=float(has_two_turn_shape),
                is_score_valid=True,
                reason="match" if has_two_turn_shape else "unexpected roles",
            ),
            "all_turns_have_logprobs": MetricResult(
                score=float(all_turns_have_logprobs),
                is_score_valid=True,
                reason="present" if all_turns_have_logprobs else "missing",
            ),
            "final_logprobs_lengths_match": MetricResult(
                score=float(final_lengths_match),
                is_score_valid=True,
                reason="match" if final_lengths_match else "mismatch",
            ),
            "turn_payload_lengths_match": MetricResult(
                score=float(turn_payload_lengths_match),
                is_score_valid=True,
                reason="match" if turn_payload_lengths_match else "mismatch",
            ),
        },
    )

    assert has_two_turn_shape, f"Expected two-turn conversation but got roles: {roles}"
    assert all_turns_have_logprobs, (
        "Expected logprobs on both assistant turns; "
        f"token_counts={logprob_lengths}, messages={row.messages}"
    )
    assert final_lengths_match, (
        "Expected final assistant logprobs to match completion_logprobs; "
        f"got {logprob_lengths[-1] if logprob_lengths else 0} vs {len(final_completion_logprobs)}"
    )
    assert turn_payload_lengths_match, (
        "Expected assistant_turn_payloads to match each assistant turn's logprobs; "
        f"payloads={assistant_turn_payloads}, token_counts={logprob_lengths}"
    )

    return row
