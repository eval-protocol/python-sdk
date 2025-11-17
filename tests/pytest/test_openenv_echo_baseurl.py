from typing import Any, Dict, List
import os

import pytest

from eval_protocol.models import EvaluationRow, Message, EvaluateResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.openenv_rollout_processor import OpenEnvRolloutProcessor

# Skip these integration-heavy tests on CI runners by default
pytestmark = pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip OpenEnv integration tests on CI")


def echo_dataset_to_rows(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    rows: List[EvaluationRow] = []
    for row in data:
        prompt = str(row.get("prompt", "hello"))
        rows.append(EvaluationRow(messages=[Message(role="user", content=prompt)]))
    return rows


def prompt_builder(observation: Any, step: int, history: List[str]) -> str:
    return "Please repeat back the next message exactly."


def action_parser(response_text: str):
    try:
        from envs.echo_env import EchoAction  # type: ignore
    except Exception:
        pytest.skip("OpenEnv (envs.echo_env) is not installed; skipping Echo base_url test.")
        raise
    text = response_text.strip() if isinstance(response_text, str) else ""
    return EchoAction(message=text or "hello")


def _score_from_system_rewards(row: EvaluationRow) -> float:
    step_rewards: List[float] = []
    try:
        for msg in row.messages or []:
            if msg.role == "system" and isinstance(msg.content, str) and msg.content.startswith("__ep_step_rewards__:"):
                import json as _json
                payload = msg.content.split(":", 1)[1]
                step_rewards = _json.loads(payload) or []
                break
    except Exception:
        step_rewards = []
    total = float(sum(step_rewards)) if step_rewards else 0.0
    return max(0.0, min(1.0, total))


try:
    from envs.echo_env import EchoEnv  # type: ignore
    _HAS_ECHO = True
except Exception:
    _HAS_ECHO = False


@evaluation_test(  # type: ignore[misc]
    input_dataset=["tests/pytest/data/echo_dataset.jsonl"],
    dataset_adapter=echo_dataset_to_rows,
    completion_params=[
        {
            "temperature": 0.0,
            "max_tokens": 16,
            "model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct",
        }
    ],
    num_runs=1,
    max_concurrent_rollouts=2,
    mode="pointwise",
    rollout_processor=(
        OpenEnvRolloutProcessor(
            env_client_cls=EchoEnv,  # type: ignore
            env_base_url=os.getenv("OPENENV_ECHO_BASE_URL"),  # e.g., http://0.0.0.0:8001 (docker or local)
            prompt_builder=prompt_builder,
            action_parser=action_parser,
            timeout_ms=5000,
            num_generations=1,
        )
        if _HAS_ECHO
        else None
    ),
)
def test_openenv_echo_baseurl_local_or_docker(row: EvaluationRow) -> EvaluationRow:
    """
    Base URL connectivity test for Echo env (local Python server or Docker).
    Requires OPENENV_ECHO_BASE_URL to be set; otherwise, test is skipped.
    """
    if not os.getenv("OPENENV_ECHO_BASE_URL"):
        pytest.skip("OPENENV_ECHO_BASE_URL not set; skipping local/docker echo test.")
    if not _HAS_ECHO:
        pytest.skip("OpenEnv (envs.echo_env) is not installed; skipping Echo base_url test.")
    score = _score_from_system_rewards(row)
    row.evaluation_result = EvaluateResult(score=score, reason=f"Echo (base_url) score={score:.2f}")
    return row


@evaluation_test(  # type: ignore[misc]
    input_dataset=["tests/pytest/data/echo_dataset.jsonl"],
    dataset_adapter=echo_dataset_to_rows,
    completion_params=[
        {
            "temperature": 0.0,
            "max_tokens": 16,
            "model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct",
        }
    ],
    num_runs=1,
    max_concurrent_rollouts=2,
    mode="pointwise",
    rollout_processor=(
        OpenEnvRolloutProcessor(
            env_client_cls=EchoEnv,  # type: ignore
            env_base_url=os.getenv("OPENENV_ECHO_SPACE_URL"),  # e.g., https://openenv-echo-env.hf.space
            prompt_builder=prompt_builder,
            action_parser=action_parser,
            timeout_ms=5000,
            num_generations=1,
        )
        if _HAS_ECHO
        else None
    ),
)
def test_openenv_echo_baseurl_space(row: EvaluationRow) -> EvaluationRow:
    """
    Space URL connectivity test for Echo env (remote HF Space).
    Requires OPENENV_ECHO_SPACE_URL to be set; otherwise, test is skipped.
    """
    if not os.getenv("OPENENV_ECHO_SPACE_URL"):
        pytest.skip("OPENENV_ECHO_SPACE_URL not set; skipping space echo test.")
    if not _HAS_ECHO:
        pytest.skip("OpenEnv (envs.echo_env) is not installed; skipping Echo base_url test.")
    score = _score_from_system_rewards(row)
    row.evaluation_result = EvaluateResult(score=score, reason=f"Echo (space) score={score:.2f}")
    return row


