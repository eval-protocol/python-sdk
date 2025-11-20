from typing import Any, Dict, List
import os
import re


from eval_protocol.models import EvaluationRow, Message, EvaluateResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.openenv_rollout_processor import OpenEnvRolloutProcessor
import pytest


# Preferred import when using the monolithic `openenv` package
from envs.echo_env import EchoEnv  # type: ignore


# Skip these integration-heavy tests on CI runners by default
pytestmark = pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip OpenEnv integration tests on CI")


def echo_dataset_to_rows(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Adapter: simple {"id": "...", "prompt": "..."} to EvaluationRows.
    """
    rows: List[EvaluationRow] = []
    for row in data:
        prompt = str(row.get("prompt", "hello"))
        rows.append(EvaluationRow(messages=[Message(role="user", content=prompt)]))
    return rows


def prompt_builder(observation: Any, step: int, history: List[str]) -> str:
    """
    Echo env is very simple; we just send a short instruction.
    """
    return "Please repeat back the next message exactly."


def action_parser(response_text: str):
    """
    Convert raw model response to EchoAction.
    """
    try:
        from envs.echo_env import EchoAction  # type: ignore
    except Exception:
        pytest.skip("OpenEnv (openenv.envs.echo_env) is not installed; skipping Echo hub test.")
        raise
    text = response_text.strip() if isinstance(response_text, str) else ""
    return EchoAction(message=text or "hello")


# try:
#     from envs.echo_env import EchoEnv  # type: ignore

#     _HAS_ECHO = True
# except Exception:
#     _HAS_ECHO = False


# Inline test data
ECHO_INLINE_DATA: List[Dict[str, Any]] = [
    {"id": "echo-1", "prompt": "hello"},
    {"id": "echo-2", "prompt": "test message"},
]


@evaluation_test(  # type: ignore[misc]
    input_rows=[echo_dataset_to_rows(ECHO_INLINE_DATA)],
    completion_params=[
        {
            "temperature": 0.0,
            "max_tokens": 16,
            # Any working model with your API key; match other tests' default
            "model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct",
        }
    ],
    num_runs=1,
    max_concurrent_rollouts=2,
    mode="pointwise",
    rollout_processor=(
        OpenEnvRolloutProcessor(
            # Use HF Hub to launch the environment container automatically
            env_client_cls=EchoEnv,  # type: ignore
            hub_repo_id=os.getenv("OPENENV_ECHO_REPO", "openenv/echo-env"),
            # Simple prompt+parser above
            prompt_builder=prompt_builder,
            action_parser=action_parser,
            # Keep defaults for timeouts/viewport/etc. (not relevant for echo)
            timeout_ms=5000,
            num_generations=1,
        )
    ),
)
def test_openenv_echo_hub(row: EvaluationRow) -> EvaluationRow:
    """
    Smoke test for Echo env via Hugging Face Hub (registry.hf.space/openenv-echo-env).
    Extracts env rewards (from rollout policy extras) and sets evaluation_result.
    """

    # Try to read rewards/usage left in execution metadata extra.
    total_reward = 0.0
    try:
        extra = getattr(row.execution_metadata, "extra", None)
        step_rewards: List[float] = []
        if isinstance(extra, dict):
            raw = extra.get("step_rewards") or []
            step_rewards = [float(r) for r in raw]
            print(f"Step rewards: {step_rewards}")
        total_reward = float(sum(step_rewards)) if step_rewards else 0.0
    except Exception:
        total_reward = 0.0

    score = max(0.0, min(1.0, total_reward))
    row.evaluation_result = EvaluateResult(score=score, reason=f"Echo total reward={total_reward:.2f}")
    return row
