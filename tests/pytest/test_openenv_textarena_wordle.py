from typing import Any, Dict, List
import os
import re

from eval_protocol.models import EvaluationRow, Message, EvaluateResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.openenv_rollout_processor import OpenEnvRolloutProcessor
import pytest
import os

# Skip these integration-heavy tests on CI runners by default
pytestmark = pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip OpenEnv integration tests on CI")


def wordle_dataset_to_rows(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Adapter: simple {"id": "...", "prompt": "..."} to EvaluationRows.
    Prompts are ignored by the environment; they just seed the conversation.
    """
    rows: List[EvaluationRow] = []
    for row in data:
        prompt = str(row.get("prompt", "start"))
        rows.append(EvaluationRow(messages=[Message(role="user", content=prompt)]))
    return rows


def prompt_builder(observation: Any, step: int, history: List[str]) -> str:
    """
    Build a minimal instruction for Wordle turns.
    """
    prompt = getattr(observation, "prompt", "") or ""
    return f"You are playing Wordle. Based on previous feedback, choose a valid 5-letter word.\nContext:\n{prompt}\nReply with only the guess."


def action_parser(response_text: str):
    """
    Convert model response to TextArenaAction (message).
    """
    try:
        from envs.textarena_env import TextArenaAction  # type: ignore
    except Exception:
        pytest.skip("OpenEnv (envs.textarena_env) is not installed; skipping TextArena test.")
        raise
    text = (response_text or "").strip()
    # Keep only the first word-like token
    guess = re.split(r"[^A-Za-z]+", text)[0] if text else ""
    guess = guess[:5] if guess else "crane"
    return TextArenaAction(message=guess.lower())


def _score_from_rewards(row: EvaluationRow) -> float:
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
    # Clamp to [0,1] for dashboard score
    return max(0.0, min(1.0, total))


try:
    from envs.textarena_env import TextArenaEnv  # type: ignore
    _HAS_TEXTARENA = True
except Exception:
    _HAS_TEXTARENA = False


@evaluation_test(  # type: ignore[misc]
    input_dataset=["tests/pytest/data/wordle_dataset.jsonl"],
    dataset_adapter=wordle_dataset_to_rows,
    completion_params=[
        {
            "temperature": 0.0,
            "max_tokens": 8,
            "model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct",
        }
    ],
    num_runs=1,
    max_concurrent_rollouts=1,
    mode="pointwise",
    rollout_processor=(
        OpenEnvRolloutProcessor(
            env_client_cls=TextArenaEnv,  # type: ignore
            hub_repo_id=os.getenv("OPENENV_TEXTARENA_REPO", "burtenshaw/textarena"),
            # Pass Wordle settings to the container
            env_vars={
                "TEXTARENA_ENV_ID": os.getenv("TEXTARENA_ENV_ID", "Wordle-v0"),
                "TEXTARENA_NUM_PLAYERS": os.getenv("TEXTARENA_NUM_PLAYERS", "1"),
            },
            prompt_builder=prompt_builder,
            action_parser=action_parser,
            timeout_ms=10000,
            num_generations=1,
        )
        if _HAS_TEXTARENA
        else None
    ),
)
def test_openenv_textarena_wordle_hub(row: EvaluationRow) -> EvaluationRow:
    """
    Smoke test for TextArena Wordle via HF Hub (registry.hf.space/burtenshaw-textarena).
    Requires Docker available to start the Space container.
    """
    if not _HAS_TEXTARENA:
        pytest.skip("OpenEnv (envs.textarena_env) is not installed; skipping TextArena test.")
    score = _score_from_rewards(row)
    row.evaluation_result = EvaluateResult(score=score, reason=f"Wordle total score={score:.2f}")
    return row


