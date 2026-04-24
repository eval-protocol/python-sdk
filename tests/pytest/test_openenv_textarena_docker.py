from typing import Any, Dict, List
import os

from eval_protocol.models import EvaluationRow, Message, EvaluateResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.openenv_rollout_processor import OpenEnvRolloutProcessor
import pytest

# Skip these integration-heavy tests on CI runners by default
pytestmark = pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip OpenEnv integration tests on CI")


def textarena_dataset_to_rows(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Adapter: simple {"id": "...", "prompt": "..."} to EvaluationRows.
    """
    rows: List[EvaluationRow] = []
    for row in data:
        prompt = str(row.get("prompt", "Let's play"))
        rows.append(EvaluationRow(messages=[Message(role="user", content=prompt)]))
    return rows


def prompt_builder(observation: Any, step: int, history: List[str]) -> str:
    """
    Build prompt for TextArena games.
    Extract the game prompt and recent messages.
    """
    prompt_text = getattr(observation, "prompt", "")
    messages = getattr(observation, "messages", [])

    # Format conversation history
    history_lines = []
    for msg in messages[-5:]:  # Last 5 messages
        sender = getattr(msg, "sender_id", "?")
        content = getattr(msg, "content", "")
        category = getattr(msg, "category", "MESSAGE")
        if content:
            history_lines.append(f"[{category}] Player {sender}: {content}")

    history_str = "\n".join(history_lines) if history_lines else "[No messages yet]"

    return (
        f"Step {step}\n"
        f"Game: {prompt_text}\n\n"
        f"Conversation:\n{history_str}\n\n"
        f"Your move (reply with your guess or action):"
    )


def action_parser(response_text: str):
    """
    Convert raw model response to TextArenaAction.
    """
    try:
        from envs.textarena_env import TextArenaAction  # type: ignore
    except Exception:
        pytest.skip("OpenEnv (envs.textarena_env) is not installed; skipping TextArena test.")
        raise

    # Extract the actual guess/action from the response
    text = response_text.strip() if isinstance(response_text, str) else ""

    # Try to extract text in brackets [guess] or quotes "guess"
    import re

    bracket_match = re.search(r"\[([^\]]+)\]", text)
    if bracket_match:
        text = bracket_match.group(1).strip()

    return TextArenaAction(message=text or "pass")


try:
    from envs.textarena_env import TextArenaEnv  # type: ignore

    _HAS_TEXTARENA = True
except Exception:
    _HAS_TEXTARENA = False


# Inline test data
TEXTARENA_INLINE_DATA: List[Dict[str, Any]] = [
    {"id": "wordle-1", "prompt": "Play Wordle"},
    {"id": "wordle-2", "prompt": "Play Wordle"},
    {"id": "wordle-3", "prompt": "Play Wordle"},
]


@evaluation_test(  # type: ignore[misc]
    input_rows=[textarena_dataset_to_rows(TEXTARENA_INLINE_DATA)],
    completion_params=[
        {
            "temperature": 0.7,
            "max_tokens": 32,
            # Any working model with your API key
            "model": "fireworks_ai/accounts/fireworks/models/kimi-k2p5",
            "reasoning_effort": "none",
        }
    ],
    num_runs=1,
    max_concurrent_rollouts=2,
    mode="pointwise",
    rollout_processor=(
        OpenEnvRolloutProcessor(
            # Use Docker image built from OpenEnv repo
            env_client_cls=TextArenaEnv if _HAS_TEXTARENA else None,  # type: ignore
            docker_image=os.getenv("TEXTARENA_DOCKER_IMAGE", "textarena-env:latest"),
            env_vars={
                "TEXTARENA_ENV_ID": os.getenv("TEXTARENA_ENV_ID", "Wordle-v0"),
                "TEXTARENA_NUM_PLAYERS": os.getenv("TEXTARENA_NUM_PLAYERS", "1"),
                # Optional: add TEXTARENA_KW_* for game-specific kwargs
                # e.g., "TEXTARENA_KW_hardcore": "true"
            },
            task_var="TEXTARENA_ENV_ID",  # Env var for task selection
            tasks=None,  # Single task per container (set via TEXTARENA_ENV_ID)
            prompt_builder=prompt_builder,
            action_parser=action_parser,
            timeout_ms=10000,
            num_generations=1,
        )
        if _HAS_TEXTARENA
        else None
    ),
)
def test_openenv_textarena_docker(row: EvaluationRow) -> EvaluationRow:
    """
    Test TextArena (Wordle, GuessTheNumber, etc.) via Docker container.

    Build the image first:
        cd /path/to/OpenEnv
        docker build -f src/envs/textarena_env/server/Dockerfile -t textarena-env:latest .

    Run with:
        TEXTARENA_ENV_ID=Wordle-v0 TEXTARENA_NUM_PLAYERS=1 \\
        FIREWORKS_API_KEY=$FIREWORKS_API_KEY \\
        pytest tests/pytest/test_openenv_textarena_docker.py -vv -s

    Or test other games:
        TEXTARENA_ENV_ID=GuessTheNumber-v0 ...
    """
    if not _HAS_TEXTARENA:
        pytest.skip("OpenEnv (envs.textarena_env) is not installed; skipping TextArena Docker test.")

    # Extract step rewards and compute score
    total_reward = 0.0
    try:
        extra = getattr(row.execution_metadata, "extra", None)
        step_rewards: List[float] = []
        if isinstance(extra, dict):
            raw = extra.get("step_rewards") or []
            step_rewards = [float(r) for r in raw]
        total_reward = float(sum(step_rewards)) if step_rewards else 0.0
    except Exception:
        total_reward = 0.0

    score = max(0.0, min(1.0, total_reward))
    steps = len(step_rewards) if "step_rewards" in locals() else 0
    row.evaluation_result = EvaluateResult(
        score=score,
        reason=f"TextArena total reward={total_reward:.2f} over {steps} steps",
    )
    return row
