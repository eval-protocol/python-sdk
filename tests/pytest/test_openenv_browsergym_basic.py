import asyncio
import os
import shutil
from typing import Any, Dict, List

import pytest

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.pytest.openenv_rollout_processor import OpenEnvRolloutProcessor

# Skip these integration-heavy tests on CI runners by default
pytestmark = pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip OpenEnv integration tests on CI")


@pytest.mark.integration
def test_openenv_browsergym_basic():
    """
    Very basic integration test to ensure OpenEnv + BrowserGym can run a single-step rollout.
    Skips automatically if Docker is not available.
    """
    if shutil.which("docker") is None:
        pytest.skip("Docker not available on PATH; skipping OpenEnv BrowserGym basic test.")

    # Build a minimal EvaluationRow (messages can be empty; processor will add user prompts)
    rows: List[EvaluationRow] = [EvaluationRow(messages=[Message(role="user", content="start")])]

    # Use tasks that are known to exist; requires MiniWoB server reachable from containers.
    tasks = ["click-test"]
    miniwob_url = os.getenv("MINIWOB_URL", "http://172.17.0.1:8888/miniwob/")
    browsergym_env_vars: Dict[str, str] = {
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_HEADLESS": "true",
        "BROWSERGYM_VIEWPORT_WIDTH": "1280",
        "BROWSERGYM_VIEWPORT_HEIGHT": "720",
        "BROWSERGYM_TIMEOUT": "10000",
        "BROWSERGYM_OBS_AXTREE": "1",
        "BROWSERGYM_OBS_PRUNED_HTML": "1",
        "BROWSERGYM_RETURN_INFO": "1",
    }
    if miniwob_url:
        browsergym_env_vars["MINIWOB_URL"] = miniwob_url

    # Construct the processor with a trivial action_parser; the model output will still be generated
    # but we parse to a safe noop action to minimize flakiness for the environment step.
    from envs.browsergym_env import BrowserGymAction, BrowserGymEnv  # type: ignore

    processor = OpenEnvRolloutProcessor(
        env_factory=None,
        prompt_builder=lambda obs, step, history: "Do nothing",
        action_parser=lambda text: BrowserGymAction(action_str="noop()"),
        env_client_cls=BrowserGymEnv,
        tasks=tasks,
        task_var="BROWSERGYM_TASK_NAME",
        miniwob_url=miniwob_url,
        docker_image="browsergym-env:latest",
        benchmark="miniwob",
        timeout_ms=10000,
        num_generations=1,
        env_vars=browsergym_env_vars,
    )

    # Completion params: rely on an available provider/model in the environment
    completion_params: Dict[str, Any] = {
        "model": os.getenv(
            "OPENENV_TEST_MODEL",
            # Default to a Fireworks public model id used elsewhere in tests; requires FIREWORKS_API_KEY
            "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct-0905",
        ),
        "temperature": 0.0,
        "max_tokens": 16,
    }

    # Limit to a single step to keep the test fast and robust
    config = RolloutProcessorConfig(
        completion_params=completion_params,
        semaphore=asyncio.Semaphore(1),
        steps=1,
        mcp_config_path="",
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:

        async def _run_all():
            tasks_ = processor(rows, config)
            return await asyncio.gather(*tasks_)

        completed_rows = loop.run_until_complete(_run_all())
    finally:
        loop.close()

    assert len(completed_rows) == 1
    # Basic sanity checks that a rollout happened and usage is populated
    row = completed_rows[0]
    assert row is not None
    assert row.execution_metadata is not None
    assert getattr(row.execution_metadata, "duration_seconds", 0.0) >= 0.0
