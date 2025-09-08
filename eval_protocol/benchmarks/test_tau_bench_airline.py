"""
Pytest test for tau bench airline evaluation using the evaluation_test decorator.

This test demonstrates how to use tau bench environments within the pytest framework,
similar to the test_entire_airline_dataset test but integrated with the pytest evaluation system.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata, Message
from eval_protocol.pytest import evaluation_test, ExceptionHandlerConfig
from eval_protocol.pytest.default_mcp_gym_rollout_processor import MCPGymRolloutProcessor
import litellm
from litellm.exceptions import RateLimitError, APIConnectionError
from vendor.tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from vendor.tau2.data_model.tasks import Action, EvaluationCriteria, RewardType, Task, UserScenario
from vendor.tau2.evaluator.evaluator import EnvironmentEvaluator
from vendor.tau2.evaluator.evaluator_action import ActionEvaluator
from vendor.tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from vendor.tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
from vendor.tau2.registry import registry
from eval_protocol.mcp_servers.tau2 import get_server_script_path, get_system_prompt


def _ensure_airline_database():
    """Ensure airline database exists, downloading if necessary."""
    import urllib.request
    from pathlib import Path

    # Get the vendor/tau2/data directory path
    try:
        from vendor.tau2.utils.utils import DATA_DIR

        domains_dir = DATA_DIR / "domains"
    except ImportError:
        # Fallback: find vendor/tau2 relative to this file
        vendor_tau2 = Path(__file__).parent.parent.parent / "vendor" / "tau2"
        domains_dir = vendor_tau2 / "data" / "domains"

    # Only download airline database for this test
    airline_db_path = domains_dir / "airline" / "db.json"
    if not airline_db_path.exists():
        print(f"📥 Downloading airline database to {airline_db_path}...")
        airline_db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            url = "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/airline/db.json"
            urllib.request.urlretrieve(url, airline_db_path)
            print(f"✅ Downloaded airline database ({airline_db_path.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"❌ Failed to download airline database: {e}")
            raise


# Ensure airline database is available before test runs
_ensure_airline_database()


def _get_airline_dataset_path() -> str:
    """Get the airline dataset file path."""
    return str(Path(__file__).parent / "data" / "airline_dataset.jsonl")


def _get_server_script_path() -> str:
    """Get the tau2 mcp server script path."""
    from eval_protocol.mcp_servers.tau2 import get_server_script_path

    return get_server_script_path()


def tau_bench_airline_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert entries from airline dataset to EvaluationRow objects.
    """
    rows = []
    # Load system prompt from file so we can change it in one place
    from eval_protocol.mcp_servers.tau2 import get_system_prompt

    domain = data[0]["environment_context"]["domain"]
    system_prompt = get_system_prompt(domain)

    for row in data:
        eval_row = EvaluationRow(
            messages=[Message(role="system", content=system_prompt)],
            input_metadata=InputMetadata(
                row_id=row["id"],
                dataset_info={
                    "environment_context": row["environment_context"],
                    "user_simulation": row["user_simulation"],
                    "evaluation_criteria": row["evaluation_criteria"],
                    "user_prompt_template": row["user_prompt_template"],
                },
            ),
        )

        rows.append(eval_row)

    return rows


@evaluation_test(
    input_dataset=[_get_airline_dataset_path()],
    dataset_adapter=tau_bench_airline_to_evaluation_row,
    completion_params=[
        {
            "temperature": 0.8,
            "max_tokens": 4096,
            "extra_body": {"reasoning_effort": "medium"},
            "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
        }
    ],
    rollout_processor=MCPGymRolloutProcessor(),
    rollout_processor_kwargs={"domain": "airline"},
    passed_threshold={"success": 0.4, "standard_error": 0.02},
    num_runs=4,
    mode="pointwise",
    max_concurrent_rollouts=50,
    server_script_path=_get_server_script_path(),
    exception_handler_config=ExceptionHandlerConfig(
        retryable_exceptions={
            RateLimitError,
            APIConnectionError,
        }
    ),
)
def test_tau_bench_airline_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Test tau bench airline evaluation using the pytest framework.

    This test now uses the tau_bench_airline_reward function which automatically
    extracts evaluation criteria from dataset entries. No wrapper needed!

    Args:
        row: EvaluationRow object from tau bench airline dataset after rollout

    Returns:
        EvaluationRow with tau2 evaluation results
    """
    messages = row.messages

    # Get evaluation criteria and user_simulation from input_metadata.dataset_info
    dataset_info = (row.input_metadata.dataset_info or {}) if row.input_metadata else {}
    evaluation_criteria = dataset_info.get("evaluation_criteria", {})

    nl_assertions = evaluation_criteria.get("nl_assertions", [])
    communicate_info = evaluation_criteria.get("communicate_info", [])
    actions = evaluation_criteria.get("actions", [])

    # Convert Message objects directly to tau2-bench message objects
    trajectory_objects = []
    for msg in messages:
        role = msg.role
        content = msg.content

        # Normalize content to str for tau2 message models
        text_content = content if isinstance(content, str) or content is None else ""
        if role == "system":
            trajectory_objects.append(SystemMessage(role=role, content=text_content))
        elif role == "assistant":
            tau2_tool_calls = []
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    arguments = json.loads(tool_call.function.arguments)
                    tau2_tool_call = ToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=arguments,
                        requestor="assistant",
                    )
                    tau2_tool_calls.append(tau2_tool_call)

            trajectory_objects.append(AssistantMessage(role=role, content=text_content, tool_calls=tau2_tool_calls))
        elif role == "user":
            trajectory_objects.append(UserMessage(role=role, content=text_content))
        elif role == "tool":
            tool_id = msg.tool_call_id
            trajectory_objects.append(
                ToolMessage(id=tool_id or "unknown_tool_call", role=role, content=text_content, requestor="assistant")
            )

    reward = 1.0

    evaluation_criteria = EvaluationCriteria(
        nl_assertions=nl_assertions,
        communicate_info=communicate_info,
        actions=actions,
        env_assertions=None,
        reward_basis=[  # Use this to adjust how to calculate reward. Tau2-bench uses DB and COMMUNICATE by default for airline tasks.
            RewardType.DB,
            RewardType.COMMUNICATE,
        ],
    )

    task = Task(
        id="Filler",
        description=None,
        user_scenario=UserScenario(instructions="Filler", persona=None),
        ticket=None,
        initial_state=None,
        evaluation_criteria=evaluation_criteria,
    )  # id and user_scenario are required for the Task type but not used in calculating reward
    assert task.evaluation_criteria is not None, "Task evaluation criteria is None"

    if RewardType.DB in task.evaluation_criteria.reward_basis:
        env_reward_info = EnvironmentEvaluator.calculate_reward(
            environment_constructor=registry.get_env_constructor("airline"),
            task=task,
            full_trajectory=trajectory_objects,
        )
    if RewardType.ACTION in task.evaluation_criteria.reward_basis:
        action_reward_info = ActionEvaluator.calculate_reward(
            task=task,
            full_trajectory=trajectory_objects,
        )
    if RewardType.COMMUNICATE in task.evaluation_criteria.reward_basis:
        communicate_reward_info = CommunicateEvaluator.calculate_reward(
            task=task,
            full_trajectory=trajectory_objects,
        )
    if RewardType.NL_ASSERTION in task.evaluation_criteria.reward_basis:
        nl_reward_info = NLAssertionsEvaluator.calculate_reward(
            task=task,
            full_trajectory=trajectory_objects,
        )

    reward = 1.0
    env_bases = {RewardType.DB, RewardType.ENV_ASSERTION}
    action_bases = {RewardType.ACTION}
    nl_bases = {RewardType.NL_ASSERTION}
    comm_bases = {RewardType.COMMUNICATE}
    task_reward_basis = set(task.evaluation_criteria.reward_basis)

    reward_breakdown = {}
    if task_reward_basis & env_bases:
        if env_reward_info.reward_breakdown is not None:
            reward_breakdown.update(env_reward_info.reward_breakdown)
        reward *= env_reward_info.reward
    if task_reward_basis & action_bases:
        if action_reward_info.reward_breakdown is not None:
            reward_breakdown.update(action_reward_info.reward_breakdown)
        reward *= action_reward_info.reward
    if task_reward_basis & nl_bases:
        if nl_reward_info.reward_breakdown is not None:
            reward_breakdown.update(nl_reward_info.reward_breakdown)
        reward *= nl_reward_info.reward
    if task_reward_basis & comm_bases:
        if communicate_reward_info.reward_breakdown is not None:
            reward_breakdown.update(communicate_reward_info.reward_breakdown)
        reward *= communicate_reward_info.reward

    # Generate reason showing only failed components
    failed_reasons = []

    if task_reward_basis & env_bases and env_reward_info.reward == 0:
        failed_reasons.append("❌ Environment/DB check failed")

    if task_reward_basis & action_bases and action_reward_info.reward == 0:
        failed_actions = []
        if hasattr(action_reward_info, "action_checks") and action_reward_info.action_checks:
            failed_actions = [
                f"{ac.action.name}({ac.action.arguments})"
                for ac in action_reward_info.action_checks
                if not ac.action_match
            ]
        if failed_actions:
            failed_reasons.append(f"❌ Failed actions: {failed_actions}")
        else:
            failed_reasons.append("❌ Actions failed")

    if task_reward_basis & nl_bases and nl_reward_info.reward == 0:
        failed_nl = []
        if hasattr(nl_reward_info, "nl_assertions") and nl_reward_info.nl_assertions:
            failed_nl = [nla.nl_assertion for nla in nl_reward_info.nl_assertions if not nla.met]
        if failed_nl:
            failed_reasons.append(f"❌ Failed NL assertions: {failed_nl}")
        else:
            failed_reasons.append("❌ NL Assertions failed")

    if task_reward_basis & comm_bases and communicate_reward_info.reward == 0:
        failed_comm = []
        if hasattr(communicate_reward_info, "communicate_checks") and communicate_reward_info.communicate_checks:
            failed_comm = [cc.info for cc in communicate_reward_info.communicate_checks if not cc.met]
        if failed_comm:
            failed_reasons.append(f"❌ Failed communication: {failed_comm}")
        else:
            failed_reasons.append("❌ Communication failed")

    # If everything passed, show success
    reason = "\n".join(failed_reasons) if failed_reasons else "✅ All checks passed"

    row.evaluation_result = EvaluateResult(
        score=reward,
        reason=reason,
        metrics={},
    )
    return row
