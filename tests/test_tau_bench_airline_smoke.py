"""
Smoke test for tau bench airline evaluation - runs with minimal configuration for CI/CD monitoring.

This is a lightweight version of the full tau bench airline test, designed specifically
for automated smoke testing in CI/CD pipelines. It runs with only 1 iteration to provide
quick feedback on system health while minimizing resource usage.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from eval_protocol.models import CompletionParams, EvaluateResult, EvaluationRow, InputMetadata, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_mcp_gym_rollout_processor import MCPGymRolloutProcessor
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


def tau_bench_airline_smoke_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert entries from airline dataset to EvaluationRow objects for smoke testing.
    """
    rows = []
    test_dir = Path(__file__).parent.parent / "examples" / "tau2_mcp" / "tests"

    # Load system prompt from file so we can change it in one place
    domain = data[0]["environment_context"]["domain"]
    prompt_file = test_dir / f"system_prompts/{domain}_agent_system_prompt.md"

    with open(prompt_file, "r") as f:
        system_prompt = f.read().strip()

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
    input_dataset=["tests/pytest/data/airline_dataset.jsonl"],
    dataset_adapter=tau_bench_airline_smoke_to_evaluation_row,
    completion_params=[
        {
            "temperature": 0.8,
            "extra_body": {"reasoning_effort": "medium"},
            "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
        }
    ],
    rollout_processor=MCPGymRolloutProcessor(),
    passed_threshold=0.36,
    num_runs=1,  # Smoke test: single run for quick feedback
    mode="pointwise",
    max_concurrent_rollouts=50,  # Standard concurrency
    server_script_path="examples/tau2_mcp/server.py",
)
def test_tau_bench_airline_smoke_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Smoke test for tau bench airline evaluation - single run version for CI/CD monitoring.

    This is a lightweight smoke test that runs the tau bench airline evaluation with
    minimal configuration (1 run) to quickly validate system health and model performance.
    It uses the same evaluation logic as the full test but with reduced resource usage.

    Args:
        row: EvaluationRow object from tau bench airline dataset after rollout

    Returns:
        EvaluationRow with tau2 evaluation results
    """
    messages = row.messages

    # Get evaluation criteria and user_simulation from input_metadata.dataset_info
    dataset_info = row.input_metadata.dataset_info if row.input_metadata else {}
    evaluation_criteria = dataset_info.get("evaluation_criteria", {})

    nl_assertions = evaluation_criteria.get("nl_assertions", [])
    communicate_info = evaluation_criteria.get("communicate_info", [])
    actions = evaluation_criteria.get("actions", [])

    # Convert Message objects directly to tau2-bench message objects
    trajectory_objects = []
    for msg in messages:
        role = msg.role
        content = msg.content

        if role == "system":
            trajectory_objects.append(SystemMessage(role=role, content=content))
        elif role == "assistant":
            tau2_tool_calls = []
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    arguments = json.loads(tool_call.function.arguments)
                    tau2_tool_call = ToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=arguments,
                    )
                    tau2_tool_calls.append(tau2_tool_call)

            trajectory_objects.append(AssistantMessage(role=role, content=content, tool_calls=tau2_tool_calls))
        elif role == "user":
            trajectory_objects.append(UserMessage(role=role, content=content))
        elif role == "tool":
            tool_id = msg.tool_call_id
            trajectory_objects.append(ToolMessage(id=tool_id, role=role, content=content))

    reward = 1.0

    evaluation_criteria = EvaluationCriteria(
        nl_assertions=nl_assertions,
        communicate_info=communicate_info,
        actions=actions,
        reward_basis=[
            RewardType.DB,
            RewardType.COMMUNICATE,
        ],
    )

    task = Task(
        id="SmokeTest", evaluation_criteria=evaluation_criteria, user_scenario=UserScenario(instructions="SmokeTest")
    )  # id and user_scenario are required for the Task type but not used in calculating reward

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
    reason = "\n".join(failed_reasons) if failed_reasons else "✅ All checks passed [SMOKE TEST]"

    row.evaluation_result = EvaluateResult(
        score=reward,
        reason=reason,
        metrics={},
    )
    return row
