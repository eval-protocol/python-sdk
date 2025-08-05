from datetime import datetime
from typing import List

from eval_protocol.models import EvaluateResult, Message, EvaluationRow
from eval_protocol.pytest import default_agent_rollout_processor, evaluation_test


@evaluation_test(
    input_messages=[
        [
            Message(
                role="user",
                content=(
                    "Can you give me a summary of every channel. "
                    "You can list servers and channels using the "
                    "list_servers and get_channels tools. And you can "
                    "read messages using the read_messages tool."
                ),
            )
        ]
    ],
    rollout_processor=default_agent_rollout_processor,
    model=["gpt-4.1"],
    mode="pointwise",
    mcp_config_path="tests/pytest/mcp_configurations/mock_discord_mcp_config.json",
)
def test_pytest_mcp_config(row: EvaluationRow) -> EvaluationRow:
    """Run math evaluation on sample dataset using pytest interface."""
    # filter for all tool calls
    tool_calls = [msg for msg in row.messages if msg.role == "tool"]

    if len(tool_calls) == 0:
        row.evaluation_result = EvaluateResult(
            score=0,
            feedback="No tool calls made",
        )
        return row

    row.evaluation_result = EvaluateResult(
        score=1,
        feedback="At least one tool call was made",
    )
    return row
