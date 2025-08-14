from datetime import datetime
from typing import List

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import AgentRolloutProcessor, evaluation_test


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
    rollout_processor=AgentRolloutProcessor(),
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-20b"}],
    mode="pointwise",
    mcp_config_path="tests/pytest/mcp_configurations/mock_discord_mcp_config.json",
)
def test_pytest_mcp_config(row: EvaluationRow) -> EvaluationRow:
    """Test Stdio MCP Config usage in decorator"""
    # filter for all tool calls
    tool_calls = [msg for msg in row.messages if msg.role == "tool"]

    if len(tool_calls) == 0:
        row.evaluation_result = EvaluateResult(
            score=0,
            reason="No tool calls made",
        )
        return row

    row.evaluation_result = EvaluateResult(
        score=1,
        reason="At least one tool call was made",
    )
    return row
