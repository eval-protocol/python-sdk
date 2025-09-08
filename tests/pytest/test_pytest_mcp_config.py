from typing_extensions import override
import pytest
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import AgentRolloutProcessor, evaluation_test


@evaluation_test(
    input_messages=[
        [
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


@pytest.mark.asyncio
async def test_pytest_tools_are_added_to_row():
    class TrackingLogger(DatasetLogger):
        """Custom logger that ensures that the final row is in an error state."""

        def __init__(self, rollouts: dict[str, EvaluationRow]):
            self.rollouts: dict[str, EvaluationRow] = rollouts

        @override
        def log(self, row: EvaluationRow):
            if row.execution_metadata.rollout_id is None:
                raise ValueError("Rollout ID is None")
            self.rollouts[row.execution_metadata.rollout_id] = row

        @override
        def read(self, row_id: str | None = None) -> list[EvaluationRow]:
            return []

    input_messages = [
        [
            Message(
                role="system",
                content="You are a helpful assistant that can answer questions about Fireworks.",
            ),
        ]
    ]
    completion_params_list = [
        {"model": "dummy/local-model"},
    ]

    rollouts: dict[str, EvaluationRow] = {}
    logger = TrackingLogger(rollouts)

    @evaluation_test(
        input_messages=[input_messages],
        completion_params=completion_params_list,
        rollout_processor=AgentRolloutProcessor(),
        mode="pointwise",
        mcp_config_path="tests/pytest/mcp_configurations/mock_discord_mcp_config.json",
        logger=logger,
    )
    def eval_fn(row: EvaluationRow) -> EvaluationRow:
        return row

    await eval_fn(input_messages=input_messages, completion_params=completion_params_list[0])  # pyright: ignore[reportCallIssue]

    # ensure that the row has tools that were set during AgentRolloutProcessor
    assert len(rollouts) == 1
    row = list(rollouts.values())[0]
    if row.tools is None:
        raise ValueError("Row has no tools")
    assert sorted([tool["function"].name for tool in row.tools]) == sorted(  # pyright: ignore[reportAny]
        ["list_servers", "get_channels", "read_messages"]
    )
