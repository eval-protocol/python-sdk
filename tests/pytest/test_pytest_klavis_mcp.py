from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import AgentRolloutProcessor, evaluation_test


@evaluation_test(
    input_messages=[
        [
            [
                Message(
                    role="system",
                    content=(
                        "You are a helpful assistant that can answer questions about Gmail. You have access to tools to help you find information.\n"
                    ),
                ),
                Message(
                    role="user",
                    content=("Find the first 5 emails title in my inbox."),
                ),
            ]
        ]
    ],
    rollout_processor=AgentRolloutProcessor(),
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct"}],
    mode="pointwise",
    mcp_config_path="tests/pytest/mcp_configurations/klavis_strata_mcp.json",
)
def test_pytest_klavis_mcp(row: EvaluationRow) -> EvaluationRow:
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
