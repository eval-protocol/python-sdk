from eval_protocol.models import EvaluateResult, Message, EvaluationRow
from eval_protocol.pytest import default_agent_rollout_processor, evaluation_test


@evaluation_test(
    input_messages=[
        [
            Message(
                role="system",
                content=(
                    "You are a helpful assistant that can answer questions about Fireworks.\n"
                    "ALWAYS provide code or commands to execute to answer the question."
                ),
            ),
            Message(
                role="user",
                content=("Can you teach me about how to manage deployments on Fireworks"),
            ),
        ]
    ],
    rollout_processor=default_agent_rollout_processor,
    model=["fireworks_ai/accounts/fireworks/models/kimi-k2-instruct"],
    mode="pointwise",
    mcp_config_path="tests/pytest/mcp_configurations/docs_mcp_config.json",
)
def test_pytest_mcp_url(row: EvaluationRow) -> EvaluationRow:
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
