from typing import Set
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.default_agent_rollout_processor import AgentRolloutProcessor
from eval_protocol.dataset_logger import DatasetLogger


class TrackingLogger(DatasetLogger):
    """Custom logger that ensures that the final row is in an error state."""

    def __init__(self, rollouts: dict[str, EvaluationRow]):
        self.rollouts = rollouts

    def log(self, row: EvaluationRow):
        self.rollouts[row.execution_metadata.rollout_id] = row

    def read(self):
        return []


async def test_pytest_propagate_error():
    """
    Properly propagate errors from rollout processing to eval_metadata.status.
    To test this, we use a broken MCP configuration that should fail during the
    rollout processing. Then the final eval_metadata.status should be an error.
    This way the UI can properly render an error state for the rollout and a
    developer can identify and investigate the error.
    """
    from eval_protocol.pytest.evaluation_test import evaluation_test

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
        input_messages=input_messages,
        completion_params=completion_params_list,
        rollout_processor=AgentRolloutProcessor(),
        mode="pointwise",
        num_runs=5,
        mcp_config_path="tests/pytest/mcp_configurations/docs_mcp_config_broken.json",
        logger=logger,
    )
    def eval_fn(row: EvaluationRow) -> EvaluationRow:
        return row

    # Manually invoke all parameter combinations within a single test
    for params in completion_params_list:
        await eval_fn(input_messages=input_messages, completion_params=params)

    # assert that the status of eval_metadata.status is "error"
    assert len(rollouts) == 5
    assert all(row.eval_metadata.status.is_error() for row in rollouts.values())

    # make sure the error message includes details of the error
    assert all("HTTPStatusError" in row.rollout_status.message for row in rollouts.values())
    assert all("405 Method Not Allowed" in row.rollout_status.message for row in rollouts.values())
    assert all("https://docs.fireworks.ai/mcp-non-existent" in row.rollout_status.message for row in rollouts.values())
