from typing import List, Set
import asyncio

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from tests.pytest.test_markdown_highlighting import markdown_dataset_to_evaluation_row


class TrackingRolloutProcessor(RolloutProcessor):
    """Custom rollout processor that tracks which rollout IDs are generated during rollout phase."""

    def __init__(self, shared_rollout_ids: Set[str]):
        self.shared_rollout_ids = shared_rollout_ids

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Process rows and track rollout IDs generated during rollout phase."""

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            # Track this rollout ID as being generated during rollout phase
            self.shared_rollout_ids.add(row.execution_metadata.rollout_id)
            return row

        # Create tasks that process the rows and track IDs
        tasks = [asyncio.create_task(process_row(row)) for row in rows]
        return tasks


class TrackingLogger(DatasetLogger):
    """Custom logger that tracks all rollout IDs that are logged."""

    def __init__(self, shared_rollout_ids: Set[str]):
        self.shared_rollout_ids = shared_rollout_ids

    def log(self, row: EvaluationRow):
        self.shared_rollout_ids.add(row.execution_metadata.rollout_id)

    def read(self):
        return []


async def test_assertion_error_no_new_rollouts():
    """
    Test that when an assertion error occurs due to failing threshold,
    no new rollout IDs are logged beyond those generated during the rollout phase.
    """
    from eval_protocol.pytest.evaluation_test import evaluation_test

    # Create shared set to track rollout IDs generated during rollout phase
    shared_rollout_ids: Set[str] = set()

    # Create custom processor and logger for tracking with shared set
    rollout_processor = TrackingRolloutProcessor(shared_rollout_ids)
    logger = TrackingLogger(shared_rollout_ids)

    input_dataset: list[str] = [
        "tests/pytest/data/markdown_dataset.jsonl",
    ]
    completion_params: list[dict] = [{"temperature": 0.0, "model": "dummy/local-model"}]

    @evaluation_test(
        input_dataset=input_dataset,
        completion_params=completion_params,
        dataset_adapter=markdown_dataset_to_evaluation_row,
        rollout_processor=rollout_processor,
        mode="pointwise",
        combine_datasets=False,
        num_runs=1,  # Single run to simplify tracking
        passed_threshold=0.5,  # Threshold that will fail since we return 0.0
        logger=logger,
    )
    def eval_fn(row: EvaluationRow) -> EvaluationRow:
        # Always return score 0.0, which will fail the 0.5 threshold
        from eval_protocol.models import EvaluateResult

        row.evaluation_result = EvaluateResult(score=0.0)
        return row

    try:
        # This should fail due to threshold not being met
        for ds_path in input_dataset:
            for completion_param in completion_params:
                await eval_fn(dataset_path=ds_path, completion_params=completion_param)
    except AssertionError:
        # Expected - the threshold check should fail
        pass
    else:
        assert False, "Expected AssertionError due to failing threshold"

    # Get the final set of rollout IDs that were generated during rollout phase
    assert len(shared_rollout_ids) == 19, "Only 19 rollout IDs should have been logged"
