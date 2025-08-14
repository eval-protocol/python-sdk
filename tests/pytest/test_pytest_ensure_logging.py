import os
from unittest.mock import Mock, patch

import eval_protocol.dataset_logger as dataset_logger
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.dataset_logger.sqlite_evaluation_row_store import SqliteEvaluationRowStore
from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.default_no_op_rollout_processor import NoOpRolloutProcessor
from tests.pytest.test_markdown_highlighting import markdown_dataset_to_evaluation_row


async def test_ensure_logging(monkeypatch):
    """
    Ensure that default SQLITE logger gets called by mocking the storage and checking that the storage is called.
    """
    # Mock the SqliteEvaluationRowStore to track calls
    mock_store = Mock()
    mock_store.upsert_row = Mock()
    mock_store.read_rows = Mock(return_value=[])
    mock_store.db_path = "/tmp/test.db"

    # Mock the SqliteEvaluationRowStore constructor so that when SqliteDatasetLoggerAdapter
    # creates its store, it gets our mock instead
    with patch(
        "eval_protocol.dataset_logger.sqlite_dataset_logger_adapter.SqliteEvaluationRowStore", return_value=mock_store
    ):
        from eval_protocol.models import EvaluationRow
        from eval_protocol.pytest.evaluation_test import evaluation_test
        from tests.pytest.test_markdown_highlighting import markdown_dataset_to_evaluation_row

    @evaluation_test(
        input_dataset=[
            "tests/pytest/data/markdown_dataset.jsonl",
        ],
        completion_params=[{"temperature": 0.0, "model": "dummy/local-model"}],
        dataset_adapter=markdown_dataset_to_evaluation_row,
        rollout_processor=NoOpRolloutProcessor(),
        mode="pointwise",
        combine_datasets=False,
        num_runs=2,
        # Don't pass logger parameter - let it use the default_logger (which we've replaced)
    )
    def eval_fn(row: EvaluationRow) -> EvaluationRow:
        return row

    await eval_fn(
        dataset_path=["tests/pytest/data/markdown_dataset.jsonl"],
        completion_params={"temperature": 0.0, "model": "dummy/local-model"},
    )

    # Verify that the store's upsert_row method was called
    assert mock_store.upsert_row.called, "SqliteEvaluationRowStore.upsert_row should have been called"

    # Check that it was called multiple times (once for each row)
    call_count = mock_store.upsert_row.call_count
    assert call_count > 0, f"Expected upsert_row to be called at least once, but it was called {call_count} times"

    # Verify the calls were made with proper data structure
    for call in mock_store.upsert_row.call_args_list:
        args, kwargs = call
        data = args[0] if args else kwargs.get("data")
        assert data is not None, "upsert_row should be called with data parameter"
        assert isinstance(data, dict), "data should be a dictionary"
        assert "execution_metadata" in data, "data should contain execution_metadata"
        assert "rollout_id" in data["execution_metadata"], "data should contain rollout_id in execution_metadata"
