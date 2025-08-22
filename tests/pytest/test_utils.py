import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from eval_protocol.pytest.utils import rollout_processor_with_retry
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.models import EvaluationRow, Status, InputMetadata, ExecutionMetadata
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger


class TestRolloutProcessorWithRetry:
    """Test the rollout_processor_with_retry function to ensure logging works correctly."""

    @pytest.fixture
    def mock_rollout_processor(self):
        """Create a mock rollout processor that returns async tasks."""
        processor = MagicMock()
        processor.cleanup = MagicMock()
        return processor

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with a logger."""
        config = MagicMock(spec=RolloutProcessorConfig)
        config.logger = MagicMock(spec=DatasetLogger)
        config.logger.log = MagicMock()
        config.exception_handler_config = None
        config.kwargs = {}
        return config

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        from datetime import datetime

        row = EvaluationRow(
            messages=[],
            input_metadata=InputMetadata(completion_params={"model": "test-model"}),
            rollout_status=Status.rollout_finished(),
            execution_metadata=ExecutionMetadata(),
            created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
        )
        return [row]

    @pytest.mark.asyncio
    async def test_logger_called_on_successful_execution(self, mock_rollout_processor, mock_config, sample_dataset):
        """Test that the logger is called when execution succeeds."""

        # Create mock tasks that will complete successfully
        async def mock_task():
            from datetime import datetime

            row = EvaluationRow(
                messages=[],
                input_metadata=InputMetadata(completion_params={"model": "test-model"}),
                rollout_status=Status.rollout_finished(),
                execution_metadata=ExecutionMetadata(),
                created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
            )
            return row

        # Mock the processor to return a list of tasks
        mock_rollout_processor.return_value = [asyncio.create_task(mock_task())]

        # Call the function
        results = []
        async for result in rollout_processor_with_retry(mock_rollout_processor, sample_dataset, mock_config):
            results.append(result)

        # Verify that the logger was called for each result
        assert mock_config.logger.log.call_count == 1
        mock_config.logger.log.assert_called_once_with(results[0])

        # Verify cleanup was called
        mock_rollout_processor.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_logger_called_on_failed_execution(self, mock_rollout_processor, mock_config, sample_dataset):
        """Test that the logger is called when execution fails."""

        # Mock the processor to return a task that raises an exception
        async def failing_task():
            raise ValueError("Test error")

        mock_rollout_processor.return_value = [asyncio.create_task(failing_task())]

        # Call the function
        results = []
        async for result in rollout_processor_with_retry(mock_rollout_processor, sample_dataset, mock_config):
            results.append(result)

        # Verify that the logger was called for the failed result
        assert mock_config.logger.log.call_count == 1
        mock_config.logger.log.assert_called_once_with(results[0])

        # Verify the result has an error status
        assert results[0].rollout_status.code == 13  # INTERNAL error code
        assert "Test error" in results[0].rollout_status.message

        # Verify cleanup was called
        mock_rollout_processor.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_logger_called_on_retry_execution(self, mock_rollout_processor, mock_config, sample_dataset):
        """Test that the logger is called when execution succeeds after retry."""
        # Mock the processor to return a task that fails first, then succeeds on retry
        call_count = 0

        async def flaky_task():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection failed")
            else:
                row = EvaluationRow(
                    messages=[],
                    input_metadata={},
                    rollout_status=Status.rollout_finished(),
                    execution_metadata={},
                    created_at="2024-01-01T00:00:00Z",
                )
                return row

        mock_rollout_processor.return_value = [asyncio.create_task(flaky_task())]

        # Call the function - it should handle the retry internally
        results = []
        async for result in rollout_processor_with_retry(mock_rollout_processor, sample_dataset, mock_config):
            results.append(result)

        # Verify that the logger was called for the result
        assert mock_config.logger.log.call_count == 1
        mock_config.logger.log.assert_called_once_with(results[0])

        # Verify cleanup was called
        mock_rollout_processor.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_logger_called_for_multiple_rows(self, mock_rollout_processor, mock_config):
        """Test that the logger is called for each row in a multi-row dataset."""
        # Create a dataset with multiple rows
        from datetime import datetime

        sample_dataset = [
            EvaluationRow(
                messages=[],
                input_metadata=InputMetadata(completion_params={"model": "test-model"}),
                rollout_status=Status.rollout_finished(),
                execution_metadata=ExecutionMetadata(),
                created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
            ),
            EvaluationRow(
                messages=[],
                input_metadata=InputMetadata(completion_params={"model": "test-model"}),
                rollout_status=Status.rollout_finished(),
                execution_metadata=ExecutionMetadata(),
                created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
            ),
        ]

        # Mock the processor to return multiple tasks
        async def mock_task():
            row = EvaluationRow(
                messages=[],
                input_metadata=InputMetadata(completion_params={"model": "test-model"}),
                rollout_status=Status.rollout_finished(),
                execution_metadata=ExecutionMetadata(),
                created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
            )
            return row

        mock_rollout_processor.return_value = [asyncio.create_task(mock_task()), asyncio.create_task(mock_task())]

        # Call the function
        results = []
        async for result in rollout_processor_with_retry(mock_rollout_processor, sample_dataset, mock_config):
            results.append(result)

        # Verify that the logger was called for each result
        assert mock_config.logger.log.call_count == 2
        assert len(results) == 2

        # Verify cleanup was called
        mock_rollout_processor.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_logger_called_even_when_processor_fails_to_initialize(
        self, mock_rollout_processor, mock_config, sample_dataset
    ):
        """Test that cleanup is called even when the processor fails to initialize."""
        # Mock the processor to raise an exception during initialization
        mock_rollout_processor.side_effect = RuntimeError("Processor failed to initialize")

        # Call the function and expect it to raise the exception
        with pytest.raises(RuntimeError, match="Processor failed to initialize"):
            async for result in rollout_processor_with_retry(mock_rollout_processor, sample_dataset, mock_config):
                pass

        # Verify cleanup was called even though the function failed
        mock_rollout_processor.cleanup.assert_called_once()
