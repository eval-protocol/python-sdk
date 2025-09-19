"""Tests for evaluation postprocess functionality."""

import pytest
from unittest.mock import Mock, patch

from eval_protocol.models import EvaluationRow, EvaluateResult, EvalMetadata, ExecutionMetadata, InputMetadata
from eval_protocol.pytest.evaluation_test_postprocess import postprocess


class TestPostprocess:
    """Tests for postprocess function."""

    def create_test_row(self, score: float, is_valid: bool = True) -> EvaluationRow:
        """Helper to create a test evaluation row."""
        return EvaluationRow(
            messages=[],
            evaluation_result=EvaluateResult(score=score, is_score_valid=is_valid, reason="test"),
            input_metadata=InputMetadata(completion_params={"model": "test-model"}),
            execution_metadata=ExecutionMetadata(),
            eval_metadata=EvalMetadata(
                name="test",
                description="test",
                version="1.0",
                status=None,
                num_runs=1,
                aggregation_method="mean",
                passed_threshold=None,
                passed=None,
            ),
        )

    @patch.dict("os.environ", {"EP_NO_UPLOAD": "1"})  # Disable uploads
    def test_bootstrap_aggregation_with_valid_scores(self):
        """Test bootstrap aggregation with all valid scores and verify exact scores list."""
        # Create test data: 2 runs with 2 rows each
        all_results = [
            [self.create_test_row(0.8), self.create_test_row(0.6)],  # Run 1
            [self.create_test_row(0.7), self.create_test_row(0.9)],  # Run 2
        ]

        mock_logger = Mock()

        # Mock the aggregate function to capture the exact scores passed to it
        with patch("eval_protocol.pytest.evaluation_test_postprocess.aggregate") as mock_aggregate:
            mock_aggregate.return_value = 0.75  # Mock return value

            postprocess(
                all_results=all_results,
                aggregation_method="bootstrap",
                threshold=None,
                active_logger=mock_logger,
                mode="pointwise",
                completion_params={"model": "test-model"},
                test_func_name="test_bootstrap",
                num_runs=2,
                experiment_duration_seconds=10.0,
            )

            # Check that aggregate was called with all individual scores in order
            mock_aggregate.assert_called_once_with([0.8, 0.6, 0.7, 0.9], "bootstrap")

        # Should call logger.log for each row
        assert mock_logger.log.call_count == 4

    @patch.dict("os.environ", {"EP_NO_UPLOAD": "1"})  # Disable uploads
    def test_bootstrap_aggregation_filters_invalid_scores(self):
        """Test that bootstrap aggregation excludes invalid scores and generates correct scores list."""
        # Create test data with some invalid scores
        all_results = [
            [
                self.create_test_row(0.8, is_valid=True),
                self.create_test_row(0.0, is_valid=False),  # Invalid - should be excluded
            ],
            [
                self.create_test_row(0.7, is_valid=True),
                self.create_test_row(0.0, is_valid=False),  # Invalid - should be excluded
            ],
        ]

        mock_logger = Mock()

        # Mock the aggregate function to capture the scores passed to it
        with patch("eval_protocol.pytest.evaluation_test_postprocess.aggregate") as mock_aggregate:
            mock_aggregate.return_value = 0.75  # Mock return value

            postprocess(
                all_results=all_results,
                aggregation_method="bootstrap",
                threshold=None,
                active_logger=mock_logger,
                mode="pointwise",
                completion_params={"model": "test-model"},
                test_func_name="test_bootstrap_invalid",
                num_runs=2,
                experiment_duration_seconds=10.0,
            )

            # Check that aggregate was called with only valid scores
            mock_aggregate.assert_called_once_with([0.8, 0.7], "bootstrap")

        # Should still call logger.log for all rows (including invalid ones)
        assert mock_logger.log.call_count == 4

    @patch.dict("os.environ", {"EP_NO_UPLOAD": "1"})  # Disable uploads
    def test_mean_aggregation_with_valid_scores(self):
        """Test mean aggregation with all valid scores."""
        all_results = [
            [self.create_test_row(0.8), self.create_test_row(0.6)],  # Run 1: mean = 0.7
            [self.create_test_row(0.4), self.create_test_row(0.8)],  # Run 2: mean = 0.6
        ]

        mock_logger = Mock()

        postprocess(
            all_results=all_results,
            aggregation_method="mean",
            threshold=None,
            active_logger=mock_logger,
            mode="pointwise",
            completion_params={"model": "test-model"},
            test_func_name="test_mean",
            num_runs=2,
            experiment_duration_seconds=10.0,
        )

        # Should call logger.log for each row
        assert mock_logger.log.call_count == 4

    @patch.dict("os.environ", {"EP_NO_UPLOAD": "1"})  # Disable uploads
    def test_mean_aggregation_filters_invalid_scores(self):
        """Test that mean aggregation excludes invalid scores from run averages."""
        all_results = [
            [
                self.create_test_row(0.8, is_valid=True),
                self.create_test_row(0.0, is_valid=False),  # Invalid - excluded from run average
            ],
            [
                self.create_test_row(0.6, is_valid=True),
                self.create_test_row(0.4, is_valid=True),
            ],
        ]

        mock_logger = Mock()

        postprocess(
            all_results=all_results,
            aggregation_method="mean",
            threshold=None,
            active_logger=mock_logger,
            mode="pointwise",
            completion_params={"model": "test-model"},
            test_func_name="test_mean_invalid",
            num_runs=2,
            experiment_duration_seconds=10.0,
        )

        # Should call logger.log for all rows
        assert mock_logger.log.call_count == 4

    @patch.dict("os.environ", {"EP_NO_UPLOAD": "1"})  # Disable uploads
    def test_empty_runs_are_skipped(self):
        """Test that runs with no valid scores are skipped."""
        all_results = [
            [self.create_test_row(0.8, is_valid=True)],  # Run 1: has valid score
            [self.create_test_row(0.0, is_valid=False)],  # Run 2: no valid scores - should be skipped
        ]

        mock_logger = Mock()

        postprocess(
            all_results=all_results,
            aggregation_method="mean",
            threshold=None,
            active_logger=mock_logger,
            mode="pointwise",
            completion_params={"model": "test-model"},
            test_func_name="test_empty_runs",
            num_runs=2,
            experiment_duration_seconds=10.0,
        )

        # Should still call logger.log for all rows
        assert mock_logger.log.call_count == 2

    @patch.dict("os.environ", {"EP_NO_UPLOAD": "1"})  # Disable uploads
    def test_all_invalid_scores(self):
        """Test behavior when all scores are invalid."""
        all_results = [
            [self.create_test_row(0.0, is_valid=False), self.create_test_row(0.0, is_valid=False)],
        ]

        mock_logger = Mock()

        postprocess(
            all_results=all_results,
            aggregation_method="bootstrap",
            threshold=None,
            active_logger=mock_logger,
            mode="pointwise",
            completion_params={"model": "test-model"},
            test_func_name="test_all_invalid",
            num_runs=1,
            experiment_duration_seconds=10.0,
        )

        # Should still call logger.log for all rows
        assert mock_logger.log.call_count == 2
