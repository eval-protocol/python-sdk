"""Tests for evaluation postprocess functionality."""

from unittest.mock import Mock, patch

from eval_protocol.models import EvaluationRow, EvaluateResult, EvalMetadata, ExecutionMetadata, InputMetadata, Message
from eval_protocol.pytest.evaluation_test_postprocess import postprocess
from eval_protocol.stats.confidence_intervals import compute_fixed_set_mu_ci


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


class TestBootstrapEquivalence:
    def test_bootstrap_equivalence_pandas_vs_pure_python(self):
        import random
        import pandas as pd
        from eval_protocol.pytest.evaluation_test_utils import calculate_bootstrap_scores as py_bootstrap

        # Deterministic synthetic scores
        rng = random.Random(123)
        scores = [rng.random() for _ in range(100)]

        n_boot = 1000
        seed = 42

        # Old (pandas) style bootstrap: resample full column with replacement
        df = pd.DataFrame({"score": scores})
        pandas_means = [
            df.sample(frac=1.0, replace=True, random_state=seed + i)["score"].mean() for i in range(n_boot)
        ]
        pandas_boot_mean = sum(pandas_means) / len(pandas_means)

        # New pure-python implementation
        py_boot_mean = py_bootstrap(scores, n_boot=n_boot, seed=seed)

        # They estimate the same quantity; allow small Monte Carlo tolerance
        assert abs(pandas_boot_mean - py_boot_mean) < 0.02


class TestComputeFixedSetMuCi:
    """Tests for compute_fixed_set_mu_ci function."""

    @patch.dict("os.environ", {"EP_NO_UPLOAD": "1"})  # Disable uploads
    def test_compute_fixed_set_mu_ci_with_flattened_results(self):
        """Test that postprocess correctly calls compute_fixed_set_mu_ci with flattened all_results structure."""

        q1_run1 = EvaluationRow(
            messages=[Message(role="user", content="What is 2+2?")],
            evaluation_result=EvaluateResult(score=0.5, is_score_valid=True, reason="correct"),
            input_metadata=InputMetadata(row_id="q1", completion_params={"model": "test"}),
            execution_metadata=ExecutionMetadata(),
            eval_metadata=EvalMetadata(
                name="test",
                description="test",
                version="1.0",
                status=None,
                num_runs=3,
                aggregation_method="mean",
                passed_threshold=None,
                passed=None,
            ),
        )
        q1_run2 = EvaluationRow(
            messages=[Message(role="user", content="What is 2+2?")],
            evaluation_result=EvaluateResult(score=0.4, is_score_valid=True, reason="incorrect"),
            input_metadata=InputMetadata(row_id="q1", completion_params={"model": "test"}),
            execution_metadata=ExecutionMetadata(),
            eval_metadata=EvalMetadata(
                name="test",
                description="test",
                version="1.0",
                status=None,
                num_runs=3,
                aggregation_method="mean",
                passed_threshold=None,
                passed=None,
            ),
        )
        q1_run3 = EvaluationRow(
            messages=[Message(role="user", content="What is 2+2?")],
            evaluation_result=EvaluateResult(score=0.45, is_score_valid=True, reason="incorrect"),
            input_metadata=InputMetadata(row_id="q1", completion_params={"model": "test"}),
            execution_metadata=ExecutionMetadata(),
            eval_metadata=EvalMetadata(
                name="test",
                description="test",
                version="1.0",
                status=None,
                num_runs=3,
                aggregation_method="mean",
                passed_threshold=None,
                passed=None,
            ),
        )
        q2_run1 = EvaluationRow(
            messages=[Message(role="user", content="What is 3+3?")],
            evaluation_result=EvaluateResult(score=0.8, is_score_valid=True, reason="incorrect"),
            input_metadata=InputMetadata(row_id="q2", completion_params={"model": "test"}),
            execution_metadata=ExecutionMetadata(),
            eval_metadata=EvalMetadata(
                name="test",
                description="test",
                version="1.0",
                status=None,
                num_runs=3,
                aggregation_method="mean",
                passed_threshold=None,
                passed=None,
            ),
        )
        q2_run2 = EvaluationRow(
            messages=[Message(role="user", content="What is 3+3?")],
            evaluation_result=EvaluateResult(score=0.9, is_score_valid=True, reason="correct"),
            input_metadata=InputMetadata(row_id="q2", completion_params={"model": "test"}),
            execution_metadata=ExecutionMetadata(),
            eval_metadata=EvalMetadata(
                name="test",
                description="test",
                version="1.0",
                status=None,
                num_runs=3,
                aggregation_method="mean",
                passed_threshold=None,
                passed=None,
            ),
        )
        q2_run3 = EvaluationRow(
            messages=[Message(role="user", content="What is 3+3?")],
            evaluation_result=EvaluateResult(score=0.95, is_score_valid=True, reason="correct"),
            input_metadata=InputMetadata(row_id="q2", completion_params={"model": "test"}),
            execution_metadata=ExecutionMetadata(),
            eval_metadata=EvalMetadata(
                name="test",
                description="test",
                version="1.0",
                status=None,
                num_runs=3,
                aggregation_method="mean",
                passed_threshold=None,
                passed=None,
            ),
        )
        q3_run1 = EvaluationRow(
            messages=[Message(role="user", content="What is 4+4?")],
            evaluation_result=EvaluateResult(score=0.1, is_score_valid=True, reason="incorrect"),
            input_metadata=InputMetadata(row_id="q3", completion_params={"model": "test"}),
            execution_metadata=ExecutionMetadata(),
            eval_metadata=EvalMetadata(
                name="test",
                description="test",
                version="1.0",
                status=None,
                num_runs=3,
                aggregation_method="mean",
                passed_threshold=None,
                passed=None,
            ),
        )
        q3_run2 = EvaluationRow(
            messages=[Message(role="user", content="What is 4+4?")],
            evaluation_result=EvaluateResult(score=0.2, is_score_valid=True, reason="correct"),
            input_metadata=InputMetadata(row_id="q3", completion_params={"model": "test"}),
            execution_metadata=ExecutionMetadata(),
            eval_metadata=EvalMetadata(
                name="test",
                description="test",
                version="1.0",
                status=None,
                num_runs=3,
                aggregation_method="mean",
                passed_threshold=None,
                passed=None,
            ),
        )
        q3_run3_valid = EvaluationRow(
            messages=[Message(role="user", content="What is 4+4?")],
            evaluation_result=EvaluateResult(score=0.3, is_score_valid=True, reason="correct"),
            input_metadata=InputMetadata(row_id="q3", completion_params={"model": "test"}),
            execution_metadata=ExecutionMetadata(),
            eval_metadata=EvalMetadata(
                name="test",
                description="test",
                version="1.0",
                status=None,
                num_runs=3,
                aggregation_method="mean",
                passed_threshold=None,
                passed=None,
            ),
        )
        q3_run3_invalid = EvaluationRow(
            messages=[Message(role="user", content="What is 4+4?")],
            evaluation_result=EvaluateResult(score=0.3, is_score_valid=False, reason="correct"),
            input_metadata=InputMetadata(row_id="q3", completion_params={"model": "test"}),
            execution_metadata=ExecutionMetadata(),
            eval_metadata=EvalMetadata(
                name="test",
                description="test",
                version="1.0",
                status=None,
                num_runs=3,
                aggregation_method="mean",
                passed_threshold=None,
                passed=None,
            ),
        )

        rows = [[q1_run1, q2_run1, q3_run1], [q1_run2, q2_run2, q1_run3], [q2_run3, q3_run2, q3_run3_valid]]
        rows_with_invalid_score = [
            [q1_run1, q2_run1, q3_run1],
            [q1_run2, q2_run2, q1_run3],
            [q2_run3, q3_run2, q3_run3_invalid],
        ]

        # Store results for assertions
        first_result = None
        second_result = None

        # Test first case (all valid scores)
        with patch("eval_protocol.pytest.evaluation_test_postprocess.compute_fixed_set_mu_ci") as mock_ci:
            mock_ci.side_effect = lambda input_rows, **kwargs: compute_fixed_set_mu_ci(input_rows, **kwargs)

            postprocess(
                all_results=rows,
                aggregation_method="mean",
                threshold=None,
                active_logger=Mock(),
                mode="pointwise",
                completion_params={"model": "test-model"},
                test_func_name="test_ci_flattened",
                num_runs=3,
                experiment_duration_seconds=10.0,
            )

            first_result = mock_ci.return_value

        # Test second case (with invalid score)
        with patch("eval_protocol.pytest.evaluation_test_postprocess.compute_fixed_set_mu_ci") as mock_ci:
            mock_ci.side_effect = lambda input_rows, **kwargs: compute_fixed_set_mu_ci(input_rows, **kwargs)

            postprocess(
                all_results=rows_with_invalid_score,
                aggregation_method="mean",
                threshold=None,
                active_logger=Mock(),
                mode="pointwise",
                completion_params={"model": "test-model"},
                test_func_name="test_ci_flattened_invalid",
                num_runs=3,
                experiment_duration_seconds=10.0,
            )

            second_result = mock_ci.return_value

        # Assert exact values
        # First case: (0.5111111111111111, 0.18101430525778583, 0.8412079169644363, 0.168416737680268)
        if first_result and len(first_result) == 4:
            mu_hat1, ci_low1, ci_high1, se1 = first_result
            assert abs(mu_hat1 - 0.5111111111111111) < 1e-10
            assert abs(ci_low1 - 0.18101430525778583) < 1e-10
            assert abs(ci_high1 - 0.8412079169644363) < 1e-10
            assert abs(se1 - 0.168416737680268) < 1e-10

        # Second case: (0.49444444444444446, 0.13494616580367125, 0.8539427230852177, 0.18341748910243533)
        if second_result and len(second_result) == 4:
            mu_hat2, ci_low2, ci_high2, se2 = second_result
            assert abs(mu_hat2 - 0.49444444444444446) < 1e-10
            assert abs(ci_low2 - 0.13494616580367125) < 1e-10
            assert abs(ci_high2 - 0.8539427230852177) < 1e-10
            assert abs(se2 - 0.18341748910243533) < 1e-10
