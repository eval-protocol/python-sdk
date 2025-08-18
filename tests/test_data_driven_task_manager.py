"""
Tests for data-driven evaluation functionality in TaskManager.

This module specifically tests the enhanced TaskManager capabilities for:
- Loading and processing JSONL datasets
- Executing multiple rollouts per sample
- Aggregating results across data-driven evaluations
- Error handling and edge cases
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eval_protocol.agent.orchestrator import Orchestrator
from eval_protocol.agent.task_manager import TaskManager
from eval_protocol.models import TaskDefinitionModel


class TestDataDrivenTaskManager:
    """Tests for data-driven evaluation in TaskManager."""

    def setup_method(self):
        """Set up a TaskManager instance for each test."""
        self.task_manager = TaskManager()

    def create_test_dataset(self, samples: list) -> str:
        """Helper to create a temporary JSONL dataset file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for sample in samples:
                json.dump(sample, f)
                f.write("\n")
            return f.name

    def create_test_task_definition(
        self, dataset_path: str = None, num_rollouts_per_sample: int = 1
    ) -> TaskDefinitionModel:
        """Helper to create a test task definition."""
        task_def_dict = {
            "name": "test_data_driven_task",
            "description": "Test task for data-driven evaluation",
            "resource_type": "http_rollout",
            "base_resource_config": {"base_url": "http://localhost:8080"},
            "reward_function_path": "test.reward_function",
            "messages": [{"role": "user", "content": "Test message"}],
        }

        if dataset_path:
            task_def_dict["dataset_path"] = dataset_path
            task_def_dict["num_rollouts_per_sample"] = num_rollouts_per_sample
        else:
            task_def_dict["num_rollouts"] = 3

        return TaskDefinitionModel(**task_def_dict)


class TestDatasetLoading(TestDataDrivenTaskManager):
    """Tests for dataset loading functionality."""

    def test_load_simple_dataset(self):
        """Test loading a simple JSONL dataset."""
        samples = [
            {"id": "sample1", "seed": 42, "difficulty": "easy"},
            {"id": "sample2", "seed": 123, "difficulty": "medium"},
            {"id": "sample3", "seed": 999, "difficulty": "hard"},
        ]

        dataset_path = self.create_test_dataset(samples)

        try:
            loaded_samples = self.task_manager._load_dataset_samples(dataset_path)
            assert len(loaded_samples) == 3
            assert loaded_samples == samples
        finally:
            Path(dataset_path).unlink()

    def test_load_dataset_with_empty_lines(self):
        """Test loading dataset that contains empty lines."""
        # Create dataset with empty lines
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "sample1", "seed": 42}\n')
            f.write("\n")  # Empty line
            f.write("  \n")  # Whitespace only line
            f.write('{"id": "sample2", "seed": 123}\n')
            dataset_path = f.name

        try:
            loaded_samples = self.task_manager._load_dataset_samples(dataset_path)
            assert len(loaded_samples) == 2
            assert loaded_samples[0]["id"] == "sample1"
            assert loaded_samples[1]["id"] == "sample2"
        finally:
            Path(dataset_path).unlink()

    def test_load_dataset_with_complex_objects(self):
        """Test loading dataset with complex nested objects."""
        samples = [
            {
                "id": "complex_sample1",
                "seed": 42,
                "config": {
                    "map_size": "4x4",
                    "slippery": True,
                    "holes": [[1, 1], [2, 3]],
                },
                "metadata": {"author": "test", "version": "1.0"},
            },
            {
                "id": "complex_sample2",
                "seed": 123,
                "config": {"map_size": "8x8", "slippery": False, "holes": []},
                "metadata": {"author": "test", "version": "1.1"},
            },
        ]

        dataset_path = self.create_test_dataset(samples)

        try:
            loaded_samples = self.task_manager._load_dataset_samples(dataset_path)
            assert len(loaded_samples) == 2
            assert loaded_samples == samples

            # Verify complex objects are preserved
            assert loaded_samples[0]["config"]["holes"] == [[1, 1], [2, 3]]
            assert loaded_samples[1]["metadata"]["version"] == "1.1"
        finally:
            Path(dataset_path).unlink()

    def test_load_dataset_error_handling(self):
        """Test error handling for various dataset loading scenarios."""
        # Test nonexistent file
        samples = self.task_manager._load_dataset_samples("nonexistent_file.jsonl")
        assert samples == []

        # Test malformed JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"valid": "json"}\n')
            f.write("{invalid json}\n")
            f.write('{"another": "valid"}\n')
            malformed_path = f.name

        try:
            loaded_samples = self.task_manager._load_dataset_samples(malformed_path)
            # Should skip malformed line and load valid ones
            assert len(loaded_samples) == 2
            assert loaded_samples[0]["valid"] == "json"
            assert loaded_samples[1]["another"] == "valid"
        finally:
            Path(malformed_path).unlink()


@pytest.mark.asyncio
class TestDataDrivenExecution(TestDataDrivenTaskManager):
    """Tests for data-driven execution logic."""

    async def test_execute_data_driven_rollouts_basic(self):
        """Test basic data-driven rollout execution."""
        samples = [{"id": "sample1", "seed": 42}, {"id": "sample2", "seed": 123}]

        # Mock the orchestrator execution
        with (
            patch.object(self.task_manager, "_start_resource_server", return_value=8080),
            patch.object(self.task_manager, "_stop_resource_server"),
            patch("eval_protocol.agent.task_manager.Orchestrator") as mock_orchestrator_class,
        ):
            # Set up mock orchestrator
            mock_orchestrator = AsyncMock()
            mock_orchestrator.setup_base_resource = AsyncMock()
            mock_orchestrator.execute_task_poc = AsyncMock(
                side_effect=[
                    {"score": 1.0, "reason": "Success"},
                    {"score": 0.0, "reason": "Failed"},
                ]
            )
            mock_orchestrator.base_resource = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Create task definition
            task_def = self.create_test_task_definition(num_rollouts_per_sample=1)
            self.task_manager.register_task("test_task", task_def)

            # Execute data-driven rollouts
            results = await self.task_manager._execute_data_driven_rollouts(
                "test_task", samples, rollouts_per_sample=1, max_concurrency=2
            )

            assert len(results) == 2
            assert results[0]["score"] == 1.0
            assert results[1]["score"] == 0.0
            assert results[0]["sample_data"] == samples[0]
            assert results[1]["sample_data"] == samples[1]

    async def test_execute_multiple_rollouts_per_sample(self):
        """Test executing multiple rollouts per sample."""
        samples = [{"id": "sample1", "seed": 42}]
        rollouts_per_sample = 3

        with (
            patch.object(self.task_manager, "_start_resource_server", return_value=8080),
            patch.object(self.task_manager, "_stop_resource_server"),
            patch("eval_protocol.agent.task_manager.Orchestrator") as mock_orchestrator_class,
        ):
            # Set up mock orchestrator to return different scores for each rollout
            mock_orchestrator = AsyncMock()
            mock_orchestrator.setup_base_resource = AsyncMock()
            mock_orchestrator.execute_task_poc = AsyncMock(
                side_effect=[
                    {"score": 1.0, "reason": "Success"},
                    {"score": 0.5, "reason": "Partial"},
                    {"score": 0.0, "reason": "Failed"},
                ]
            )
            mock_orchestrator.base_resource = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Create task definition
            task_def = self.create_test_task_definition(num_rollouts_per_sample=rollouts_per_sample)
            self.task_manager.register_task("test_task", task_def)

            # Execute data-driven rollouts
            results = await self.task_manager._execute_data_driven_rollouts(
                "test_task",
                samples,
                rollouts_per_sample=rollouts_per_sample,
                max_concurrency=2,
            )

            # Should have 3 results (1 sample × 3 rollouts)
            assert len(results) == 3

            # All results should have same sample data but different rollout indices
            for i, result in enumerate(results):
                assert result["sample_data"] == samples[0]
                assert result["rollout_index"] == i
                assert result["sample_index"] == 0

    async def test_execute_data_driven_with_failures(self):
        """Test handling of failures during data-driven execution."""
        samples = [{"id": "sample1", "seed": 42}, {"id": "sample2", "seed": 123}]

        with (
            patch.object(self.task_manager, "_start_resource_server", return_value=8080),
            patch.object(self.task_manager, "_stop_resource_server"),
            patch("eval_protocol.agent.task_manager.Orchestrator") as mock_orchestrator_class,
        ):
            # Set up mock orchestrator with one success and one failure
            mock_orchestrator = AsyncMock()
            mock_orchestrator.setup_base_resource = AsyncMock()
            mock_orchestrator.execute_task_poc = AsyncMock(
                side_effect=[
                    {"score": 1.0, "reason": "Success"},
                    Exception("Execution failed"),
                ]
            )
            mock_orchestrator.base_resource = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Create task definition
            task_def = self.create_test_task_definition(num_rollouts_per_sample=1)
            self.task_manager.register_task("test_task", task_def)

            # Execute data-driven rollouts
            results = await self.task_manager._execute_data_driven_rollouts(
                "test_task", samples, rollouts_per_sample=1, max_concurrency=2
            )

            assert len(results) == 2

            # First should be successful
            assert results[0]["score"] == 1.0
            assert "error" not in results[0]

            # Second should have error
            assert "error" in results[1]
            assert "Execution failed" in results[1]["error"]

    async def test_concurrency_limiting(self):
        """Test that concurrency is properly limited."""
        samples = [{"id": f"sample{i}", "seed": i} for i in range(10)]
        max_concurrency = 3

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent_observed = 0

        async def mock_execute(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent_observed
            concurrent_count += 1
            max_concurrent_observed = max(max_concurrent_observed, concurrent_count)

            # Simulate some async work
            await asyncio.sleep(0.1)

            concurrent_count -= 1
            return {"score": 1.0, "reason": "Success"}

        with (
            patch.object(self.task_manager, "_start_resource_server", return_value=8080),
            patch.object(self.task_manager, "_stop_resource_server"),
            patch("eval_protocol.agent.task_manager.Orchestrator") as mock_orchestrator_class,
        ):
            mock_orchestrator = AsyncMock()
            mock_orchestrator.setup_base_resource = AsyncMock()
            mock_orchestrator.execute_task_poc = AsyncMock(side_effect=mock_execute)
            mock_orchestrator.base_resource = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Create task definition
            task_def = self.create_test_task_definition(num_rollouts_per_sample=1)
            self.task_manager.register_task("test_task", task_def)

            # Execute with concurrency limit
            results = await self.task_manager._execute_data_driven_rollouts(
                "test_task",
                samples,
                rollouts_per_sample=1,
                max_concurrency=max_concurrency,
            )

            assert len(results) == 10
            # Should not exceed max concurrency
            assert max_concurrent_observed <= max_concurrency


@pytest.mark.asyncio
class TestTaskExecutionFlow(TestDataDrivenTaskManager):
    """Tests for complete task execution flow with data-driven evaluation."""

    async def test_execute_tasks_data_driven_vs_traditional(self):
        """Test that TaskManager correctly chooses between data-driven and traditional execution."""
        # Create dataset
        samples = [{"id": "sample1", "seed": 42}]
        dataset_path = self.create_test_dataset(samples)

        try:
            # Create data-driven task
            data_driven_task = self.create_test_task_definition(dataset_path=dataset_path, num_rollouts_per_sample=2)

            # Create traditional task
            traditional_task = self.create_test_task_definition()  # No dataset_path

            self.task_manager.register_task("data_driven", data_driven_task)
            self.task_manager.register_task("traditional", traditional_task)

            with (
                patch.object(self.task_manager, "_execute_data_driven_rollouts") as mock_data_driven,
                patch.object(self.task_manager, "_execute_batch_rollouts") as mock_traditional,
            ):
                mock_data_driven.return_value = [{"score": 1.0}]
                mock_traditional.return_value = [{"score": 0.5}]

                # Execute both tasks
                results = await self.task_manager.execute_tasks(["data_driven", "traditional"], max_concurrency=2)

                # Verify correct execution methods were called
                mock_data_driven.assert_called_once()
                mock_traditional.assert_called_once()

                # Verify results
                assert len(results) == 2
                assert "data_driven" in results
                assert "traditional" in results

        finally:
            Path(dataset_path).unlink()

    async def test_data_driven_evaluation_with_empty_dataset(self):
        """Test handling of empty dataset."""
        # Create empty dataset
        empty_dataset_path = self.create_test_dataset([])

        try:
            task_def = self.create_test_task_definition(dataset_path=empty_dataset_path, num_rollouts_per_sample=1)
            self.task_manager.register_task("empty_dataset_task", task_def)

            results = await self.task_manager.execute_tasks(["empty_dataset_task"])

            # Should handle empty dataset gracefully
            assert "empty_dataset_task" in results
            assert "error" in results["empty_dataset_task"]
            assert "empty" in results["empty_dataset_task"]["error"].lower()

        finally:
            Path(empty_dataset_path).unlink()

    async def test_data_driven_evaluation_file_not_found(self):
        """Test handling of missing dataset file."""
        task_def = self.create_test_task_definition(
            dataset_path="nonexistent_dataset.jsonl", num_rollouts_per_sample=1
        )
        self.task_manager.register_task("missing_file_task", task_def)

        results = await self.task_manager.execute_tasks(["missing_file_task"])

        # Should handle missing file gracefully
        assert "missing_file_task" in results
        assert "error" in results["missing_file_task"]


class TestResultAggregation(TestDataDrivenTaskManager):
    """Tests for result aggregation in data-driven evaluation."""

    def test_aggregate_data_driven_results(self):
        """Test aggregation of results from data-driven evaluation."""
        # Simulate results from multiple samples and rollouts
        rollout_results = [
            # Sample 0, rollouts 0-2
            {
                "score": 1.0,
                "sample_index": 0,
                "rollout_index": 0,
                "sample_data": {"seed": 42},
            },
            {
                "score": 0.5,
                "sample_index": 0,
                "rollout_index": 1,
                "sample_data": {"seed": 42},
            },
            {
                "score": 0.0,
                "sample_index": 0,
                "rollout_index": 2,
                "sample_data": {"seed": 42},
            },
            # Sample 1, rollouts 0-1
            {
                "score": 1.0,
                "sample_index": 1,
                "rollout_index": 0,
                "sample_data": {"seed": 123},
            },
            {
                "score": 1.0,
                "sample_index": 1,
                "rollout_index": 1,
                "sample_data": {"seed": 123},
            },
        ]

        aggregated = self.task_manager._aggregate_results(rollout_results)

        # Check basic aggregation
        assert aggregated["total_rollouts"] == 5
        assert aggregated["successful_rollouts"] == 5  # No failed rollouts
        assert aggregated["success_rate"] == 1.0

        # Check score statistics
        expected_avg_score = (1.0 + 0.5 + 0.0 + 1.0 + 1.0) / 5  # 0.7
        assert abs(aggregated["average_score"] - expected_avg_score) < 0.001

        # Check detailed results are preserved
        assert "detailed_results" in aggregated
        assert len(aggregated["detailed_results"]) == 5

    def test_aggregate_results_with_failures(self):
        """Test aggregation when some rollouts failed."""
        rollout_results = [
            {"score": 1.0, "sample_index": 0, "rollout_index": 0},
            {"error": "Failed", "sample_index": 0, "rollout_index": 1},
            {"score": 0.5, "sample_index": 1, "rollout_index": 0},
            {"error": "Another failure", "sample_index": 1, "rollout_index": 1},
        ]

        aggregated = self.task_manager._aggregate_results(rollout_results)

        assert aggregated["total_rollouts"] == 4
        assert aggregated["successful_rollouts"] == 2
        assert aggregated["failed_rollouts"] == 2
        assert aggregated["success_rate"] == 0.5

        # Average should only consider successful rollouts
        expected_avg_score = (1.0 + 0.5) / 2  # 0.75
        assert abs(aggregated["average_score"] - expected_avg_score) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
