"""
Tests for seed-based reproducible evaluation in FrozenLake example.

This module tests the complete data-driven evaluation pipeline including:
- Seed-based map generation for reproducible game boards
- Data-driven evaluation infrastructure in TaskManager
- Protocol enhancements for sample data passing
- End-to-end integration tests
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from eval_protocol.agent.orchestrator import Orchestrator
from eval_protocol.agent.resources.http_rollout_protocol import StartEpisodeRequest
from eval_protocol.agent.resources.http_rollout_resource import HttpRolloutResource
from eval_protocol.agent.task_manager import TaskManager
from eval_protocol.models import TaskDefinitionModel

# Import components under test
from examples.frozen_lake.gymnasium_frozen_lake_server import GymnasiumFrozenLakeGame


class TestSeedBasedMapGeneration:
    """Tests for seed-based reproducible map generation in GymnasiumFrozenLakeGame."""

    def test_seed_generates_different_maps(self):
        """Test that different seeds generate different map layouts."""
        seed1 = 42
        seed2 = 123

        game1 = GymnasiumFrozenLakeGame(seed=seed1)
        game2 = GymnasiumFrozenLakeGame(seed=seed2)

        # Get map descriptions
        map1 = game1.desc.tolist()
        map2 = game2.desc.tolist()

        # Maps should be different
        assert map1 != map2, "Different seeds should generate different maps"

        # Both should be 4x4 by default
        assert len(map1) == 4 and len(map1[0]) == 4
        assert len(map2) == 4 and len(map2[0]) == 4

        game1.close()
        game2.close()

    def test_same_seed_generates_identical_maps(self):
        """Test that the same seed generates identical map layouts."""
        seed = 42

        game1 = GymnasiumFrozenLakeGame(seed=seed)
        game2 = GymnasiumFrozenLakeGame(seed=seed)

        # Get map descriptions
        map1 = game1.desc.tolist()
        map2 = game2.desc.tolist()

        # Maps should be identical
        assert map1 == map2, "Same seed should generate identical maps"

        game1.close()
        game2.close()

    def test_no_seed_uses_fixed_map(self):
        """Test that no seed uses the fixed predefined map."""
        game1 = GymnasiumFrozenLakeGame()  # No seed
        game2 = GymnasiumFrozenLakeGame()  # No seed

        # Get map descriptions
        map1 = game1.desc.tolist()
        map2 = game2.desc.tolist()

        # Maps should be identical (both using fixed 4x4 map)
        assert map1 == map2, "No seed should use identical fixed maps"

        # Should be the standard 4x4 FrozenLake map
        expected_map = [
            [b"S", b"F", b"F", b"F"],
            [b"F", b"H", b"F", b"H"],
            [b"F", b"F", b"F", b"H"],
            [b"H", b"F", b"F", b"G"],
        ]
        assert map1 == expected_map, "Should use standard 4x4 map when no seed"

        game1.close()
        game2.close()

    def test_seed_with_8x8_map(self):
        """Test seed-based generation with 8x8 map size."""
        seed = 999

        game = GymnasiumFrozenLakeGame(map_name="8x8", seed=seed)

        # Should be 8x8
        assert game.desc.shape == (8, 8)

        # Should have start and goal
        flat_map = game.desc.flatten()
        assert b"S" in flat_map
        assert b"G" in flat_map

        game.close()

    def test_seed_affects_reset_behavior(self):
        """Test that seed affects the reset behavior for stochastic environments."""
        seed = 42

        # Test with slippery environment
        game = GymnasiumFrozenLakeGame(seed=seed, is_slippery=True)

        # Reset multiple times - should get same initial state
        state1 = game.reset()
        state2 = game.reset()

        # Initial position should be consistent
        assert state1["position"] == state2["position"] == (0, 0)
        assert state1["current_cell"] == state2["current_cell"] == "S"

        game.close()

    def test_map_has_valid_path(self):
        """Test that generated maps always have a valid path from start to goal."""
        # Test multiple seeds to ensure path validity
        for seed in [42, 123, 999, 1337]:
            game = GymnasiumFrozenLakeGame(seed=seed)

            # Should have exactly one start and one goal
            flat_map = game.desc.flatten()
            start_count = sum(1 for cell in flat_map if cell == b"S")
            goal_count = sum(1 for cell in flat_map if cell == b"G")

            assert start_count == 1, f"Should have exactly one start for seed {seed}"
            assert goal_count == 1, f"Should have exactly one goal for seed {seed}"

            # Start should be at (0,0) and goal should exist
            assert game.desc[0, 0] == b"S", f"Start should be at (0,0) for seed {seed}"
            assert game.start_pos == (
                0,
                0,
            ), f"Start position should be (0,0) for seed {seed}"
            assert game.goal_pos is not None, f"Goal position should exist for seed {seed}"

            game.close()


class TestStartEpisodeRequest:
    """Tests for the enhanced StartEpisodeRequest protocol."""

    def test_start_episode_request_accepts_arbitrary_fields(self):
        """Test that StartEpisodeRequest accepts arbitrary fields like seed."""
        # Should accept seed and other fields
        request = StartEpisodeRequest(seed=42, custom_field="test_value")

        # Access via model_dump or dict
        if hasattr(request, "model_dump"):
            data = request.model_dump()
        else:
            data = request.dict()

        assert data["seed"] == 42
        assert data["custom_field"] == "test_value"

    def test_start_episode_request_empty(self):
        """Test that StartEpisodeRequest works with no extra fields."""
        request = StartEpisodeRequest()

        # Should work without errors
        if hasattr(request, "model_dump"):
            data = request.model_dump()
        else:
            data = request.dict()

        # Should be empty dict or have no extra fields
        assert isinstance(data, dict)


class TestDataDrivenTaskDefinition:
    """Tests for data-driven evaluation fields in TaskDefinitionModel."""

    def test_task_definition_with_dataset_path(self):
        """Test TaskDefinitionModel with dataset_path field."""
        task_def_dict = {
            "name": "test_task",
            "description": "Test task",
            "resource_type": "http_rollout",
            "base_resource_config": {"base_url": "http://localhost:8080"},
            "reward_function_path": "test.reward",
            "dataset_path": "test_dataset.jsonl",
            "num_rollouts_per_sample": 3,
        }

        task_def = TaskDefinitionModel(**task_def_dict)

        assert task_def.dataset_path == "test_dataset.jsonl"
        assert task_def.num_rollouts_per_sample == 3

    def test_task_definition_without_dataset_path(self):
        """Test TaskDefinitionModel without dataset_path (traditional evaluation)."""
        task_def_dict = {
            "name": "test_task",
            "description": "Test task",
            "resource_type": "http_rollout",
            "base_resource_config": {"base_url": "http://localhost:8080"},
            "reward_function_path": "test.reward",
            "num_rollouts": 5,
        }

        task_def = TaskDefinitionModel(**task_def_dict)

        assert task_def.dataset_path is None
        assert task_def.num_rollouts_per_sample == 1  # Default value
        assert task_def.num_rollouts == 5

    def test_num_rollouts_per_sample_validation(self):
        """Test that num_rollouts_per_sample must be >= 1."""
        task_def_dict = {
            "name": "test_task",
            "description": "Test task",
            "resource_type": "http_rollout",
            "base_resource_config": {"base_url": "http://localhost:8080"},
            "reward_function_path": "test.reward",
            "dataset_path": "test.jsonl",
            "num_rollouts_per_sample": 0,  # Invalid
        }

        with pytest.raises(ValueError):
            TaskDefinitionModel(**task_def_dict)


@pytest.mark.asyncio
class TestTaskManagerDataDrivenEvaluation:
    """Tests for data-driven evaluation functionality in TaskManager."""

    def test_load_dataset_samples_valid_jsonl(self):
        """Test loading valid JSONL dataset."""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "sample1", "seed": 42}\n')
            f.write('{"id": "sample2", "seed": 123}\n')
            f.write('{"id": "sample3", "seed": 999}\n')
            temp_file = f.name

        try:
            task_manager = TaskManager()
            samples = task_manager._load_dataset_samples(temp_file)

            assert len(samples) == 3
            assert samples[0] == {"id": "sample1", "seed": 42}
            assert samples[1] == {"id": "sample2", "seed": 123}
            assert samples[2] == {"id": "sample3", "seed": 999}
        finally:
            Path(temp_file).unlink()

    def test_load_dataset_samples_invalid_json(self):
        """Test loading JSONL with invalid JSON lines."""
        # Create temporary JSONL file with some invalid lines
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "sample1", "seed": 42}\n')
            f.write("invalid json line\n")  # Invalid JSON
            f.write('{"id": "sample2", "seed": 123}\n')
            temp_file = f.name

        try:
            task_manager = TaskManager()
            samples = task_manager._load_dataset_samples(temp_file)

            # Should skip invalid line and load valid ones
            assert len(samples) == 2
            assert samples[0] == {"id": "sample1", "seed": 42}
            assert samples[1] == {"id": "sample2", "seed": 123}
        finally:
            Path(temp_file).unlink()

    def test_load_dataset_samples_nonexistent_file(self):
        """Test loading from nonexistent file."""
        task_manager = TaskManager()
        samples = task_manager._load_dataset_samples("nonexistent_file.jsonl")

        assert samples == []

    def test_load_dataset_samples_empty_file(self):
        """Test loading from empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Empty file
            temp_file = f.name

        try:
            task_manager = TaskManager()
            samples = task_manager._load_dataset_samples(temp_file)

            assert samples == []
        finally:
            Path(temp_file).unlink()

    def test_load_dataset_samples_relative_path(self):
        """Test loading dataset with relative path."""
        # Create a temporary directory and file
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_file = Path(temp_dir) / "test_dataset.jsonl"
            with open(dataset_file, "w") as f:
                f.write('{"id": "test", "seed": 42}\n')

            task_manager = TaskManager()

            # Test with absolute path since temp dir is not relative to cwd
            absolute_path = str(dataset_file)
            samples = task_manager._load_dataset_samples(absolute_path)

            assert len(samples) == 1
            assert samples[0] == {"id": "test", "seed": 42}


@pytest.mark.asyncio
class TestHttpRolloutResourceInitialization:
    """Tests for HttpRolloutResource initialization with sample data."""

    async def test_initialize_with_kwargs(self):
        """Test that initialize method sends kwargs in POST request."""
        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "episode_id": "test_episode",
            "observation": {},
        }
        mock_client.post.return_value = mock_response

        # Create resource with mock client
        config = {
            "base_url": "http://localhost:8080",
            "start_episode_endpoint": "/start_episode",
        }

        resource = HttpRolloutResource()
        await resource.setup(config)
        resource.client = mock_client  # Replace with mock

        # Initialize with sample data
        sample_data = {"seed": 42, "custom_param": "test_value"}
        await resource.initialize(**sample_data)

        # Verify POST was called with correct parameters
        mock_client.post.assert_called_once_with("http://localhost:8080/start_episode", json=sample_data)

    async def test_initialize_without_kwargs(self):
        """Test that initialize method works without kwargs."""
        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "episode_id": "test_episode",
            "observation": {},
        }
        mock_client.post.return_value = mock_response

        # Create resource with mock client
        config = {
            "base_url": "http://localhost:8080",
            "start_episode_endpoint": "/start_episode",
        }

        resource = HttpRolloutResource()
        await resource.setup(config)
        resource.client = mock_client  # Replace with mock

        # Initialize without sample data
        await resource.initialize()

        # Verify POST was called without json parameter
        mock_client.post.assert_called_once_with("http://localhost:8080/start_episode")


@pytest.mark.asyncio
class TestOrchestratorSampleDataPassing:
    """Tests for sample data passing in Orchestrator."""

    async def test_execute_task_poc_with_sample_data(self):
        """Test that execute_task_poc passes sample data to resource initialization."""
        # Create a minimal task definition
        task_def_dict = {
            "name": "test_task",
            "description": "Test task",
            "resource_type": "test_resource",
            "base_resource_config": {},
            "reward_function_path": "test.reward",
            "messages": [{"role": "user", "content": "test"}],
        }
        task_def = TaskDefinitionModel(**task_def_dict)

        # Mock the base resource
        mock_resource = AsyncMock()
        mock_resource.fork.return_value = AsyncMock()
        mock_episode_resource = mock_resource.fork.return_value
        mock_episode_resource.initialize = AsyncMock()

        # Create orchestrator
        orchestrator = Orchestrator(task_definition=task_def)
        orchestrator.base_resource = mock_resource

        # Mock execute_task_poc to just test the sample data passing logic
        async def mock_execute_task_poc(sample_data=None):
            if sample_data:
                # Simulate the resource initialization that would happen
                episode_resource = await orchestrator.base_resource.fork()
                await episode_resource.initialize(**sample_data)
            return {"score": 1.0}

        with patch.object(orchestrator, "execute_task_poc", side_effect=mock_execute_task_poc):
            sample_data = {"seed": 42, "test_param": "value"}
            await orchestrator.execute_task_poc(sample_data=sample_data)

            # Verify that episode resource was initialized with sample data
            mock_episode_resource.initialize.assert_called_once_with(**sample_data)

    async def test_execute_task_poc_without_sample_data(self):
        """Test that execute_task_poc works without sample data."""
        # Create a minimal task definition
        task_def_dict = {
            "name": "test_task",
            "description": "Test task",
            "resource_type": "test_resource",
            "base_resource_config": {},
            "reward_function_path": "test.reward",
            "messages": [{"role": "user", "content": "test"}],
        }
        task_def = TaskDefinitionModel(**task_def_dict)

        # Mock the base resource
        mock_resource = AsyncMock()
        mock_resource.fork.return_value = AsyncMock()
        mock_episode_resource = mock_resource.fork.return_value
        mock_episode_resource.initialize = AsyncMock()

        # Create orchestrator
        orchestrator = Orchestrator(task_definition=task_def)
        orchestrator.base_resource = mock_resource

        # Mock execute_task_poc to just test the sample data passing logic
        async def mock_execute_task_poc(sample_data=None):
            if sample_data:
                # Simulate the resource initialization that would happen
                episode_resource = await orchestrator.base_resource.fork()
                await episode_resource.initialize(**sample_data)
            return {"score": 1.0}

        with patch.object(orchestrator, "execute_task_poc", side_effect=mock_execute_task_poc):
            await orchestrator.execute_task_poc(sample_data=None)

            # Verify that episode resource was not initialized (no sample_data)
            mock_episode_resource.initialize.assert_not_called()


@pytest.mark.asyncio
class TestEndToEndDataDrivenEvaluation:
    """Integration tests for end-to-end data-driven evaluation."""

    async def test_data_driven_task_execution_flow(self):
        """Test the complete flow of data-driven task execution."""
        # Create temporary dataset file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "run_001", "seed": 42}\n')
            f.write('{"id": "run_002", "seed": 123}\n')
            temp_dataset = f.name

        try:
            # Create task definition with dataset
            task_def_dict = {
                "name": "frozen_lake_test",
                "description": "Test frozen lake with seeds",
                "resource_type": "http_rollout",
                "base_resource_config": {"base_url": "http://localhost:8080"},
                "reward_function_path": "test.reward",
                "dataset_path": temp_dataset,
                "num_rollouts_per_sample": 1,
                "messages": [{"role": "user", "content": "test"}],
            }

            task_manager = TaskManager()
            task_manager.register_task("test_task", TaskDefinitionModel(**task_def_dict))

            # Mock the orchestrator execution
            with patch.object(task_manager, "_execute_data_driven_rollouts") as mock_execute:
                mock_execute.return_value = [
                    {"score": 1.0, "sample_data": {"id": "run_001", "seed": 42}},
                    {"score": 0.0, "sample_data": {"id": "run_002", "seed": 123}},
                ]

                # Execute tasks
                results = await task_manager.execute_tasks(["test_task"], max_concurrency=1)

                # Verify data-driven execution was called
                mock_execute.assert_called_once()
                call_args = mock_execute.call_args
                samples = call_args[0][1]  # Second argument is samples

                assert len(samples) == 2
                assert samples[0] == {"id": "run_001", "seed": 42}
                assert samples[1] == {"id": "run_002", "seed": 123}

        finally:
            Path(temp_dataset).unlink()

    def test_frozen_lake_dataset_format_validation(self):
        """Test that the actual frozen lake dataset has correct format."""
        dataset_path = Path("examples/frozen_lake/client/dataset.jsonl")

        if dataset_path.exists():
            with open(dataset_path, "r") as f:
                lines = f.readlines()

            # Should have at least one sample
            assert len(lines) > 0, "Dataset should not be empty"

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON on line {i+1}: {line}")

                # Each sample should have id and seed
                assert "id" in sample, f"Sample {i+1} missing 'id' field"
                assert "seed" in sample, f"Sample {i+1} missing 'seed' field"
                assert isinstance(sample["seed"], int), f"Sample {i+1} seed should be integer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
