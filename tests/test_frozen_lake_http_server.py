"""
Tests for FrozenLake HTTP rollout server seed handling.

This module tests the HTTP server's ability to accept and use seed parameters
to create reproducible game environments.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from examples.frozen_lake.gymnasium_frozen_lake_server import GymnasiumFrozenLakeGame

# Import the server components
from examples.frozen_lake.server.http_rollout_server import app


class TestFrozenLakeHttpServer:
    """Tests for the FrozenLake HTTP rollout server."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_start_episode_without_seed(self):
        """Test starting episode without seed uses default behavior."""
        response = self.client.post("/start_episode")

        assert response.status_code == 200
        data = response.json()

        assert "episode_id" in data
        assert "observation" in data

        # Should have standard game state
        observation = data["observation"]
        assert "position" in observation
        assert "current_cell" in observation
        assert "visual" in observation
        assert observation["position"] == [0, 0]  # Start position
        assert observation["current_cell"] == "S"  # Start cell

    def test_start_episode_with_seed(self):
        """Test starting episode with seed parameter."""
        seed_value = 42
        request_data = {"seed": seed_value}

        response = self.client.post("/start_episode", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "episode_id" in data
        assert "observation" in data

        # Should still start at position (0,0)
        observation = data["observation"]
        assert observation["position"] == [0, 0]
        assert observation["current_cell"] == "S"

    def test_different_seeds_create_different_episodes(self):
        """Test that different seeds create episodes with different board layouts."""
        # Start episode with seed 42
        response1 = self.client.post("/start_episode", json={"seed": 42})
        assert response1.status_code == 200
        data1 = response1.json()
        episode_id1 = data1["episode_id"]
        visual1 = data1["observation"]["visual"]

        # Start episode with seed 123
        response2 = self.client.post("/start_episode", json={"seed": 123})
        assert response2.status_code == 200
        data2 = response2.json()
        episode_id2 = data2["episode_id"]
        visual2 = data2["observation"]["visual"]

        # Episodes should have different IDs
        assert episode_id1 != episode_id2

        # Board layouts should be different (high probability with different seeds)
        assert visual1 != visual2, "Different seeds should create different board layouts"

    def test_same_seed_creates_identical_episodes(self):
        """Test that same seed creates episodes with identical board layouts."""
        seed_value = 999

        # Start two episodes with same seed
        response1 = self.client.post("/start_episode", json={"seed": seed_value})
        assert response1.status_code == 200
        data1 = response1.json()
        visual1 = data1["observation"]["visual"]

        response2 = self.client.post("/start_episode", json={"seed": seed_value})
        assert response2.status_code == 200
        data2 = response2.json()
        visual2 = data2["observation"]["visual"]

        # Board layouts should be identical
        assert visual1 == visual2, "Same seed should create identical board layouts"

    def test_start_episode_with_additional_parameters(self):
        """Test that server accepts additional parameters beyond seed."""
        request_data = {"seed": 42, "custom_param": "test_value", "id": "test_run_001"}

        response = self.client.post("/start_episode", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should work normally despite extra parameters
        assert "episode_id" in data
        assert "observation" in data

    def test_step_action_in_seeded_episode(self):
        """Test taking actions in a seeded episode."""
        # Start episode with seed
        response = self.client.post("/start_episode", json={"seed": 42})
        assert response.status_code == 200
        episode_id = response.json()["episode_id"]

        # Take a step action
        step_response = self.client.post("/step", json={"episode_id": episode_id, "action": "right"})

        assert step_response.status_code == 200
        step_data = step_response.json()

        assert "observation" in step_data
        assert "is_done" in step_data

        # Position should have changed (unless blocked)
        observation = step_data["observation"]
        assert "position" in observation
        assert "current_cell" in observation

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_episode_cleanup_on_completion(self):
        """Test that episodes are properly tracked and can be cleaned up."""
        # Start an episode
        response = self.client.post("/start_episode", json={"seed": 42})
        assert response.status_code == 200
        episode_id = response.json()["episode_id"]

        # Episode should be trackable via step endpoint
        step_response = self.client.post("/step", json={"episode_id": episode_id, "action": "right"})
        assert step_response.status_code == 200

    def test_invalid_episode_id_handling(self):
        """Test handling of invalid episode IDs."""
        # Try to step with non-existent episode ID
        response = self.client.post("/step", json={"episode_id": "non_existent_episode", "action": "right"})

        # Should return an error (400 or 404)
        assert response.status_code in [400, 404]

    def test_server_configuration_with_slippery_environment(self):
        """Test that server is configured with slippery environment for seed demonstration."""
        # Mock the game creation to verify configuration
        with patch("examples.frozen_lake.server.http_rollout_server.FrozenLakeGame") as mock_game_class:
            mock_game_instance = MagicMock()
            mock_game_instance.reset.return_value = {
                "position": [0, 0],
                "current_cell": "S",
                "visual": "test_visual",
                "done": False,
            }
            mock_game_class.return_value = mock_game_instance

            response = self.client.post("/start_episode", json={"seed": 42})

            # Verify game was created with correct configuration
            mock_game_class.assert_called_once()
            call_kwargs = mock_game_class.call_args[1]

            # Should include slippery=True and the seed
            assert call_kwargs.get("is_slippery") is True
            assert call_kwargs.get("seed") == 42


class TestGymnasiumFrozenLakeIntegration:
    """Integration tests between HTTP server and GymnasiumFrozenLakeGame."""

    def test_gymnasium_integration_with_seeds(self):
        """Test that the HTTP server correctly integrates with GymnasiumFrozenLakeGame seeds."""
        client = TestClient(app)

        # Test multiple seeds to ensure they work through the HTTP interface
        seeds = [42, 123, 999]
        board_layouts = []

        for seed in seeds:
            response = client.post("/start_episode", json={"seed": seed})
            assert response.status_code == 200

            observation = response.json()["observation"]
            board_layouts.append(observation["visual"])

        # All board layouts should be different
        unique_layouts = set(board_layouts)
        assert len(unique_layouts) == len(seeds), "Each seed should produce a unique board layout"

    def test_episode_state_consistency(self):
        """Test that episode state remains consistent within a single game."""
        client = TestClient(app)

        # Start episode with specific seed
        response = client.post("/start_episode", json={"seed": 42})
        assert response.status_code == 200

        episode_id = response.json()["episode_id"]
        initial_visual = response.json()["observation"]["visual"]

        # Take a few actions and verify board layout doesn't change
        for action in ["right", "down", "left"]:
            step_response = client.post("/step", json={"episode_id": episode_id, "action": action})
            assert step_response.status_code == 200

            observation = step_response.json()["observation"]
            # Visual board should remain the same (only position marker changes)
            current_visual = observation["visual"]

            # Board structure should be preserved (same letters, different position marker)
            # Extract just the board cells without position markers
            initial_cells = initial_visual.replace("[", "").replace("]", "")
            current_cells = current_visual.replace("[", "").replace("]", "")

            # The underlying board structure should be identical
            assert len(initial_cells) == len(current_cells), "Board size should remain constant"

    def test_seed_parameter_propagation(self):
        """Test that seed parameter correctly propagates to the game engine."""
        # This test verifies the complete data flow from HTTP request to game creation

        with patch("examples.frozen_lake.server.http_rollout_server.FrozenLakeGame") as mock_game_class:
            mock_game_instance = MagicMock()
            mock_game_instance.reset.return_value = {
                "position": [0, 0],
                "current_cell": "S",
                "visual": "mocked_visual",
                "done": False,
                "won": False,
                "message": "test_message",
            }
            mock_game_class.return_value = mock_game_instance

            client = TestClient(app)

            # Send request with seed
            seed_value = 1337
            response = client.post("/start_episode", json={"seed": seed_value})

            assert response.status_code == 200

            # Verify the game was created with the correct seed
            mock_game_class.assert_called_once()
            call_kwargs = mock_game_class.call_args[1]
            assert call_kwargs["seed"] == seed_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
