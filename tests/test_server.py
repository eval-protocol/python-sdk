from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from eval_protocol.models import EvaluateResult, MetricResult
from eval_protocol.server import create_app


@pytest.fixture
def test_reward_func():
    """Fixture that returns a test reward function."""

    def _reward_func(
        messages: List[Dict[str, str]],
        original_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> EvaluateResult:
        """Test reward function that returns a simple score."""
        metrics = {"test": MetricResult(score=0.5, success=True, reason="Test reason")}
        return EvaluateResult(score=0.5, reason="Test score reason", metrics=metrics)

    return _reward_func


class TestServer:
    """Tests for the FastAPI server."""

    @pytest.fixture
    def client(self, test_reward_func):
        """Create a test client for the FastAPI app."""
        app = create_app(test_reward_func)
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_reward_endpoint(self, client):
        """Test the reward endpoint."""
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "original_messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post("/reward", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["score"] == 0.5
        assert data["reason"] == "Test score reason"
        assert "metrics" in data
        assert "test" in data["metrics"]
        assert data["metrics"]["test"]["score"] == 0.5
        assert data["metrics"]["test"]["reason"] == "Test reason"
        assert data["metrics"]["test"]["is_score_valid"] is True

    def test_reward_endpoint_with_metadata(self, client):
        """Test the reward endpoint with metadata."""
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "original_messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"test_key": "test_value"},
        }

        response = client.post("/reward", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["score"] == 0.5

    def test_reward_endpoint_missing_required_fields(self, client):
        """Test the reward endpoint with missing required fields."""
        # Empty payload without messages field
        payload: Dict[str, Any] = {}

        response = client.post("/reward", json=payload)
        assert response.status_code == 422  # Validation error

    def test_reward_endpoint_malformed_messages(self, client):
        """Test the reward endpoint with malformed messages."""
        # Malformed messages - missing role
        payload: Dict[str, Any] = {
            "messages": [
                {"content": "Hello"},  # Missing role
                {"role": "assistant", "content": "Hi there"},
            ],
            "original_messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post("/reward", json=payload)
        assert response.status_code == 422  # Validation error
