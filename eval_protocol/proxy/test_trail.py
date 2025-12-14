"""
Tests for Trail Management System proxy implementation.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
import redis

from proxy_core.models import ChatParams, ProxyConfig
from proxy_core.app import create_app
from proxy_core.auth import NoAuthProvider


@pytest.fixture
def mock_config():
    """Mock ProxyConfig."""
    return ProxyConfig(
        litellm_url="http://mock-litellm:8000",
        langfuse_host="https://mock-langfuse.com",
        langfuse_keys={
            "test-project": {
                "public_key": "pk-test",
                "secret_key": "sk-test"
            }
        },
        default_project_id="test-project",
        request_timeout=300.0
    )


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = Mock(spec=redis.Redis)
    mock.ping.return_value = True
    mock.close.return_value = None
    mock.sadd = Mock()
    return mock


@pytest.fixture
def app(mock_config, mock_redis):
    """Create test app."""
    app = create_app(auth_provider=NoAuthProvider())
    app.state.config = mock_config
    app.state.redis = mock_redis
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestTrailModels:
    """Test data models."""

    def test_chat_params_trail_id(self):
        """ChatParams accepts trail_id."""
        params = ChatParams(trail_id="test-trail-123", project_id="my-project")
        assert params.trail_id == "test-trail-123"
        assert params.project_id == "my-project"
        assert params.rollout_id is None

    def test_chat_params_backward_compatibility(self):
        """ChatParams still works with rollout_id."""
        params = ChatParams(
            rollout_id="rollout-123",
            invocation_id="inv-1",
            experiment_id="exp-1",
            run_id="run-1",
            row_id="row-1"
        )
        assert params.rollout_id == "rollout-123"
        assert params.trail_id is None


class TestTrailRoutes:
    """Test trail routes."""

    def test_trail_chat_routes_registered(self, client):
        """Trail chat completion routes exist."""
        routes = [route.path for route in client.app.routes]
        assert "/trails/{trail_id}/chat/completions" in routes
        assert "/v1/trails/{trail_id}/chat/completions" in routes
        assert "/project_id/{project_id}/trails/{trail_id}/chat/completions" in routes

    def test_trail_traces_routes_registered(self, client):
        """Trail traces routes exist."""
        routes = [route.path for route in client.app.routes]
        assert "/trails/{trail_id}/traces" in routes
        assert "/v1/trails/{trail_id}/traces" in routes
        assert "/trails/{trail_id}/traces/pointwise" in routes

    def test_legacy_routes_preserved(self, client):
        """Legacy rollout routes still exist."""
        routes_str = " ".join([route.path for route in client.app.routes])
        assert "rollout_id" in routes_str
        assert "invocation_id" in routes_str

    def test_health_endpoint(self, client):
        """Health endpoint works."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestTrailTagInjection:
    """Test tag injection logic."""

    @pytest.mark.asyncio
    async def test_trail_simple_tags(self, mock_config, mock_redis):
        """Trail requests inject simple tags (2 tags)."""
        from proxy_core.litellm import handle_chat_completion
        from fastapi import Request

        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Bearer test-key"}
        mock_request.body = AsyncMock(return_value=b'{"model": "test", "messages": []}')

        params = ChatParams(trail_id="test-trail-123")

        with patch('proxy_core.litellm.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'{"choices": []}'
            mock_response.headers = {}

            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await handle_chat_completion(mock_config, mock_redis, mock_request, params)

            sent_data = mock_post.call_args.kwargs['json']
            tags = sent_data['metadata']['tags']

            # Trail system: only 2 tags
            assert len(tags) == 2
            trail_tags = [t for t in tags if t.startswith('trail_id:')]
            assert len(trail_tags) == 1
            assert trail_tags[0] == 'trail_id:test-trail-123'

            insertion_tags = [t for t in tags if t.startswith('insertion_id:')]
            assert len(insertion_tags) == 1

    @pytest.mark.asyncio
    async def test_rollout_complex_tags(self, mock_config, mock_redis):
        """Rollout requests inject complex tags (6 tags) - backward compat."""
        from proxy_core.litellm import handle_chat_completion
        from fastapi import Request

        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Bearer test-key"}
        mock_request.body = AsyncMock(return_value=b'{"model": "test", "messages": []}')

        params = ChatParams(
            rollout_id="rollout-123",
            invocation_id="inv-1",
            experiment_id="exp-1",
            run_id="run-1",
            row_id="row-1"
        )

        with patch('proxy_core.litellm.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'{"choices": []}'
            mock_response.headers = {}

            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await handle_chat_completion(mock_config, mock_redis, mock_request, params)

            sent_data = mock_post.call_args.kwargs['json']
            tags = sent_data['metadata']['tags']

            # Legacy system: 6 tags
            assert len(tags) == 6
            tag_prefixes = [t.split(':')[0] for t in tags]
            assert 'rollout_id' in tag_prefixes
            assert 'invocation_id' in tag_prefixes
            assert 'experiment_id' in tag_prefixes


class TestRedisTracking:
    """Test Redis tracking."""

    @pytest.mark.asyncio
    async def test_redis_uses_trail_id_as_key(self, mock_config, mock_redis):
        """Redis uses trail_id as key."""
        from proxy_core.litellm import handle_chat_completion
        from fastapi import Request

        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Bearer test-key"}
        mock_request.body = AsyncMock(return_value=b'{"model": "test", "messages": []}')

        params = ChatParams(trail_id="my-trail-456")

        with patch('proxy_core.litellm.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'{"choices": []}'
            mock_response.headers = {}

            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await handle_chat_completion(mock_config, mock_redis, mock_request, params)

            # Verify Redis sadd was called with trail_id
            assert mock_redis.sadd.called
            call_args = mock_redis.sadd.call_args[0]
            assert call_args[0] == "my-trail-456"

            # Second arg should be insertion_id
            insertion_id = call_args[1]
            assert isinstance(insertion_id, str)
            assert len(insertion_id) > 0






