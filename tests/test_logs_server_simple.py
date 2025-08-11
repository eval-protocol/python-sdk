import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from eval_protocol.utils.logs_server import EvaluationWatcher, WebSocketManager


class TestWebSocketManagerBasic:
    """Basic tests for WebSocketManager without starting real loops."""

    def test_initialization(self):
        """Test WebSocketManager initialization."""
        manager = WebSocketManager()
        assert len(manager.active_connections) == 0
        assert manager._broadcast_queue is not None
        assert manager._broadcast_task is None

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test WebSocket connection and disconnection."""
        manager = WebSocketManager()
        mock_websocket = AsyncMock()

        # Test connection
        await manager.connect(mock_websocket)
        assert len(manager.active_connections) == 1
        assert mock_websocket in manager.active_connections
        mock_websocket.accept.assert_called_once()

        # Test disconnection
        manager.disconnect(mock_websocket)
        assert len(manager.active_connections) == 0
        assert mock_websocket not in manager.active_connections

    def test_broadcast_row_upserted(self):
        """Test broadcasting row upsert events."""
        manager = WebSocketManager()

        # Create a simple mock row
        mock_row = Mock()
        mock_row.model_dump.return_value = {"id": "test-123", "content": "test"}

        # Test that broadcast doesn't fail when no connections
        manager.broadcast_row_upserted(mock_row)

        # Test that message is queued
        assert not manager._broadcast_queue.empty()
        queued_message = manager._broadcast_queue.get_nowait()
        assert "type" in queued_message
        assert "row" in queued_message
        json_message = json.loads(queued_message)
        assert json_message["row"]["id"] == "test-123"
        assert json_message["row"]["content"] == "test"


class TestEvaluationWatcherBasic:
    """Basic tests for EvaluationWatcher without starting real threads."""

    def test_initialization(self):
        """Test EvaluationWatcher initialization."""
        mock_manager = Mock()
        watcher = EvaluationWatcher(mock_manager)
        assert watcher.websocket_manager == mock_manager
        assert watcher._thread is None
        assert watcher._stop_event is not None

    def test_start_stop(self):
        """Test starting and stopping the watcher."""
        mock_manager = Mock()
        watcher = EvaluationWatcher(mock_manager)

        # Test start
        watcher.start()
        assert watcher._thread is not None
        assert watcher._thread.is_alive()

        # Test stop
        watcher.stop()
        assert watcher._stop_event.is_set()
        if watcher._thread:
            watcher._thread.join(timeout=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
