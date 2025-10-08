import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import psutil
import pytest
from fastapi import FastAPI
from fastapi.routing import APIWebSocketRoute
from fastapi.testclient import TestClient

from eval_protocol.dataset_logger import default_logger
from eval_protocol.dataset_logger.dataset_logger import LOG_EVENT_TYPE
from eval_protocol.event_bus import event_bus
from eval_protocol.models import EvalMetadata, EvaluationRow, InputMetadata, Message, Status
from eval_protocol.utils.logs_server import (
    EvaluationWatcher,
    LogsServer,
    WebSocketManager,
    create_app,
    serve_logs,
)


class TestWebSocketManager:
    """Test WebSocketManager class."""

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
        with patch.object(default_logger, "read", return_value=[]):
            await manager.connect(mock_websocket)
        assert len(manager.active_connections) == 1
        assert mock_websocket in manager.active_connections
        mock_websocket.accept.assert_called_once()

        # Test disconnection
        manager.disconnect(mock_websocket)
        assert len(manager.active_connections) == 0
        assert mock_websocket not in manager.active_connections

    @pytest.mark.asyncio
    async def test_connect_sends_initial_logs(self):
        """Test that connecting sends initial logs."""
        manager = WebSocketManager()
        mock_websocket = AsyncMock()

        # Mock default_logger.read()
        mock_logs = [
            EvaluationRow(
                messages=[Message(role="user", content="test")],
                input_metadata=InputMetadata(row_id="test-123"),
            )
        ]

        with patch.object(default_logger, "read", return_value=mock_logs):
            await manager.connect(mock_websocket)

        # Verify that initial logs were sent
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "initialize_logs"
        assert len(sent_data["logs"]) == 1

    def test_broadcast_row_upserted(self):
        """Test broadcasting row upsert events."""
        manager = WebSocketManager()
        test_row = EvaluationRow(
            messages=[Message(role="user", content="test")],
            input_metadata=InputMetadata(row_id="test-123"),
        )

        # Test that broadcast doesn't fail when no connections
        manager.broadcast_row_upserted(test_row)

        # Test that message is queued
        assert not manager._broadcast_queue.empty()
        queued_message = manager._broadcast_queue.get_nowait()
        data = json.loads(queued_message)
        assert data["type"] == "log"
        assert "row" in data
        assert data["row"]["messages"][0]["content"] == "test"
        assert data["row"]["input_metadata"]["row_id"] == "test-123"

    @pytest.mark.asyncio
    async def test_broadcast_loop(self):
        """Test the broadcast loop functionality."""
        manager = WebSocketManager()
        mock_websocket = AsyncMock()
        await manager.connect(mock_websocket)

        # Test that broadcast loop can be started and stopped
        manager.start_broadcast_loop()
        assert manager._broadcast_task is not None

        # Stop broadcast loop
        manager.stop_broadcast_loop()
        assert manager._broadcast_task is None

    @pytest.mark.asyncio
    async def test_send_text_to_all_connections(self):
        """Test sending text to all connections."""
        manager = WebSocketManager()
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()

        # Mock default_logger.read() to return empty logs
        with patch.object(default_logger, "read", return_value=[]):
            await manager.connect(mock_websocket1)
            await manager.connect(mock_websocket2)

        test_message = "test message"
        await manager._send_text_to_all_connections(test_message)

        # Check that the test message was sent to both websockets
        mock_websocket1.send_text.assert_any_call(test_message)
        mock_websocket2.send_text.assert_any_call(test_message)

    @pytest.mark.asyncio
    async def test_send_text_handles_failed_connections(self):
        """Test that failed connections are handled gracefully."""
        manager = WebSocketManager()
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()

        # Mock default_logger.read() to return empty logs
        with patch.object(default_logger, "read", return_value=[]):
            await manager.connect(mock_websocket1)
            await manager.connect(mock_websocket2)

        # Make the second websocket fail AFTER connection is established
        # We need to make send_text raise an exception when awaited
        async def failing_send_text(text):
            raise Exception("Connection failed")

        mock_websocket2.send_text = failing_send_text

        test_message = "test message"
        await manager._send_text_to_all_connections(test_message)

        # First websocket should receive the message
        mock_websocket1.send_text.assert_any_call(test_message)
        # Second websocket should have been removed due to failure
        assert len(manager.active_connections) == 1
        assert mock_websocket1 in manager.active_connections


class TestEvaluationWatcher:
    """Test EvaluationWatcher class."""

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

    @patch("psutil.Process")
    def test_should_update_status_running_process(self, mock_process):
        """Test status update for running process."""
        mock_manager = Mock()
        watcher = EvaluationWatcher(mock_manager)

        # Mock a running process
        mock_process_instance = Mock()
        mock_process_instance.is_running.return_value = True
        mock_process.return_value = mock_process_instance

        test_row = EvaluationRow(
            messages=[Message(role="user", content="test")],
            input_metadata=InputMetadata(row_id="test-123"),
            eval_metadata=EvalMetadata(
                name="test_eval", num_runs=1, aggregation_method="mean", status=Status.rollout_running()
            ),
            pid=12345,
        )

        # Process is running, should not update
        assert watcher._should_update_status(test_row) is False

    @patch("psutil.Process")
    def test_should_update_status_stopped_process(self, mock_process):
        """Test status update for stopped process."""
        mock_manager = Mock()
        watcher = EvaluationWatcher(mock_manager)

        # Mock a stopped process
        mock_process_instance = Mock()
        mock_process_instance.is_running.return_value = False
        mock_process.return_value = mock_process_instance

        test_row = EvaluationRow(
            messages=[Message(role="user", content="test")],
            input_metadata=InputMetadata(row_id="test-123"),
            eval_metadata=EvalMetadata(
                name="test_eval", num_runs=1, aggregation_method="mean", status=Status.rollout_running()
            ),
            pid=12345,
        )

        # Process is stopped, should update
        assert watcher._should_update_status(test_row) is True

    @patch("psutil.Process")
    def test_should_update_status_no_such_process(self, mock_process):
        """Test status update for non-existent process."""
        mock_manager = Mock()
        watcher = EvaluationWatcher(mock_manager)

        # Mock process not found
        mock_process.side_effect = psutil.NoSuchProcess(pid=999)

        test_row = EvaluationRow(
            messages=[Message(role="user", content="test")],
            input_metadata=InputMetadata(row_id="test-123"),
            eval_metadata=EvalMetadata(
                name="test_eval", num_runs=1, aggregation_method="mean", status=Status.rollout_running()
            ),
            pid=999,
        )

        # Process doesn't exist, should update
        assert watcher._should_update_status(test_row) is True

    def test_should_update_status_not_running(self):
        """Test status update for non-running evaluation."""
        mock_manager = Mock()
        watcher = EvaluationWatcher(mock_manager)

        test_row = EvaluationRow(
            messages=[Message(role="user", content="test")],
            input_metadata=InputMetadata(row_id="test-123"),
            eval_metadata=EvalMetadata(
                name="test_eval", num_runs=1, aggregation_method="mean", status=Status.rollout_finished()
            ),
            rollout_status=Status.rollout_finished(),
            pid=12345,
        )

        # Not running status, should not update
        assert watcher._should_update_status(test_row) is False

    def test_should_update_status_no_pid(self):
        """Test status update for evaluation without PID."""
        mock_manager = Mock()
        watcher = EvaluationWatcher(mock_manager)

        test_row = EvaluationRow(
            messages=[Message(role="user", content="test")],
            input_metadata=InputMetadata(row_id="test-123"),
            eval_metadata=EvalMetadata(
                name="test_eval", num_runs=1, aggregation_method="mean", status=Status.rollout_running()
            ),
            pid=None,
        )

        # No PID, should not update
        assert watcher._should_update_status(test_row) is False


class TestLogsServer:
    """Test LogsServer class."""

    @pytest.fixture
    def temp_build_dir(self):
        """Create a temporary build directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create a minimal index.html file
            (temp_path / "index.html").write_text("<html><body>Test</body></html>")
            # Create assets directory (required by ViteServer)
            (temp_path / "assets").mkdir(exist_ok=True)
            yield temp_path

    def test_initialization(self, temp_build_dir: Path):
        """Test LogsServer initialization."""
        server = LogsServer(build_dir=str(temp_build_dir))
        assert server.build_dir == temp_build_dir
        assert server.websocket_manager is not None
        assert server.evaluation_watcher is not None

    def test_initialization_invalid_build_dir(self):
        """Test LogsServer initialization with invalid build directory."""
        with pytest.raises(FileNotFoundError, match="Build directory '/nonexistent/path' does not exist"):
            LogsServer(build_dir="/nonexistent/path")

    def test_websocket_routes(self, temp_build_dir):
        """Test that WebSocket routes are properly set up."""
        server = LogsServer(build_dir=str(temp_build_dir))

        # Check that the WebSocket endpoint exists
        if not server.app.routes:
            raise ValueError("No routes found")
        for route in server.app.routes:
            if isinstance(route, APIWebSocketRoute) and route.path == "/ws":
                break
        else:
            raise ValueError("WebSocket route not found")

    @pytest.mark.asyncio
    async def test_handle_event(self, temp_build_dir):
        """Test event handling."""
        server = LogsServer(build_dir=str(temp_build_dir))

        # Test handling a log event
        test_row = {
            "messages": [{"role": "user", "content": "test"}],
            "input_metadata": {"row_id": "test-123"},
        }

        server._handle_event(LOG_EVENT_TYPE, test_row)
        # The event should be queued for broadcasting
        assert not server.websocket_manager._broadcast_queue.empty()

    @pytest.mark.asyncio
    async def test_create_app_factory(self, temp_build_dir):
        """Test the create_app factory function."""
        with patch("eval_protocol.utils.logs_server.LogsServer.start_loops") as mock_start_loops:
            app = create_app(build_dir=str(temp_build_dir))
            assert isinstance(app, FastAPI)
            # Verify that start_loops was called
            mock_start_loops.assert_called_once()

    def test_serve_logs_convenience_function(self, temp_build_dir):
        """Test the serve_logs convenience function."""
        # Mock the LogsServer.run method to avoid actually starting a server
        with patch("eval_protocol.utils.logs_server.LogsServer.run") as mock_run:
            # This should not raise an error
            serve_logs(port=8001)
            # Verify that the run method was called
            mock_run.assert_called_once()

    def test_serve_logs_port_parameter(self, temp_build_dir):
        """Test that serve_logs properly passes the port parameter to LogsServer."""
        with patch("eval_protocol.utils.logs_server.LogsServer") as mock_logs_server_class:
            mock_server_instance = Mock()
            mock_logs_server_class.return_value = mock_server_instance

            # Call serve_logs with a specific port
            test_port = 9000
            serve_logs(port=test_port)

            # Verify that LogsServer was created with the correct port
            mock_logs_server_class.assert_called_once_with(port=test_port, elasticsearch_config=None, debug=False)
            # Verify that the run method was called on the instance
            mock_server_instance.run.assert_called_once()

    def test_serve_logs_default_port(self, temp_build_dir):
        """Test that serve_logs uses default port when none is specified."""
        with patch("eval_protocol.utils.logs_server.LogsServer") as mock_logs_server_class:
            mock_server_instance = Mock()
            mock_logs_server_class.return_value = mock_server_instance

            # Call serve_logs without specifying a port
            serve_logs()

            # Verify that LogsServer was created with None port (which will use LogsServer's default of 8000)
            mock_logs_server_class.assert_called_once_with(port=None, elasticsearch_config=None, debug=False)
            # Verify that the run method was called on the instance
            mock_server_instance.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_async_lifecycle(self, temp_build_dir):
        """Test the async lifecycle of the server."""
        server = LogsServer(build_dir=str(temp_build_dir))

        # Mock the uvicorn.Server to avoid actually starting a server
        with patch("uvicorn.Server") as mock_uvicorn_server:
            mock_server = Mock()
            mock_server.serve = AsyncMock()
            mock_uvicorn_server.return_value = mock_server

            # Start the server
            start_task = asyncio.create_task(server.run_async())

            # Wait a bit for it to start
            await asyncio.sleep(0.1)

            # Cancel the task instead of calling non-existent stop method
            start_task.cancel()

            # Wait for the task to complete
            try:
                await start_task
            except asyncio.CancelledError:
                pass


class TestLogsServerIntegration:
    """Integration tests for LogsServer."""

    @pytest.fixture
    def temp_build_dir_with_files(self):
        """Create a temporary build directory with index.html and assets/ directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create index.html
            (temp_path / "index.html").write_text("<html><body>Test</body></html>")

            # Create assets directory and some files inside it
            assets_dir = temp_path / "assets"
            assets_dir.mkdir()
            (assets_dir / "app.js").write_text("console.log('test');")
            (assets_dir / "style.css").write_text("body { color: black; }")

            # Optionally, create a nested directory inside assets
            (assets_dir / "nested").mkdir()
            (assets_dir / "nested" / "file.txt").write_text("nested content")

            yield temp_path

    def test_static_file_serving(self, temp_build_dir_with_files):
        """Test that static files are served correctly."""
        server = LogsServer(build_dir=str(temp_build_dir_with_files))
        client = TestClient(server.app)

        # Test serving index.html
        response = client.get("/")
        assert response.status_code == 200
        assert "Test" in response.text

        # Test serving static files
        response = client.get("/assets/app.js")
        assert response.status_code == 200
        assert "console.log('test')" in response.text

        response = client.get("/assets/style.css")
        assert response.status_code == 200
        assert "color: black" in response.text

    def test_spa_routing(self, temp_build_dir_with_files):
        """Test SPA routing fallback."""
        server = LogsServer(build_dir=str(temp_build_dir_with_files))
        client = TestClient(server.app)

        # Test that non-existent routes fall back to index.html
        response = client.get("/some/nonexistent/route")
        assert response.status_code == 200
        assert "Test" in response.text

    def test_root_endpoint(self, temp_build_dir_with_files):
        """Test the root endpoint."""
        server = LogsServer(build_dir=str(temp_build_dir_with_files))
        client = TestClient(server.app)

        response = client.get("/")
        assert response.status_code == 200
        assert "Test" in response.text

    def test_health_endpoint(self, temp_build_dir_with_files):
        """Test the health endpoint."""
        server = LogsServer(build_dir=str(temp_build_dir_with_files))
        client = TestClient(server.app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_server_runs_on_specific_port(self):
        """Integration test: verify that LogsServer runs on specified port and handles port parameters correctly."""
        import multiprocessing
        import socket

        def find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port

        test_port = find_free_port()

        # Start server with dynamic port and build_dir
        server_process = multiprocessing.Process(target=serve_logs, kwargs={"port": test_port}, daemon=True)
        server_process.start()

        # Wait for server to be ready
        for _ in range(30):
            try:
                response = httpx.get(f"http://localhost:{test_port}/health", timeout=1)
                if response.status_code == 200:
                    break
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)

        async with httpx.AsyncClient() as client:
            # Test health endpoint
            response = await client.get(f"http://localhost:{test_port}/health", timeout=10)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"

        # Clean up server
        if server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=2)
            if server_process.is_alive():
                server_process.kill()
                server_process.join(timeout=1)


@pytest.mark.asyncio
class TestAsyncWebSocketOperations:
    """Test async WebSocket operations."""

    async def test_websocket_connection_lifecycle(self):
        """Test complete WebSocket connection lifecycle."""
        manager = WebSocketManager()

        # Create mock WebSocket
        mock_websocket = AsyncMock()

        # Test connection
        with patch.object(default_logger, "read", return_value=[]):
            await manager.connect(mock_websocket)
        assert len(manager.active_connections) == 1

        # Test broadcasting without starting the loop
        test_row = EvaluationRow(
            messages=[Message(role="user", content="test")],
            input_metadata=InputMetadata(row_id="test-123"),
        )
        manager.broadcast_row_upserted(test_row)

        # Verify message was queued
        assert not manager._broadcast_queue.empty()

        # Test disconnection
        manager.disconnect(mock_websocket)
        assert len(manager.active_connections) == 0

    async def test_multiple_websocket_connections(self):
        """Test handling multiple WebSocket connections."""
        manager = WebSocketManager()

        # Create multiple mock WebSockets
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        mock_websocket3 = AsyncMock()

        # Connect all
        with patch.object(default_logger, "read", return_value=[]):
            await manager.connect(mock_websocket1)
            await manager.connect(mock_websocket2)
            await manager.connect(mock_websocket3)
        assert len(manager.active_connections) == 3

        # Test broadcasting to all without starting the loop
        test_row = EvaluationRow(
            messages=[Message(role="user", content="test")],
            input_metadata=InputMetadata(row_id="test-123"),
        )
        manager.broadcast_row_upserted(test_row)

        # Verify message was queued
        assert not manager._broadcast_queue.empty()

        # Disconnect one
        manager.disconnect(mock_websocket2)
        assert len(manager.active_connections) == 2
