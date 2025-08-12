import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from queue import Queue
from typing import TYPE_CHECKING, Any, List, Optional

import psutil
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from eval_protocol.dataset_logger import default_logger
from eval_protocol.dataset_logger.dataset_logger import LOG_EVENT_TYPE
from eval_protocol.event_bus import event_bus
from eval_protocol.utils.vite_server import ViteServer

if TYPE_CHECKING:
    from eval_protocol.models import EvaluationRow

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts messages."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._broadcast_queue: Queue = Queue()
        self._broadcast_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self.active_connections.append(websocket)
            connection_count = len(self.active_connections)
        logger.info(f"WebSocket connected. Total connections: {connection_count}")
        logs = default_logger.read()
        data = {
            "type": "initialize_logs",
            "logs": [log.model_dump(exclude_none=True, mode="json") for log in logs],
        }
        await websocket.send_text(json.dumps(data))

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            connection_count = len(self.active_connections)
        logger.info(f"WebSocket disconnected. Total connections: {connection_count}")

    def broadcast_row_upserted(self, row: "EvaluationRow"):
        """Broadcast a row-upsert event to all connected clients.

        Safe no-op if server loop is not running or there are no connections.
        """
        try:
            # Serialize pydantic model
            json_message = json.dumps({"type": "log", "row": row.model_dump(exclude_none=True, mode="json")})
            # Queue the message for broadcasting in the main event loop
            self._broadcast_queue.put(json_message)
        except Exception as e:
            logger.error(f"Failed to serialize row for broadcast: {e}")

    async def _start_broadcast_loop(self):
        """Start the broadcast loop that processes queued messages."""
        while True:
            try:
                # Wait for a message to be queued
                message = await asyncio.get_event_loop().run_in_executor(None, self._broadcast_queue.get)
                await self._send_text_to_all_connections(message)
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                logger.info("Broadcast loop cancelled")
                break

    async def _send_text_to_all_connections(self, text: str):
        with self._lock:
            connections = list(self.active_connections)

        if not connections:
            return

        tasks = []
        failed_connections = []

        for connection in connections:
            try:
                tasks.append(connection.send_text(text))
            except Exception as e:
                logger.error(f"Failed to send text to WebSocket: {e}")
                failed_connections.append(connection)

        # Execute all sends in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for any exceptions that occurred during execution
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to send text to WebSocket: {result}")
                    failed_connections.append(connections[i])

        # Remove all failed connections
        with self._lock:
            for connection in failed_connections:
                try:
                    self.active_connections.remove(connection)
                except ValueError:
                    pass

    def start_broadcast_loop(self):
        """Start the broadcast loop in the current event loop."""
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._start_broadcast_loop())

    def stop_broadcast_loop(self):
        """Stop the broadcast loop."""
        if self._broadcast_task and not self._broadcast_task.done():
            self._broadcast_task.cancel()
            self._broadcast_task = None


class EvaluationWatcher:
    """Monitors running evaluations and updates their status when processes stop."""

    def __init__(self, websocket_manager: "WebSocketManager"):
        self.websocket_manager = websocket_manager
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the evaluation watcher thread."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("Evaluation watcher started")

    def stop(self):
        """Stop the evaluation watcher thread."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Evaluation watcher stopped")

    def _watch_loop(self):
        """Main loop that checks for stopped processes every 2 seconds."""
        while self._running and not self._stop_event.is_set():
            try:
                self._check_running_evaluations()
                # Wait 2 seconds before next check
                self._stop_event.wait(2)
            except Exception as e:
                logger.error(f"Error in evaluation watcher loop: {e}")
                # Continue running even if there's an error
                time.sleep(1)

    def _check_running_evaluations(self):
        """Check all running evaluations and update status for stopped processes."""
        try:
            logs = default_logger.read()
            updated_rows = []

            for row in logs:
                if self._should_update_status(row):
                    logger.info(f"Updating status to 'stopped' for row {row.input_metadata.row_id} (PID {row.pid})")
                    if row.eval_metadata is not None:
                        row.eval_metadata.status = "stopped"
                    updated_rows.append(row)

            # Log all updated rows
            for row in updated_rows:
                default_logger.log(row)
                # Broadcast the update to connected clients
                self.websocket_manager.broadcast_row_upserted(row)

        except Exception as e:
            logger.error(f"Error checking running evaluations: {e}")

    def _should_update_status(self, row: "EvaluationRow") -> bool:
        """Check if a row's status should be updated to 'stopped'."""
        # Check if the row has running status and a PID
        if row.eval_metadata and row.eval_metadata.status == "running" and row.pid is not None:

            # Check if the process is still running
            try:
                process = psutil.Process(row.pid)
                # Check if process is still running
                if not process.is_running():
                    return True
            except psutil.NoSuchProcess:
                # Process no longer exists
                return True
            except psutil.AccessDenied:
                # Can't access process info, assume it's stopped
                logger.warning(f"Access denied to process {row.pid}, assuming stopped")
                return True
            except Exception as e:
                logger.error(f"Error checking process {row.pid}: {e}")
                # On error, assume process is still running to be safe
                return False

        return False


class LogsServer(ViteServer):
    """
    Enhanced server for serving Vite-built SPA with file watching and WebSocket support.

    This server extends ViteServer to add:
    - WebSocket connections for real-time updates
    - Live log streaming
    """

    def __init__(
        self,
        build_dir: str = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vite-app", "dist")
        ),
        host: str = "localhost",
        port: Optional[int] = 8000,
        index_file: str = "index.html",
    ):
        # Initialize WebSocket manager
        self.websocket_manager = WebSocketManager()

        super().__init__(build_dir, host, port, index_file)

        # Initialize evaluation watcher
        self.evaluation_watcher = EvaluationWatcher(self.websocket_manager)

        # Add WebSocket endpoint
        self._setup_websocket_routes()

        # Subscribe to events and start listening for cross-process events
        event_bus.subscribe(self._handle_event)

        logger.info(f"LogsServer initialized on {host}:{port}")

    def _setup_websocket_routes(self):
        """Set up WebSocket routes for real-time communication."""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.websocket_manager.disconnect(websocket)

        @self.app.get("/api/status")
        async def status():
            """Get server status including active connections."""
            with self.websocket_manager._lock:
                active_connections_count = len(self.websocket_manager.active_connections)
            return {
                "status": "ok",
                "build_dir": str(self.build_dir),
                "active_connections": active_connections_count,
                "watch_paths": self.watch_paths,
            }

    def _handle_event(self, event_type: str, data: Any) -> None:
        """Handle events from the event bus."""
        if event_type in [LOG_EVENT_TYPE]:
            from eval_protocol.models import EvaluationRow

            data = EvaluationRow(**data)
            self.websocket_manager.broadcast_row_upserted(data)

    def start_loops(self):
        """Start the broadcast loop and evaluation watcher."""
        self.websocket_manager.start_broadcast_loop()
        self.evaluation_watcher.start()
        event_bus.start_listening()

    async def run_async(self):
        """
        Run the logs server asynchronously with file watching.

        Args:
            reload: Whether to enable auto-reload (default: False)
        """
        try:
            logger.info(f"Starting LogsServer on {self.host}:{self.port}")
            logger.info(f"Serving files from: {self.build_dir}")
            logger.info("WebSocket endpoint available at /ws")

            self.start_loops()

            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info",
            )

            server = uvicorn.Server(config)
            await server.serve()

        except KeyboardInterrupt:
            logger.info("Shutting down LogsServer...")
        finally:
            # Clean up evaluation watcher
            self.evaluation_watcher.stop()
            # Clean up broadcast loop
            self.websocket_manager.stop_broadcast_loop()

    def run(self):
        """
        Run the logs server with file watching.

        Args:
            reload: Whether to enable auto-reload (default: False)
        """
        asyncio.run(self.run_async())


def create_app(host: str = "localhost", port: int = 8000, build_dir: Optional[str] = None) -> FastAPI:
    """
    Factory function to create a FastAPI app instance and start the server with async loops.

    This creates a LogsServer instance and starts it in a background thread to ensure
    all async loops (WebSocket broadcast, evaluation watching) are running.

    Args:
        host: Host to bind to
        port: Port to bind to
        build_dir: Optional custom build directory path

    Returns:
        FastAPI app instance with server running in background
    """
    if build_dir is None:
        build_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vite-app", "dist")
        )

    server = LogsServer(host=host, port=port, build_dir=build_dir)
    server.start_loops()
    return server.app


# For backward compatibility and direct usage
def serve_logs(port: Optional[int] = None):
    """
    Convenience function to create and run a LogsServer.
    """
    server = LogsServer(port=port)
    server.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the evaluation logs server")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--build-dir", help="Path to Vite build directory")

    args = parser.parse_args()

    # Create server with command line arguments
    if args.build_dir:
        server = LogsServer(host=args.host, port=args.port, build_dir=args.build_dir)
    else:
        server = LogsServer(host=args.host, port=args.port)

    server.run()
