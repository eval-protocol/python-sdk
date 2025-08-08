import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from queue import Queue
from typing import TYPE_CHECKING, Any, List, Optional

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
        await websocket.send_text(
            json.dumps({"type": "initialize_logs", "logs": [log.model_dump_json(exclude_none=True) for log in logs]})
        )

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
            json_message = json.dumps({"type": "log", "row": json.loads(row.model_dump_json(exclude_none=True))})
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
        for connection in connections:
            try:
                tasks.append(connection.send_text(text))
            except Exception as e:
                logger.error(f"Failed to send text to WebSocket: {e}")
                with self._lock:
                    try:
                        self.active_connections.remove(connection)
                    except ValueError:
                        pass
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def start_broadcast_loop(self):
        """Start the broadcast loop in the current event loop."""
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._start_broadcast_loop())

    def stop_broadcast_loop(self):
        """Stop the broadcast loop."""
        if self._broadcast_task and not self._broadcast_task.done():
            self._broadcast_task.cancel()


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

        # Add WebSocket endpoint
        self._setup_websocket_routes()

        # Subscribe to events and start listening for cross-process events
        event_bus.subscribe(self._handle_event)
        event_bus.start_listening()

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

            # Start the broadcast loop
            self.websocket_manager.start_broadcast_loop()

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
            # Clean up broadcast loop
            self.websocket_manager.stop_broadcast_loop()

    def run(self):
        """
        Run the logs server with file watching.

        Args:
            reload: Whether to enable auto-reload (default: False)
        """
        asyncio.run(self.run_async())


server = LogsServer()
app = server.app


def serve_logs():
    """
    Convenience function to create and run a LogsServer.
    """
    global server, app
    if server is None:
        server = LogsServer()
        app = server.app
    server.run()


if __name__ == "__main__":
    serve_logs()
