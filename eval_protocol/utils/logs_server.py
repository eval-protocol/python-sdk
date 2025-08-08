import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from eval_protocol.dataset_logger import default_logger
from eval_protocol.utils.vite_server import ViteServer

default_logger

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts messages."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._loop = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        logs = default_logger.read()
        asyncio.run_coroutine_threadsafe(
            websocket.send_text(
                json.dumps(
                    {"type": "initialize_logs", "logs": [log.model_dump_json(exclude_none=True) for log in logs]}
                )
            ),
            self._loop,
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    def broadcast_file_update(self, update_type: str, file_path: str):
        """Broadcast file update to all connected clients."""
        if not file_path.startswith(default_logger.datasets_dir) or not file_path.endswith(".jsonl"):
            """
            .lock files are often created and deleted by the singleton lock
            mechanism so we only broadcast .jsonl files
            """
            return
        logger.info(f"Broadcasting file update: {update_type} {file_path}")

        logs = default_logger.read()
        # send initialize_logs message to all connected clients
        for connection in self.active_connections:
            asyncio.run_coroutine_threadsafe(
                connection.send_text(
                    json.dumps(
                        {"type": "initialize_logs", "logs": [log.model_dump_json(exclude_none=True) for log in logs]}
                    )
                ),
                self._loop,
            )

        message = {"type": update_type, "path": file_path, "timestamp": time.time()}
        # Include file contents for created and modified events
        if update_type in ["file_created", "file_changed"] and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    message["contents"] = f.read()
            except Exception as e:
                logger.warning(f"Failed to read file contents for {file_path}: {e}")
                message["contents"] = None
        elif update_type == "file_deleted":
            message["contents"] = None

        json_message = json.dumps(message)

        # Broadcast to all active connections
        for connection in self.active_connections:
            try:
                asyncio.run_coroutine_threadsafe(connection.send_text(json_message), self._loop)
            except Exception as e:
                logger.error(f"Failed to send message to WebSocket: {e}")
                # Remove broken connection
                self.active_connections.remove(connection)


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

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self.websocket_manager._loop = asyncio.get_running_loop()
            yield

        super().__init__(build_dir, host, port, index_file, lifespan=lifespan)

        # Add WebSocket endpoint
        self._setup_websocket_routes()

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
            return {
                "status": "ok",
                "build_dir": str(self.build_dir),
                "active_connections": len(self.websocket_manager.active_connections),
                "watch_paths": self.watch_paths,
            }

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

            # Store the event loop for WebSocket manager
            self.websocket_manager._loop = asyncio.get_running_loop()

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
    server.run()


if __name__ == "__main__":
    serve_logs()
