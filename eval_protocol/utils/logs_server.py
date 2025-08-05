import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import time
from typing import List, Optional
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from eval_protocol.dataset_logger import default_logger
from eval_protocol.utils.vite_server import ViteServer

default_logger

logger = logging.getLogger(__name__)


class FileWatcher(FileSystemEventHandler):
    """File system watcher that tracks file changes."""

    def __init__(self, websocket_manager):
        self.websocket_manager: WebSocketManager = websocket_manager
        self.ignored_patterns = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".DS_Store",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".coverage",
            "*.log",
            "*.tmp",
            "*.swp",
            "*.swo",
            "*~",
        }

    def should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored."""
        path_lower = path.lower()
        for pattern in self.ignored_patterns:
            if pattern.startswith("*"):
                if path_lower.endswith(pattern[1:]):
                    return True
            elif pattern in path_lower:
                return True
        return False

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and not self.should_ignore(event.src_path):
            self.websocket_manager.broadcast_file_update("file_created", event.src_path)

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and not self.should_ignore(event.src_path):
            self.websocket_manager.broadcast_file_update("file_changed", event.src_path)

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory and not self.should_ignore(event.src_path):
            self.websocket_manager.broadcast_file_update("file_deleted", event.src_path)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts messages."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._loop = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    def broadcast_file_update(self, update_type: str, file_path: str):
        """Broadcast file update to all connected clients."""
        if not file_path.startswith(default_logger.datasets_dir):
            return
        logger.info(f"Broadcasting file update: {update_type} {file_path}")

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
    - File system watching
    - WebSocket connections for real-time updates
    - Live log streaming
    """

    def __init__(
        self,
        build_dir: str = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vite-app", "dist")
        ),
        host: str = "localhost",
        port: Optional[int] = None,
        index_file: str = "index.html",
        watch_paths: Optional[List[str]] = None,
    ):
        # Initialize WebSocket manager
        self.websocket_manager = WebSocketManager()

        # Set up file watching
        self.watch_paths = watch_paths or [os.getcwd()]
        self.observer = Observer()
        self.file_watcher = FileWatcher(self.websocket_manager)
        self._file_watching_started = False

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            print("before")
            self.start_file_watching()
            self.websocket_manager._loop = asyncio.get_running_loop()
            yield
            print("after")
            self.stop_file_watching()

        super().__init__(build_dir, host, port, index_file, lifespan=lifespan)

        # Add WebSocket endpoint
        self._setup_websocket_routes()

        logger.info(f"LogsServer initialized on {host}:{port}")
        logger.info(f"Watching paths: {self.watch_paths}")

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

    def start_file_watching(self):
        """Start watching file system for changes."""
        # Check if file watching has already been started
        if self._file_watching_started:
            logger.info("File watching already started, skipping")
            return

        for path in self.watch_paths:
            if os.path.exists(path):
                self.observer.schedule(self.file_watcher, path, recursive=True)
                logger.info(f"Started watching: {path}")
            else:
                logger.warning(f"Watch path does not exist: {path}")

        self.observer.start()
        self._file_watching_started = True
        logger.info("File watching started")

    def stop_file_watching(self):
        """Stop watching file system."""
        if self._file_watching_started:
            self.observer.stop()
            self.observer.join()
            self._file_watching_started = False
            logger.info("File watching stopped")

    async def run_async(self):
        """
        Run the logs server asynchronously with file watching.

        Args:
            reload: Whether to enable auto-reload (default: False)
        """
        try:
            # Start file watching
            self.start_file_watching()

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
        finally:
            self.stop_file_watching()

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

    Args:
        build_dir: Path to the Vite build output directory
        host: Host to bind the server to
        port: Port to bind the server to (default: 4789 for logs)
        index_file: Name of the main index file
        watch_paths: List of paths to watch for file changes
        reload: Whether to enable auto-reload
    """
    server.run()


if __name__ == "__main__":
    serve_logs()
