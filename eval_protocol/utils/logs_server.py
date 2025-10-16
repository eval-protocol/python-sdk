import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Literal

import psutil
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from eval_protocol.dataset_logger import default_logger
from eval_protocol.dataset_logger.dataset_logger import LOG_EVENT_TYPE
from eval_protocol.event_bus import event_bus
from eval_protocol.models import Status
from eval_protocol.pytest.elasticsearch_setup import ElasticsearchSetup
from eval_protocol.utils.vite_server import ViteServer
from eval_protocol.log_utils.elasticsearch_client import ElasticsearchClient
from eval_protocol.types.remote_rollout_processor import ElasticsearchConfig
from eval_protocol.utils.logs_models import LogEntry, LogsResponse
from eval_protocol.utils.browser_utils import write_pid_file

if TYPE_CHECKING:
    from eval_protocol.models import EvaluationRow

logger = logging.getLogger(__name__)


def enable_debug_mode():
    """Enable debug mode for all relevant loggers in the logs server system."""
    # Set debug level for all relevant loggers
    logger.setLevel(logging.DEBUG)

    # Set debug level for event bus logger
    from eval_protocol.event_bus.logger import logger as event_bus_logger

    event_bus_logger.setLevel(logging.DEBUG)

    print("Debug mode enabled for all relevant loggers")


class WebSocketManager:
    """Manages WebSocket connections and broadcasts messages."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._broadcast_queue: Queue = Queue()
        self._broadcast_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        logger.debug("[WEBSOCKET_CONNECT] New websocket connection attempt")
        await websocket.accept()
        with self._lock:
            self.active_connections.append(websocket)
            connection_count = len(self.active_connections)
        logger.info(f"[WEBSOCKET_CONNECT] WebSocket connected. Total connections: {connection_count}")

        logger.debug("[WEBSOCKET_CONNECT] Reading logs for initialization")
        logs = default_logger.read()
        logger.debug(f"[WEBSOCKET_CONNECT] Found {len(logs)} logs to send")

        data = {
            "type": "initialize_logs",
            "logs": [log.model_dump(exclude_none=True, mode="json") for log in logs],
        }
        logger.debug("[WEBSOCKET_CONNECT] Sending initialization data")
        await websocket.send_text(json.dumps(data))
        logger.debug("[WEBSOCKET_CONNECT] Successfully sent initialization data")

    def disconnect(self, websocket: WebSocket):
        logger.debug("[WEBSOCKET_DISCONNECT] WebSocket disconnection")
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                logger.debug("[WEBSOCKET_DISCONNECT] Removed websocket from active connections")
            else:
                logger.debug("[WEBSOCKET_DISCONNECT] Websocket was not in active connections")
            connection_count = len(self.active_connections)
        logger.info(f"[WEBSOCKET_DISCONNECT] WebSocket disconnected. Total connections: {connection_count}")

    def broadcast_row_upserted(self, row: "EvaluationRow"):
        """Broadcast a row-upsert event to all connected clients.

        Safe no-op if server loop is not running or there are no connections.
        """
        rollout_id = row.execution_metadata.rollout_id if row.execution_metadata else "unknown"
        logger.debug(f"[WEBSOCKET_BROADCAST] Starting broadcast for rollout_id: {rollout_id}")

        with self._lock:
            active_connections_count = len(self.active_connections)
        logger.debug(f"[WEBSOCKET_BROADCAST] Active connections: {active_connections_count}")

        try:
            # Serialize pydantic model
            logger.debug(f"[WEBSOCKET_BROADCAST] Serializing row for rollout_id: {rollout_id}")
            json_message = json.dumps({"type": "log", "row": row.model_dump(exclude_none=True, mode="json")})
            logger.debug(
                f"[WEBSOCKET_BROADCAST] Successfully serialized message (length: {len(json_message)}) for rollout_id: {rollout_id}"
            )

            # Queue the message for broadcasting in the main event loop
            logger.debug(f"[WEBSOCKET_BROADCAST] Queuing message for broadcast for rollout_id: {rollout_id}")
            self._broadcast_queue.put(json_message)
            logger.debug(f"[WEBSOCKET_BROADCAST] Successfully queued message for rollout_id: {rollout_id}")
        except Exception as e:
            logger.error(
                f"[WEBSOCKET_BROADCAST] Failed to serialize row for broadcast for rollout_id {rollout_id}: {e}"
            )

    async def _start_broadcast_loop(self):
        """Start the broadcast loop that processes queued messages."""
        logger.debug("[WEBSOCKET_BROADCAST_LOOP] Starting broadcast loop")
        while True:
            try:
                # Wait for a message to be queued
                logger.debug("[WEBSOCKET_BROADCAST_LOOP] Waiting for message from queue")
                message_data = await asyncio.get_event_loop().run_in_executor(None, self._broadcast_queue.get)
                logger.debug(
                    f"[WEBSOCKET_BROADCAST_LOOP] Retrieved message from queue (length: {len(str(message_data))})"
                )

                # Regular string message for all connections
                logger.debug("[WEBSOCKET_BROADCAST_LOOP] Sending message to all connections")
                await self._send_text_to_all_connections(str(message_data))
                logger.debug("[WEBSOCKET_BROADCAST_LOOP] Successfully sent message to all connections")

            except Exception as e:
                logger.error(f"[WEBSOCKET_BROADCAST_LOOP] Error in broadcast loop: {e}")
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                logger.info("[WEBSOCKET_BROADCAST_LOOP] Broadcast loop cancelled")
                break

    async def _send_text_to_all_connections(self, text: str):
        with self._lock:
            connections = list(self.active_connections)

        logger.debug(f"[WEBSOCKET_SEND] Attempting to send to {len(connections)} connections")

        if not connections:
            logger.debug("[WEBSOCKET_SEND] No connections available, skipping send")
            return

        tasks = []
        failed_connections = []

        for i, connection in enumerate(connections):
            try:
                logger.debug(f"[WEBSOCKET_SEND] Preparing to send to connection {i}")
                tasks.append(connection.send_text(text))
            except Exception as e:
                logger.error(f"[WEBSOCKET_SEND] Failed to prepare send to WebSocket {i}: {e}")
                failed_connections.append(connection)

        # Execute all sends in parallel
        if tasks:
            logger.debug(f"[WEBSOCKET_SEND] Executing {len(tasks)} parallel sends")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug("[WEBSOCKET_SEND] Completed parallel sends")

            # Check for any exceptions that occurred during execution
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"[WEBSOCKET_SEND] Failed to send text to WebSocket {i}: {result}")
                    failed_connections.append(connections[i])
                else:
                    logger.debug(f"[WEBSOCKET_SEND] Successfully sent to connection {i}")

        # Remove all failed connections
        if failed_connections:
            logger.debug(f"[WEBSOCKET_SEND] Removing {len(failed_connections)} failed connections")
            with self._lock:
                for connection in failed_connections:
                    try:
                        self.active_connections.remove(connection)
                    except ValueError:
                        pass

    def start_broadcast_loop(self):
        """Start the broadcast loop in the current event loop."""
        if self._broadcast_task is None or self._broadcast_task.done():
            logger.debug("[WEBSOCKET_BROADCAST_LOOP] Creating new broadcast task")
            self._broadcast_task = asyncio.create_task(self._start_broadcast_loop())
            logger.debug("[WEBSOCKET_BROADCAST_LOOP] Broadcast task created")
        else:
            logger.debug("[WEBSOCKET_BROADCAST_LOOP] Broadcast task already running")

    def stop_broadcast_loop(self):
        """Stop the broadcast loop."""
        if self._broadcast_task and not self._broadcast_task.done():
            logger.debug("[WEBSOCKET_BROADCAST_LOOP] Cancelling broadcast task")
            self._broadcast_task.cancel()
            self._broadcast_task = None
            logger.debug("[WEBSOCKET_BROADCAST_LOOP] Broadcast task cancelled")
        else:
            logger.debug("[WEBSOCKET_BROADCAST_LOOP] No active broadcast task to stop")


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

                    # Update eval_metadata.status if it's running
                    if row.eval_metadata and row.eval_metadata.status and row.eval_metadata.status.is_running():
                        row.eval_metadata.status = Status.aborted(
                            f"Evaluation aborted since process {row.pid} stopped"
                        )

                    # Update rollout_status if it's running
                    if row.rollout_status and row.rollout_status.is_running():
                        row.rollout_status = Status.aborted(f"Rollout aborted since process {row.pid} stopped")

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
        # Check if any status field should be updated
        return self._should_update_status_field(
            row.eval_metadata.status if row.eval_metadata else None, row.pid
        ) or self._should_update_status_field(row.rollout_status, row.pid)

    def _should_update_status_field(self, status: Optional["Status"], pid: Optional[int]) -> bool:
        """Check if a specific status field should be updated to 'stopped'."""
        # Check if the status is running and there's a PID
        if status and status.is_running() and pid is not None:
            # Check if the process is still running
            try:
                process = psutil.Process(pid)
                # Check if process is still running
                if not process.is_running():
                    return True
            except psutil.NoSuchProcess:
                # Process no longer exists
                return True
            except psutil.AccessDenied:
                # Can't access process info, assume it's stopped
                logger.warning(f"Access denied to process {pid}, assuming stopped")
                return True
            except Exception as e:
                logger.error(f"Error checking process {pid}: {e}")
                # On error, assume process is still running to be safe
                return False

        return False


class LogsServer(ViteServer):
    """
    Enhanced server for serving Vite-built SPA with file watching and WebSocket support.

    This server extends ViteServer to add:
    - WebSocket connections for real-time evaluation row updates
    - REST API for log querying
    """

    def __init__(
        self,
        build_dir: str = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vite-app", "dist")
        ),
        host: str = "localhost",
        port: Optional[int] = 8000,
        index_file: str = "index.html",
        elasticsearch_config: Optional[ElasticsearchConfig] = None,
        backend: Literal["fireworks", "elasticsearch"] = "elasticsearch",
        fireworks_base_url: Optional[str] = None,
        debug: bool = False,
    ):
        # Enable debug mode if requested
        if debug:
            enable_debug_mode()

        # Initialize WebSocket manager
        self.websocket_manager = WebSocketManager()

        # Backend selection
        self.backend: Literal["fireworks", "elasticsearch"] = backend
        self.fireworks_base_url = fireworks_base_url

        # Initialize Elasticsearch client if config is provided
        self.elasticsearch_client: Optional[ElasticsearchClient] = None
        if elasticsearch_config:
            self.elasticsearch_client = ElasticsearchClient(elasticsearch_config)

        self.app = FastAPI(title="Logs Server")

        # Add WebSocket endpoint and API routes
        self._setup_websocket_routes()
        self._setup_api_routes()

        super().__init__(build_dir, host, port if port is not None else 8000, index_file, self.app)

        # Add CORS middleware to allow frontend access
        allowed_origins = [
            "http://localhost:5173",  # Vite dev server
            "http://127.0.0.1:5173",  # Vite dev server (alternative)
            f"http://{host}:{port}",  # Server's own origin
            f"http://localhost:{port}",  # Server on localhost
        ]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize evaluation watcher
        self.evaluation_watcher = EvaluationWatcher(self.websocket_manager)

        # Log all registered routes for debugging
        logger.info("Registered routes:")
        for route in self.app.routes:
            path = getattr(route, "path", "UNKNOWN")
            methods = getattr(route, "methods", {"UNKNOWN"})
            logger.info(f"  {methods} {path}")

        # Subscribe to events and start listening for cross-process events
        logger.debug("[LOGS_SERVER_INIT] Subscribing to event bus")
        event_bus.subscribe(self._handle_event)
        logger.debug("[LOGS_SERVER_INIT] Successfully subscribed to event bus")

        logger.info(f"[LOGS_SERVER_INIT] LogsServer initialized on {self.host}:{self.port}")

    def _setup_websocket_routes(self):
        """Set up WebSocket routes for real-time communication."""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive (for evaluation row updates)
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.websocket_manager.disconnect(websocket)

    def _setup_api_routes(self):
        """Set up API routes."""

        @self.app.get("/api/status")
        async def status():
            """Get server status including active connections."""
            with self.websocket_manager._lock:
                active_connections_count = len(self.websocket_manager.active_connections)
            return {
                "status": "ok",
                "build_dir": str(self.build_dir),
                "active_connections": active_connections_count,
                # LogsServer inherits from ViteServer which doesn't expose watch_paths
                # Expose an empty list to satisfy consumers and type checker
                "watch_paths": [],
                "elasticsearch_enabled": self.elasticsearch_client is not None,
                "backend": self.backend,
                "fireworks_enabled": self.backend == "fireworks",
            }

        @self.app.get("/api/logs/{rollout_id}", response_model=LogsResponse, response_model_exclude_none=True)
        async def get_logs(
            rollout_id: str,
            level: Optional[str] = Query(None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR)"),
            limit: int = Query(100, description="Maximum number of log entries to return"),
        ) -> LogsResponse:
            """Get logs for a specific rollout ID from the configured backend."""
            # Fireworks backend
            if self.backend == "fireworks":
                try:
                    from eval_protocol.adapters.fireworks_tracing import FireworksTracingAdapter

                    base_url = self.fireworks_base_url or "https://tracing.fireworks.ai"
                    adapter = FireworksTracingAdapter(base_url=base_url)
                    # Fetch lightweight log entries filtered by rollout_id tag
                    tags = [f"rollout_id:{rollout_id}"]
                    entries = adapter.search_logs(tags=tags, limit=limit)
                    # Map to LogEntry responses
                    log_entries: List[LogEntry] = []
                    for e in entries:
                        ts = e.get("timestamp") or datetime.utcnow().isoformat() + "Z"
                        msg = e.get("message") or "trace"
                        sev = e.get("severity") or "INFO"
                        entry = LogEntry(
                            **{
                                "@timestamp": ts,
                                "level": sev,
                                "message": str(msg),
                                "logger_name": "fireworks",
                                "rollout_id": rollout_id,
                            }
                        )
                        log_entries.append(entry)

                    return LogsResponse(
                        logs=log_entries,
                        total=len(log_entries),
                        rollout_id=rollout_id,
                        filtered_by_level=level,
                    )
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Error retrieving Fireworks logs for rollout {rollout_id}: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to retrieve Fireworks logs: {str(e)}")

            # Elasticsearch backend
            if not self.elasticsearch_client:
                raise HTTPException(status_code=503, detail="Elasticsearch is not configured for this logs server")

            try:
                # Search for logs by rollout_id using a term filter (exact match),
                # sorted by timestamp desc with a secondary deterministic tie-breaker on _id desc
                sort_spec = [
                    {"@timestamp": {"order": "asc"}},
                ]
                query = {
                    "bool": {
                        "must": [
                            {"term": {"rollout_id": rollout_id}},
                        ]
                    }
                }
                search_results = self.elasticsearch_client.search(query, size=limit, sort=sort_spec)

                if not search_results or "hits" not in search_results:
                    # Return empty response using Pydantic model
                    return LogsResponse(
                        logs=[],
                        total=0,
                        rollout_id=rollout_id,
                        filtered_by_level=level,
                    )

                log_entries = []
                for hit in search_results["hits"]["hits"]:
                    log_data = hit["_source"]

                    # Filter by level if specified
                    if level and log_data.get("level") != level:
                        continue

                    # Create LogEntry using Pydantic model for validation
                    try:
                        log_entry = LogEntry(
                            **log_data  # Use ** to unpack the dict, Pydantic will handle field mapping
                        )
                        log_entries.append(log_entry)
                    except Exception as e:
                        # Log the error but continue processing other entries
                        logger.warning(f"Failed to parse log entry: {e}, data: {log_data}")
                        continue

                # Get total count
                total_hits = search_results["hits"]["total"]
                if isinstance(total_hits, dict):
                    # Elasticsearch 7+ format
                    total_count = total_hits["value"]
                else:
                    # Elasticsearch 6 format
                    total_count = total_hits

                # Return response using Pydantic model
                return LogsResponse(
                    logs=log_entries,
                    total=total_count,
                    rollout_id=rollout_id,
                    filtered_by_level=level,
                )

            except Exception as e:
                logger.error(f"Error retrieving logs for rollout {rollout_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")

    def _handle_event(self, event_type: str, data: Any) -> None:
        """Handle events from the event bus."""
        logger.debug(f"[EVENT_BUS_RECEIVE] Received event type: {event_type}")

        if event_type in [LOG_EVENT_TYPE]:
            from eval_protocol.models import EvaluationRow

            try:
                logger.debug("[EVENT_BUS_RECEIVE] Processing LOG_EVENT_TYPE event")
                data = EvaluationRow(**data)
                rollout_id = data.execution_metadata.rollout_id if data.execution_metadata else "unknown"
                logger.debug(f"[EVENT_BUS_RECEIVE] Successfully parsed EvaluationRow for rollout_id: {rollout_id}")

                logger.debug("[EVENT_BUS_RECEIVE] Broadcasting row_upserted to websocket manager")
                self.websocket_manager.broadcast_row_upserted(data)
                logger.debug(f"[EVENT_BUS_RECEIVE] Successfully queued broadcast for rollout_id: {rollout_id}")
            except Exception as e:
                logger.error(f"[EVENT_BUS_RECEIVE] Failed to process LOG_EVENT_TYPE event: {e}")
        else:
            logger.debug(f"[EVENT_BUS_RECEIVE] Ignoring event type: {event_type} (not LOG_EVENT_TYPE)")

    def start_loops(self):
        """Start the broadcast loop and evaluation watcher."""
        logger.debug("[LOGS_SERVER_LOOPS] Starting all loops")
        self.websocket_manager.start_broadcast_loop()
        logger.debug("[LOGS_SERVER_LOOPS] Started websocket broadcast loop")
        self.evaluation_watcher.start()
        logger.debug("[LOGS_SERVER_LOOPS] Started evaluation watcher")
        event_bus.start_listening()
        logger.debug("[LOGS_SERVER_LOOPS] Started event bus listening")

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

            # Write PID file after server is configured but before serving
            logger.debug(f"[LOGS_SERVER_RUN_ASYNC] Writing PID file for port {self.port}")
            write_pid_file(os.getpid(), self.port)
            logger.debug(f"[LOGS_SERVER_RUN_ASYNC] Successfully wrote PID file for port {self.port}")

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


def create_app(
    host: str = "localhost",
    port: int = 8000,
    build_dir: Optional[str] = None,
    elasticsearch_config: Optional[ElasticsearchConfig] = None,
    backend: Literal["fireworks", "elasticsearch"] = "elasticsearch",
    fireworks_base_url: Optional[str] = None,
    debug: bool = False,
) -> FastAPI:
    """
    Factory function to create a FastAPI app instance and start the server with async loops.

    This creates a LogsServer instance and starts it in a background thread to ensure
    all async loops (WebSocket broadcast, evaluation watching) are running.

    Args:
        host: Host to bind to
        port: Port to bind to
        build_dir: Optional custom build directory path
        elasticsearch_config: Optional Elasticsearch configuration for log querying

    Returns:
        FastAPI app instance with server running in background
    """
    if build_dir is None:
        build_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vite-app", "dist")
        )

    server = LogsServer(
        host=host,
        port=port,
        build_dir=build_dir,
        elasticsearch_config=elasticsearch_config,
        backend=backend,
        fireworks_base_url=fireworks_base_url,
        debug=debug,
    )
    server.start_loops()
    return server.app


# For backward compatibility and direct usage
def serve_logs(
    port: Optional[int] = None,
    elasticsearch_config: Optional[ElasticsearchConfig] = None,
    debug: bool = False,
    backend: Literal["fireworks", "elasticsearch"] = "elasticsearch",
    fireworks_base_url: Optional[str] = None,
):
    """
    Convenience function to create and run a LogsServer.
    """
    # For backward compatibility with tests that assert exact constructor kwargs,
    # only pass additional backend-related kwargs when they are actually needed.
    logs_server_kwargs: Dict[str, Any] = {
        "port": port,
        "elasticsearch_config": elasticsearch_config,
        "debug": debug,
    }

    # If non-default backend (fireworks) is requested or a base URL is provided, include them.
    if backend != "elasticsearch" or fireworks_base_url is not None:
        logs_server_kwargs["backend"] = backend
        logs_server_kwargs["fireworks_base_url"] = fireworks_base_url

    server = LogsServer(**logs_server_kwargs)
    server.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the evaluation logs server")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--build-dir", help="Path to Vite build directory")
    parser.add_argument("--debug", help="Set logger level to DEBUG")

    args = parser.parse_args()

    if args.debug:
        enable_debug_mode()

    elasticsearch_config = ElasticsearchSetup().setup_elasticsearch()

    # Create server with command line arguments
    if args.build_dir:
        server = LogsServer(
            host=args.host,
            port=args.port,
            build_dir=args.build_dir,
            elasticsearch_config=elasticsearch_config,
            debug=bool(args.debug),
        )
    else:
        server = LogsServer(
            host=args.host, port=args.port, elasticsearch_config=elasticsearch_config, debug=bool(args.debug)
        )

    server.run()
