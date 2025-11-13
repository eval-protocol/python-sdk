import asyncio
import atexit
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

import eval_protocol as ep
from eval_protocol.mcp.execution.manager import ExecutionManager
from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig, ServerMode


class MCPServerManager:
    """Manages MCP server lifecycle for testing."""

    # Class-level tracking of all server instances
    _active_servers = []
    _cleanup_registered = False

    def __init__(self, server_script: str, port: int = 8000, **kwargs):
        self.server_script = server_script
        self.port = port
        self.domain = kwargs.get("domain", None)
        self.process: Optional[subprocess.Popen] = None
        self.base_dir = Path(".").resolve()
        self._log_file = None
        self._log_file_path = None

        # Register this server for cleanup
        MCPServerManager._active_servers.append(self)

        # Register cleanup handlers only once
        if not MCPServerManager._cleanup_registered:
            MCPServerManager._register_cleanup_handlers()
            MCPServerManager._cleanup_registered = True

    def start(self) -> None:
        """Start the MCP server."""
        if self.process:
            return

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(("localhost", self.port))
                if result == 0:
                    raise RuntimeError(
                        f"Port {self.port} is already in use! Please use a different port or kill the process using it."
                    )
        except socket.error:
            pass

        # Set environment for server
        env = os.environ.copy()
        env["PORT"] = str(self.port)

        # Build command, add --domain only if provided (e.g. tau2 needs it, frozen_lake doesn't)
        cmd = ["python", self.server_script, "--port", str(self.port)]
        if self.domain:
            cmd.extend(["--domain", self.domain])

        # Setup log file with cleanup
        domain_part = self.domain if self.domain else "server"
        log_file_path = os.path.join(self.base_dir, f"server_output_{domain_part}_{self.port}.log")
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

        log_file = open(log_file_path, "w")

        self.process = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            env=env,
            stdout=log_file,
            stderr=log_file,
            text=True,
        )

        # Store log file reference for cleanup
        self._log_file = log_file
        self._log_file_path = log_file_path

        # Wait for server to be ready with proper health check
        if not self._wait_for_server_ready(timeout=15):
            try:
                with open(self._log_file_path, "r") as f:
                    log_content = f.read()
                print("❌ Server failed to start!")
                print(f"📋 Server log ({self._log_file_path}):")
                print("=" * 50)
                print(log_content)
                print("=" * 50)
                raise RuntimeError("Server failed to start or become ready. Check log above for details.")
            except Exception as e:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(f"Server failed to start or become ready. stderr: {stderr}, log error: {e}")

        print(f"✅ Server started successfully on port {self.port}")

    def _wait_for_server_ready(self, timeout: int = 15) -> bool:
        """
        Wait for server to be ready by polling socket connection.
        """
        start_time = time.time()
        health_check_failures = 0

        while time.time() - start_time < timeout:
            # Check if process is still running
            if self.process and self.process.poll() is not None:
                print("Server process exited early")
                return False

            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(("localhost", self.port))
                    if result == 0:
                        time.sleep(0.5)
                        return True
            except Exception as e:
                health_check_failures += 1
                # Print first few failures for debugging
                if health_check_failures <= 3:
                    print(f"Health check failed: {e}")

            # Wait before next check
            time.sleep(0.1)

        print(f"Server failed to become ready within {timeout} seconds")
        return False

    def stop(self) -> None:
        """Stop the MCP server."""
        if self.process:
            print(f"🛑 Stopping server on port {self.port}...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"⚡ Force killing server on port {self.port}...")
                self.process.kill()
                self.process.wait()
            self.process = None

        # Clean up log file
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

        # Remove from active servers list
        if self in MCPServerManager._active_servers:
            MCPServerManager._active_servers.remove(self)

    @classmethod
    def _cleanup_all_servers(cls):
        """Clean up all active servers on exit"""
        print(f"\n🧹 Cleaning up {len(cls._active_servers)} active servers...")
        for server in cls._active_servers.copy():
            try:
                server.stop()
            except Exception as e:
                print(f"⚠️  Error stopping server: {e}")
        cls._active_servers.clear()

    @classmethod
    def _signal_handler(cls, signum, frame):
        """Handle interrupt signals"""
        print(f"\n🛑 Received signal {signum}, cleaning up...")
        cls._cleanup_all_servers()
        exit(1)

    @classmethod
    def _register_cleanup_handlers(cls):
        """Register cleanup handlers - called only once"""
        atexit.register(cls._cleanup_all_servers)
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, cls._signal_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, cls._signal_handler)  # Termination signal

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup even on exceptions"""
        self.stop()
        if exc_type:
            print(f"⚠️  Server cleanup after exception: {exc_type.__name__}")
        return False  # Don't suppress exceptions


class MCPGymRolloutProcessor(RolloutProcessor):
    """
    Rollout processor for MCP gym environments.

    This processor starts an MCP server, creates an environment, and returns rollout tasks
    using the eval_protocol framework with proper cleanup handling.
    """

    # Shared server state for "shared" mode
    _shared_server_lock = threading.Lock()
    _shared_server: Optional[MCPServerManager] = None
    _shared_server_started: bool = False

    def __init__(self):
        # Instance-level server handle (used in "per_run" mode)
        self.server: Optional[MCPServerManager] = None
        self.policy = None
        # Track which mode this instance last used ("per_run" or "shared")
        self.server_mode: ServerMode = "per_run"

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Process evaluation rows with MCP gym environments."""
        server_kwargs = dict(config.kwargs or {})
        start_server = bool(server_kwargs.pop("start_server", True))
        server_mode: ServerMode = server_kwargs.pop("server_mode", "per_run")
        port = int(server_kwargs.pop("port", 9700))

        self.server_mode = server_mode

        if server_mode == "shared":
            # Shared, class-level server used across calls
            if start_server and not MCPGymRolloutProcessor._shared_server_started:
                with MCPGymRolloutProcessor._shared_server_lock:
                    if not MCPGymRolloutProcessor._shared_server_started:
                        if config.server_script_path is None:
                            raise ValueError("server_script_path is required for MCPGymRolloutProcessor")

                        shared_server = MCPServerManager(config.server_script_path, port=port, **server_kwargs)

                        try:
                            shared_server.start()
                        except Exception as e:
                            shared_server.stop()
                            raise e

                        MCPGymRolloutProcessor._shared_server = shared_server
                        MCPGymRolloutProcessor._shared_server_started = True

            if MCPGymRolloutProcessor._shared_server is None:
                raise RuntimeError(
                    "Shared MCP server not started. Call with server_mode='shared' and start_server=True first."
                )
            # Bind this instance to the shared server for this call
            self.server = MCPGymRolloutProcessor._shared_server

        else:
            # Default "per_run" behavior: fresh server per call, reused only for retries
            if start_server:
                # Create fresh MCP server and environments for this run
                if config.server_script_path is None:
                    raise ValueError("server_script_path is required for MCPGymRolloutProcessor")

                self.server = MCPServerManager(config.server_script_path, port=port, **server_kwargs)

                try:
                    self.server.start()

                except Exception as e:
                    if self.server:
                        self.server.stop()
                    self.server = None
                    self.policy = None
                    raise e

            else:
                # Reuse existing MCP environments for retry (per_run mode)
                if not self.server:
                    raise RuntimeError(
                        "Cannot retry without existing server/environments. Call with start_server=True first."
                    )

        model_id = str((config.completion_params.get("model") if config.completion_params else None) or "gpt-4o-mini")
        temperature = config.completion_params.get("temperature", 0.0)
        max_tokens = config.completion_params.get("max_tokens", 4096)

        # Pass all other completion_params (e.g. stream=True) via kwargs
        other_params = {
            k: v
            for k, v in (config.completion_params or {}).items()
            if k not in ["model", "temperature", "max_tokens", "extra_body"]
        }
        extra_body = config.completion_params.get("extra_body", {}) or {}

        self.policy = ep.LiteLLMPolicy(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_body,
            **other_params,
        )
        # Create MCP environments directly from evaluation_rows
        envs = ep.make(
            f"http://localhost:{port}/mcp/",
            evaluation_rows=rows,
            model_id=self.policy.model_id,
        )

        # TODO: chat with benny/dylan about when they're back. can we just bypass ep.rollout()? i don't really see the point of it anymore. or turn it into a return list of tasks.
        execution_manager = ExecutionManager()
        tasks = execution_manager.execute_rollouts(
            envs,
            policy=self.policy,
            semaphore=config.semaphore,
            steps=config.steps,
            evaluation_rows=rows,
        )
        return tasks

    def cleanup(self) -> None:
        """Cleanup MCP server and environments."""
        # For shared mode, don't stop the shared server here; rely on global cleanup
        # (atexit or an explicit class-level shutdown) so multiple users can share it.
        if self.server_mode == "shared":
            self.policy = None
            return

        # Per-run mode: stop this instance's server
        if self.server:
            self.server.stop()
            self.server = None
            self.policy = None
