import asyncio
import atexit
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import eval_protocol as ep
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.types import RolloutProcessorConfig


class MCPServerManager:
    """Manages MCP server lifecycle for testing."""

    # Class-level tracking of all server instances
    _active_servers = []
    _cleanup_registered = False

    def __init__(self, server_script: str, port: int = 8000, domain: str = "airline"):
        self.server_script = server_script
        self.port = port
        self.domain = domain
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

        # Start server process (no domain argument needed for tau2_mcp server)
        cmd = ["python", self.server_script, "--port", str(self.port)]

        # Setup log file with cleanup
        log_file_path = os.path.join(self.base_dir, f"server_output_{self.domain}_{self.port}.log")
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
                print(f"❌ Server failed to start!")
                print(f"📋 Server log ({self._log_file_path}):")
                print("=" * 50)
                print(log_content)
                print("=" * 50)
                raise RuntimeError(f"Server failed to start or become ready. Check log above for details.")
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
            if self.process.poll() is not None:
                print(f"Server process exited early")
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


async def default_mcp_gym_rollout_processor(
    rows: List[EvaluationRow], config: RolloutProcessorConfig
) -> List[EvaluationRow]:
    """
    Rollout processor for tau bench environments.


    This processor starts an MCP server, creates tau bench environments, and runs rollouts
    using the eval_protocol framework, following the pattern from test_tau2_e2e.py.


    Args:
        rows: List of EvaluationRow objects containing messages and dataset info in input_metadata
        config: RolloutProcessorConfig with model and other parameters


    Returns:
        List of EvaluationRow objects with completed conversations
    """
    if config.server_script_path is None:
        raise ValueError("server_script_path is required for default_mcp_gym_rollout_processor")
    server = MCPServerManager(config.server_script_path, port=9700)

    try:
        server.start()

        policy = ep.LiteLLMPolicy(
            model_id=config.model,
            temperature=config.input_params.get("temperature", 0.0),
            max_tokens=config.input_params.get("max_tokens", 4096),
            reasoning_effort=config.input_params.get("reasoning_effort", None),
        )

        # Create MCP environments directly from evaluation_rows
        envs = ep.make(
            "http://localhost:9700/mcp/",
            evaluation_rows=rows,
            model_id=policy.model_id,
        )

        # Run rollout with environments and policy
        evaluation_rows = await ep.rollout(
            envs,
            policy=policy,
            evaluation_rows=rows,
            steps=config.steps,
            max_concurrent_rollouts=config.max_concurrent_rollouts,
        )

        return evaluation_rows

    finally:
        # Always clean up the server
        server.stop()
