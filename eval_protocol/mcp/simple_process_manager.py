"""
Simplified process manager for MCP servers running in separate processes.

This module provides a simpler alternative to the conda-based process manager
for testing and development scenarios where full environment isolation is not required.
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
import uuid
from contextlib import AsyncExitStack
from typing import Dict, Tuple, Optional

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Implementation


class SimpleServerProcessManager:
    """Manages the lifecycle of server subprocesses using the current Python environment."""

    def __init__(
        self,
        script_path: str,
        python_executable: Optional[str] = None,
        port_range: Tuple[int, int] = (10000, 11000),
    ):
        """
        Initialize the process manager.

        Args:
            script_path: Path to the server script to run
            python_executable: Python executable to use (defaults to current Python)
            port_range: Tuple of (min_port, max_port) for server instances
        """
        self.script_path = script_path
        self.python_executable = python_executable or sys.executable
        self.port_range = port_range
        self.processes: Dict[int, Tuple[subprocess.Popen, str]] = {}  # port -> (process, instance_id)
        self.used_ports: set = set()  # Track used ports for better management

    def find_free_port(self) -> int:
        """
        Finds and returns an available TCP port within the configured range.

        Returns:
            Available port number

        Raises:
            RuntimeError: If no ports are available in the range
        """
        min_port, max_port = self.port_range

        # Try ports in the configured range, avoiding recently used ones
        attempted_ports = set()

        for _ in range(max_port - min_port):
            # Generate a candidate port, preferring unused ones
            import random

            # First try unused ports
            available_ports = set(range(min_port, max_port)) - self.used_ports
            if available_ports:
                candidate_port = random.choice(list(available_ports))
            else:
                # If all ports have been used, try any port in range
                candidate_port = random.randint(min_port, max_port - 1)

            # Skip if we already tried this port
            if candidate_port in attempted_ports:
                continue
            attempted_ports.add(candidate_port)

            # Test if the port is actually available
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("localhost", candidate_port))
                    # Port is available
                    self.used_ports.add(candidate_port)
                    print(f"Allocated port {candidate_port} from range {min_port}-{max_port}")
                    return candidate_port
            except OSError:
                # Port is in use, try next one
                continue

        # No available ports found
        raise RuntimeError(f"No available ports in range {min_port}-{max_port}. Used ports: {len(self.used_ports)}")

    def start_server(self, seed: int) -> int:
        """Starts a server instance with the given seed."""
        port = self.find_free_port()
        instance_id = f"simple-server-{uuid.uuid4().hex[:8]}"

        print(f"Starting server instance '{instance_id}' on port {port} with seed {seed}")

        env = os.environ.copy()
        env["PORT"] = str(port)

        # Command to run the server with the current Python environment
        cmd = [
            self.python_executable,
            self.script_path,
            "--port",
            str(port),
            "--seed",
            str(seed),
        ]

        # Start the process with visible output for debugging
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Keep stderr separate to see error output
            text=True,
            env=env,
        )

        self.processes[port] = (process, instance_id)

        # Wait for server to be ready with health check polling
        if not self._wait_for_server_ready(port, instance_id, process):
            # Clean up failed process
            if port in self.processes:
                del self.processes[port]
            raise RuntimeError(f"Server instance '{instance_id}' failed to start or become ready")

        print(f"Server instance '{instance_id}' started successfully on port {port}")
        return port

    def _wait_for_server_ready(
        self, port: int, instance_id: str, process: subprocess.Popen, timeout: int = 15
    ) -> bool:
        """
        Wait for server to be ready by polling MCP health check.

        Args:
            port: Server port
            instance_id: Server instance ID for logging
            process: Server process
            timeout: Maximum time to wait in seconds

        Returns:
            True if server is ready, False otherwise
        """
        start_time = time.time()
        health_check_failures = 0  # Fix: Initialize counter properly

        while time.time() - start_time < timeout:
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"Server instance '{instance_id}' process exited early")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False

            # Try simple socket check instead of full MCP health check
            try:
                # Simple TCP socket check first
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(("localhost", port))
                    if result == 0:
                        # Port is open, server is likely ready
                        return True
            except Exception as e:
                health_check_failures += 1
                # Print first few failures for debugging
                if health_check_failures <= 3:
                    print(f"Health check failed for instance '{instance_id}': {e}")

            # Wait before next check
            time.sleep(0.5)

        print(f"Server instance '{instance_id}' failed to become ready within {timeout} seconds")
        return False

    async def _check_mcp_health(self, port: int, instance_id: str) -> bool:
        """
        Check if MCP server is responding to requests.

        Args:
            port: Server port
            instance_id: Server instance ID for logging

        Returns:
            True if MCP server is responding, False otherwise
        """
        try:
            # Fix: Use proper MCP server URL with /mcp/ path
            server_url = f"http://localhost:{port}/mcp/"

            # Use asyncio timeout to prevent hanging (compatible with older Python versions)
            try:
                await asyncio.wait_for(self._do_health_check(server_url), timeout=5.0)
                return True
            except asyncio.TimeoutError:
                return False

        except Exception as e:
            # Reduce verbosity - only show critical connection errors
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["connection", "refused", "timeout", "unreachable"]):
                # Connection errors are normal during startup
                return False
            else:
                print(f"MCP health check error for instance '{instance_id}' on port {port}: {e}")
                return False

    async def _do_health_check(self, server_url: str) -> bool:
        """Perform the actual health check."""
        try:
            async with AsyncExitStack() as exit_stack:
                # Connect to the MCP server with shorter timeout for health checks
                read_stream, write_stream, _ = await exit_stack.enter_async_context(
                    streamablehttp_client(server_url, terminate_on_close=True)
                )

                client_info = Implementation(name="health-check", version="1.0.0")
                mcp_client = await exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream, client_info=client_info)
                )
                await mcp_client.initialize()

                # Try to list tools - this should be available for all MCP servers
                result = await mcp_client.list_tools()
                return True  # If we got here, MCP server is responding
        except Exception:
            return False
        return False  # This should never be reached, but added for mypy

    def stop_server(self, port: int) -> None:
        """Stops the server instance and verifies port cleanup."""
        if port in self.processes:
            process, instance_id = self.processes[port]
            print(f"Stopping server instance '{instance_id}' on port {port}")

            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Force killing server instance '{instance_id}'")
                process.kill()
                process.wait()

            # Verify port is actually freed
            if self._verify_port_freed(port):
                print(f"✅ Port {port} successfully freed")
            else:
                print(f"⚠️ Warning: Port {port} may still be in use after server stop")

            # Clean up tracking
            del self.processes[port]
            if port in self.used_ports:
                self.used_ports.remove(port)

            print(f"Server instance '{instance_id}' stopped and cleaned up")

    def _verify_port_freed(self, port: int, max_retries: int = 3) -> bool:
        """
        Verify that a port is actually freed after stopping a server.

        Args:
            port: The port to check
            max_retries: Number of times to retry the check

        Returns:
            True if port is freed, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # Try to bind to the port - if successful, it's free
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("localhost", port))
                    return True
            except OSError:
                # Port still in use, wait a bit and retry
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                continue

        return False

    def stop_all(self) -> None:
        """Stops all managed servers."""
        for port in list(self.processes.keys()):
            self.stop_server(port)
