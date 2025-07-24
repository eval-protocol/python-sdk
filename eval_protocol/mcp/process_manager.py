"""
Generic process manager for MCP servers running in isolated Conda environments.

This module provides a reusable helper class to manage the lifecycle of server
subprocesses within dedicated Conda environments, ensuring dependency isolation.
"""

import os
import socket
import subprocess
import time
import uuid
from typing import Dict, Tuple


class CondaServerProcessManager:
    """Manages the lifecycle of server subprocesses inside Conda environments."""

    def __init__(
        self,
        script_path: str,
        requirements_path: str,
        conda_base_env: str = "base",
        port_range: Tuple[int, int] = (10000, 11000),
    ):
        """
        Initialize the process manager.

        Args:
            script_path: Path to the server script to run
            requirements_path: Path to requirements.txt for the environment
            conda_base_env: Base conda environment to clone from
            port_range: Tuple of (min_port, max_port) for server instances
        """
        self.script_path = script_path
        self.requirements_path = requirements_path
        self.conda_base_env = conda_base_env
        self.port_range = port_range
        self.processes: Dict[int, Tuple[subprocess.Popen, str]] = {}  # port -> (process, conda_env_name)
        self.used_ports: set = set()  # Track used ports for better management

    def _create_conda_env(self, env_name: str):
        """Creates a new conda environment by cloning the base."""
        print(f"Creating conda environment '{env_name}'...")
        # Clone the base environment
        clone_cmd = [
            "conda",
            "create",
            "--name",
            env_name,
            "--clone",
            self.conda_base_env,
            "-y",
        ]
        subprocess.run(clone_cmd, check=True, capture_output=True, text=True)

        # Install specific requirements into the new environment
        pip_install_cmd = [
            "conda",
            "run",
            "-n",
            env_name,
            "pip",
            "install",
            "-r",
            self.requirements_path,
        ]
        subprocess.run(pip_install_cmd, check=True, capture_output=True, text=True)
        print(f"Environment '{env_name}' created and dependencies installed.")

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
        """Creates a new Conda env and starts a server instance within it."""
        port = self.find_free_port()
        env_name = f"mcp-sim-env-{uuid.uuid4().hex[:8]}"

        self._create_conda_env(env_name)

        env = os.environ.copy()
        env["PORT"] = str(port)

        # Command to run the server inside the new conda environment
        cmd = [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            self.script_path,
            "--port",
            str(port),
            "--seed",
            str(seed),
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self.processes[port] = (process, env_name)
        time.sleep(3)  # Give the server more time to start up after env creation
        return port

    def stop_server(self, port: int):
        """Stops the server and removes its Conda environment."""
        if port in self.processes:
            process, env_name = self.processes[port]
            print(f"Stopping server on port {port} and cleaning up environment '{env_name}'")

            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Force killing server on port {port}")
                process.kill()
                process.wait()

            # Remove the conda environment
            print(f"Removing conda environment '{env_name}'...")
            rm_cmd = ["conda", "env", "remove", "--name", env_name, "-y"]
            subprocess.run(rm_cmd, check=True, capture_output=True, text=True)

            # Clean up tracking
            del self.processes[port]
            if port in self.used_ports:
                self.used_ports.remove(port)

            print(f"✅ Environment '{env_name}' removed and port {port} freed")

    def stop_all(self):
        """Stops all managed servers and cleans up all environments."""
        for port in list(self.processes.keys()):
            self.stop_server(port)
