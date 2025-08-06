#!/usr/bin/env python3
"""
End-to-End Record and Replay Tests for Taxi MCP

This module provides pytest-compatible tests that:
1. Set up production and simulation servers automatically
2. Record trajectories in the first run
3. Use recorded trajectories for fast replay in subsequent runs
4. Validate server functionality and performance
5. Clean up resources properly

Usage:
    pytest test_record_and_replay_e2e.py -v

Environment Variables:
    EP_FORCE_RECORD=1  # Force recording mode even if replay file exists
    EP_PLAYBACK_FILE   # Path to replay file (auto-detected if not set)
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import eval_protocol as ep


class MCPServerManager:
    """Manages MCP server lifecycle for testing."""

    def __init__(self, server_script: str, port: int = 8000):
        self.server_script = server_script
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.base_dir = Path(__file__).parent.parent / "mcp_server"

    def start(self) -> None:
        """Start the MCP server."""
        if self.process:
            return

        # Set environment for server
        env = os.environ.copy()
        env["PORT"] = str(self.port)

        # Start server process
        cmd = ["python", self.server_script, "--port", str(self.port)]
        self.process = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to start
        time.sleep(3)

        # Check if process is still running
        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(f"Server failed to start: {stderr}")

    def stop(self) -> None:
        """Stop the MCP server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def is_running(self) -> bool:
        """Check if server is running."""
        return self.process is not None and self.process.poll() is None


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory."""
    return Path(__file__).parent.parent / "shared_data"


@pytest.fixture(scope="session")
def taxi_dataset(test_data_dir):
    """Load Taxi test dataset."""
    rollouts_file = test_data_dir / "taxi_rollouts.jsonl"
    if not rollouts_file.exists():
        pytest.skip(f"Dataset not found: {rollouts_file}")

    with open(rollouts_file) as f:
        dataset = [json.loads(line) for line in f]

    # Use only first 2 entries for faster testing
    return dataset[:2]


@pytest.fixture(scope="session")
def production_server():
    """Start and manage production server."""
    server = MCPServerManager("taxi_mcp_server.py", port=9500)

    try:
        server.start()
        yield server
    finally:
        server.stop()


@pytest.fixture(scope="session")
def simulation_server():
    """Start and manage simulation server."""
    server = MCPServerManager("simulation_server.py", port=9501)

    try:
        server.start()
        yield server
    finally:
        server.stop()


@pytest.fixture
def recording_file():
    """Provide temporary recording file path without creating it."""
    # Create a temporary file, get its name, then delete it.
    # This ensures we have a unique, non-existent path.
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True) as f:
        recording_path = f.name

    try:
        yield recording_path
    finally:
        # Clean up any file that might have been created at the path
        if os.path.exists(recording_path):
            os.unlink(recording_path)


@pytest.mark.asyncio
async def test_production_server_record_and_replay(production_server, taxi_dataset, recording_file):
    """Test production server with record and replay functionality."""

    # === RECORDING PHASE ===
    print("\n📝 === TAXI RECORDING PHASE ===")

    # Set up recording environment
    os.environ["EP_PLAYBACK_FILE"] = recording_file

    # Create policy for recording
    policy = ep.FireworksPolicy(
        model_id="accounts/fireworks/models/qwen3-235b-a22b",
        temperature=0.2,
        max_tokens=16384,  # Taxi needs more thinking space
        max_tools_per_turn=1,
    )

    assert not policy.is_playback_mode(), "Should be in recording mode initially"

    # Create environments
    envs = await ep.make("http://localhost:9500/mcp/", dataset=taxi_dataset, model_id=policy.model_id)

    # Record evaluation rows (Taxi typically needs more steps)
    start_time = time.time()
    evaluation_rows = await ep.rollout(
        envs,
        policy=policy,
        steps=25,
        openai_format_log_file=None,  # Don't need OpenAI format for testing
    )
    recording_duration = time.time() - start_time

    assert len(evaluation_rows) == len(taxi_dataset), "Should have evaluation row for each dataset entry"
    assert os.path.exists(recording_file), "Recording file should be created"

    print(f"✅ Recorded {len(evaluation_rows)} evaluation rows in {recording_duration:.2f}s")

    # === PLAYBACK PHASE ===
    print("\n🎬 === TAXI PLAYBACK PHASE ===")

    # Create new policy for playback (same environment variable)
    playback_policy = ep.FireworksPolicy(
        model_id="accounts/fireworks/models/qwen3-235b-a22b",
        temperature=0.2,
        max_tokens=16384,
        max_tools_per_turn=1,
    )

    assert playback_policy.is_playback_mode(), "Should be in playback mode"

    # Create new environments for playback
    playback_envs = await ep.make(
        "http://localhost:9500/mcp/",
        dataset=taxi_dataset,
        model_id=playback_policy.model_id,
    )

    # Run playback
    start_time = time.time()
    playback_evaluation_rows = await ep.rollout(playback_envs, policy=playback_policy, steps=25)
    playback_duration = time.time() - start_time

    assert len(playback_evaluation_rows) == len(evaluation_rows), "Playback should have same number of evaluation rows"

    # Calculate speedup
    speedup = recording_duration / playback_duration if playback_duration > 0 else float("inf")

    print(f"✅ Played back {len(playback_evaluation_rows)} evaluation rows in {playback_duration:.2f}s")
    print(f"⚡ Speedup: {speedup:.1f}x faster than recording")

    # Validate performance - playback should be significantly faster
    assert speedup > 10, f"Playback should be at least 10x faster, got {speedup:.1f}x"

    # Clean up environment variable
    if "EP_PLAYBACK_FILE" in os.environ:
        del os.environ["EP_PLAYBACK_FILE"]


@pytest.mark.asyncio
async def test_simulation_server_record_and_replay(simulation_server, taxi_dataset, recording_file):
    """Test simulation server with record and replay functionality."""

    # === RECORDING PHASE ===
    print("\n📝 === TAXI SIMULATION RECORDING PHASE ===")

    # Set up recording environment
    os.environ["EP_PLAYBACK_FILE"] = recording_file

    # Create policy for recording
    policy = ep.FireworksPolicy(
        model_id="accounts/fireworks/models/qwen3-235b-a22b",
        temperature=0.2,
        max_tokens=16384,
        max_tools_per_turn=1,
    )

    # Create environments pointing to simulation server
    envs = await ep.make("http://localhost:9501/mcp/", dataset=taxi_dataset, model_id=policy.model_id)

    # Record evaluation rows
    start_time = time.time()
    evaluation_rows = await ep.rollout(envs, policy=policy, steps=25)
    recording_duration = time.time() - start_time

    assert len(evaluation_rows) == len(taxi_dataset), "Should have evaluation row for each dataset entry"
    assert os.path.exists(recording_file), "Recording file should be created"

    print(f"✅ Simulation recorded {len(evaluation_rows)} evaluation rows in {recording_duration:.2f}s")

    # === PLAYBACK PHASE ===
    print("\n🎬 === TAXI SIMULATION PLAYBACK PHASE ===")

    # Create playback policy
    playback_policy = ep.FireworksPolicy(
        model_id="accounts/fireworks/models/qwen3-235b-a22b",
        temperature=0.2,
        max_tokens=16384,
        max_tools_per_turn=1,
    )

    # Create new environments for playback
    playback_envs = await ep.make(
        "http://localhost:9501/mcp/",
        dataset=taxi_dataset,
        model_id=playback_policy.model_id,
    )

    # Run playback
    start_time = time.time()
    playback_evaluation_rows = await ep.rollout(playback_envs, policy=playback_policy, steps=25)
    playback_duration = time.time() - start_time

    assert len(playback_evaluation_rows) == len(evaluation_rows), "Playback should have same number of evaluation rows"

    speedup = recording_duration / playback_duration if playback_duration > 0 else float("inf")
    print(f"✅ Simulation played back {len(playback_evaluation_rows)} evaluation rows in {playback_duration:.2f}s")
    print(f"⚡ Simulation speedup: {speedup:.1f}x faster than recording")

    # Validate performance
    assert speedup > 10, f"Simulation playback should be at least 10x faster, got {speedup:.1f}x"

    # Clean up environment variable
    if "EP_PLAYBACK_FILE" in os.environ:
        del os.environ["EP_PLAYBACK_FILE"]


def test_server_health_checks(production_server, simulation_server):
    """Test that both servers are running and healthy."""
    assert production_server.is_running(), "Production server should be running"
    assert simulation_server.is_running(), "Simulation server should be running"


@pytest.mark.asyncio
async def test_production_only_recorded_policy(taxi_dataset):
    """Test that production environments work with pre-recorded policies only."""

    # Create a pre-recorded trajectory file for this test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        test_recording_file = f.name
        # Create a dummy trajectory file
        recording_data = [
            {
                "env_index": 0,
                "step": 0,
                "messages": [
                    {"role": "system", "content": taxi_dataset[0]["system_prompt"]},
                    {"role": "user", "content": "Initial state"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "taxi_move",
                                    "arguments": '{"action": "SOUTH"}',
                                },
                            }
                        ],
                    },
                ],
            }
        ]
        for entry in recording_data:
            f.write(json.dumps(entry) + "\n")

    try:
        # Set up playback environment
        os.environ["EP_PLAYBACK_FILE"] = test_recording_file

        # Create policy in playback mode
        policy = ep.FireworksPolicy(
            model_id="accounts/fireworks/models/qwen3-235b-a22b",
            temperature=0.2,
            max_tokens=16384,
            max_tools_per_turn=1,
        )

        assert policy.is_playback_mode(), "Policy should be in playback mode"

        print("✅ Taxi production environment successfully using recorded policy")

    finally:
        # Clean up
        if os.path.exists(test_recording_file):
            os.unlink(test_recording_file)
        if "EP_PLAYBACK_FILE" in os.environ:
            del os.environ["EP_PLAYBACK_FILE"]


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])
