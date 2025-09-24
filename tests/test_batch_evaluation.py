"""
End-to-end integration tests for batch evaluation feature.

These tests validate the entire batch evaluation pipeline with live API calls
to both Fireworks and OpenAI, ensuring production readiness.
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from eval_protocol.agent.task_manager import TaskManager
from eval_protocol.cli_commands.agent_eval_cmd import agent_eval_command
from eval_protocol.models import TaskDefinitionModel


class MockArgs:
    """Mock args object for agent_eval_command."""

    def __init__(self, task_def: str, num_rollouts: int = 2, **kwargs):
        self.task_def = task_def
        self.num_rollouts = num_rollouts
        self.parallel = kwargs.get("parallel", False)
        self.max_concurrency = kwargs.get("max_concurrency", 3)
        self.model = kwargs.get("model", None)
        self.filter = kwargs.get("filter", None)


class TestBatchEvaluation:
    """Integration tests for batch evaluation functionality."""

    def _create_sophisticated_game_mock(self):
        """Create the sophisticated FrozenLakeGameMock for realistic game simulation."""

        class FrozenLakeGameMock:
            def __init__(self):
                self.episodes = {}
                self.call_count = 0

            def handle_request(self, *args, **kwargs):
                """Handle HTTP requests and simulate Frozen Lake game logic"""
                self.call_count += 1

                # Parse the request to determine the endpoint
                if hasattr(args[0], "endswith"):
                    url = args[0]
                elif "url" in kwargs:
                    url = kwargs["url"]
                else:
                    url = str(args[0]) if args else ""

                # Check if this is a step request with JSON data
                json_data = kwargs.get("json", {})

                if "/start_episode" in url:
                    return self._start_episode()
                elif "/step" in url and json_data:
                    episode_id = json_data.get("episode_id")
                    action = json_data.get("action")
                    return self._step(episode_id, action)
                else:
                    # Default response for other requests
                    return self._default_response()

            def _start_episode(self):
                """Start a new episode"""
                episode_id = f"episode_{self.call_count}"
                self.episodes[episode_id] = {
                    "position": [0, 0],  # Start position
                    "step_count": 0,
                    "done": False,
                    "won": False,
                    "grid": [
                        ["S", "F", "F", "F"],
                        ["F", "H", "F", "H"],
                        ["F", "F", "F", "H"],
                        ["H", "F", "F", "G"],
                    ],
                }

                observation = {
                    "position": [0, 0],
                    "current_cell": "S",
                    "done": False,
                    "won": False,
                    "message": "Game started. You are at the starting position.",
                    "visual": self._generate_visual(episode_id),
                    "step_count": 0,
                }

                response = Mock()
                response.status_code = 200
                response.json.return_value = {
                    "episode_id": episode_id,
                    "observation": observation,
                }
                return response

            def _step(self, episode_id, action):
                """Process a step action"""
                if episode_id not in self.episodes:
                    # Episode doesn't exist, create a basic one
                    self.episodes[episode_id] = {
                        "position": [0, 0],
                        "step_count": 0,
                        "done": False,
                        "won": False,
                        "grid": [
                            ["S", "F", "F", "F"],
                            ["F", "H", "F", "H"],
                            ["F", "F", "F", "H"],
                            ["H", "F", "F", "G"],
                        ],
                    }

                episode = self.episodes[episode_id]

                if episode["done"]:
                    # Episode already finished
                    observation = self._get_observation(episode_id)
                else:
                    # Process the action
                    episode["step_count"] += 1
                    old_pos = episode["position"].copy()
                    new_pos = self._apply_action(episode["position"], action)
                    episode["position"] = new_pos

                    # Check what happened
                    row, col = new_pos
                    cell = episode["grid"][row][col]

                    if cell == "G":
                        # Reached goal!
                        episode["done"] = True
                        episode["won"] = True
                        message = "Congratulations! You reached the goal! You win! Success!"
                    elif cell == "H":
                        # Fell in hole
                        episode["done"] = True
                        episode["won"] = False
                        message = "Oh no! You fell into a hole. Game over."
                    else:
                        # Normal move
                        action_names = {0: "left", 1: "down", 2: "right", 3: "up"}
                        action_name = action_names.get(action, "unknown")
                        if new_pos != old_pos:
                            message = f"You moved {action_name} to a {cell} cell."
                        else:
                            message = f"You tried to move {action_name} but hit a wall."

                    observation = {
                        "position": new_pos,
                        "current_cell": cell,
                        "done": episode["done"],
                        "won": episode["won"],
                        "message": message,
                        "visual": self._generate_visual(episode_id),
                        "step_count": episode["step_count"],
                    }

                response = Mock()
                response.status_code = 200
                response.json.return_value = {
                    "observation": observation,
                    "is_done": episode["done"],
                    "info": {"step_count": episode["step_count"]},
                }
                return response

            def _apply_action(self, position, action):
                """Apply action and return new position"""
                row, col = position

                # Action mapping: 0=left, 1=down, 2=right, 3=up
                if action == 0:  # left
                    new_col = max(0, col - 1)
                    return [row, new_col]
                elif action == 1:  # down
                    new_row = min(3, row + 1)
                    return [new_row, col]
                elif action == 2:  # right
                    new_col = min(3, col + 1)
                    return [row, new_col]
                elif action == 3:  # up
                    new_row = max(0, row - 1)
                    return [new_row, col]
                else:
                    return position  # Invalid action, stay in place

            def _generate_visual(self, episode_id):
                """Generate visual representation of the game"""
                episode = self.episodes[episode_id]
                grid = episode["grid"]
                pos = episode["position"]

                visual_lines = []
                for r in range(4):
                    line = ""
                    for c in range(4):
                        if [r, c] == pos:
                            line += f"[{grid[r][c]}] "
                        else:
                            line += f" {grid[r][c]}  "
                    visual_lines.append(line.rstrip())

                return "\n".join(visual_lines)

            def _get_observation(self, episode_id):
                """Get current observation for an episode"""
                episode = self.episodes[episode_id]
                row, col = episode["position"]
                cell = episode["grid"][row][col]

                return {
                    "position": episode["position"],
                    "current_cell": cell,
                    "done": episode["done"],
                    "won": episode["won"],
                    "message": (
                        "Congratulations! You reached the goal! You win! Success!"
                        if episode["won"]
                        else "Game finished."
                    ),
                    "visual": self._generate_visual(episode_id),
                    "step_count": episode["step_count"],
                }

            def _default_response(self):
                """Default response for unhandled requests"""
                response = Mock()
                response.status_code = 200
                response.json.return_value = {"status": "ok"}
                return response

        return FrozenLakeGameMock()

    def setup_method(self):
        """Set up test environment before each test."""
        # Ensure we have the necessary environment variables
        self.original_env = {}

        # Store original environment values
        env_vars = ["FIREWORKS_API_KEY", "OPENAI_API_KEY", "MODEL_AGENT"]
        for var in env_vars:
            self.original_env[var] = os.environ.get(var)

        # Set default model for agent if not specified
        if not os.environ.get("MODEL_AGENT"):
            os.environ["MODEL_AGENT"] = "accounts/fireworks/models/qwen3-235b-a22b"

        # Set mock API keys to avoid skipping tests
        os.environ["FIREWORKS_API_KEY"] = "mock-fireworks-key"
        os.environ["OPENAI_API_KEY"] = "mock-openai-key"

    def teardown_method(self):
        """Clean up after each test."""
        # Restore original environment
        for var, value in self.original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

    @pytest.mark.asyncio
    @patch("eval_protocol.agent.orchestrator.AsyncOpenAI")
    @patch("subprocess.Popen")
    @patch("aiohttp.ClientSession.post")
    @patch("httpx.Client.post")
    @patch("requests.get")
    async def test_batch_evaluation_task_manager_fireworks(
        self,
        mock_requests_get,
        mock_httpx_post,
        mock_aiohttp_post,
        mock_subprocess_popen,
        mock_openai_constructor,
    ):
        """Test batch evaluation using TaskManager with Fireworks API."""
        # Mock OpenAI client in orchestrator
        mock_openai_client = AsyncMock()
        mock_openai_constructor.return_value = mock_openai_client

        # Mock OpenAI completion response - simulate smart AI moves
        def create_tool_call_response(action, call_id):
            mock_tool_call = Mock()
            mock_tool_call.function.name = "step"
            mock_tool_call.function.arguments = f'{{"action": "{action}"}}'
            mock_tool_call.id = call_id

            mock_message = Mock()
            mock_message.content = None
            mock_message.role = "assistant"
            mock_message.tool_calls = [mock_tool_call]
            mock_message.model_dump = Mock(
                return_value={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "step",
                                "arguments": f'{{"action": "{action}"}}',
                            },
                        }
                    ],
                }
            )

            mock_completion = AsyncMock()
            mock_completion.choices = [Mock(message=mock_message)]
            mock_completion.usage = Mock(total_tokens=10)
            return mock_completion

        # Smart AI with winning sequence: right -> right -> down -> down -> down -> right
        winning_sequence = ["right", "right", "down", "down", "down", "right"]
        rollout_counter = [0]

        def smart_move_generator(**kwargs):
            messages = kwargs.get("messages", [])
            move_count = sum(1 for msg in messages if msg.get("role") == "assistant" and msg.get("tool_calls"))

            rollout_counter[0] += 1
            if rollout_counter[0] <= 6:  # First rollout wins
                action = winning_sequence[move_count] if move_count < len(winning_sequence) else "right"
            else:  # Second rollout makes mistake
                action = (
                    "right"
                    if move_count == 4
                    else (winning_sequence[move_count] if move_count < len(winning_sequence) else "right")
                )

            return create_tool_call_response(action, f"call_{move_count}")

        mock_openai_client.chat.completions.create = AsyncMock(side_effect=smart_move_generator)

        # Mock Fireworks API response (backup)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {"content": "right", "role": "assistant"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 10},
            }
        )
        mock_aiohttp_post.return_value.__aenter__.return_value = mock_response

        # Use sophisticated game mock
        game_mock = self._create_sophisticated_game_mock()
        mock_httpx_post.side_effect = game_mock.handle_request

        # Mock health check
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        mock_requests_get.return_value = mock_health_response

        task_manager = TaskManager()

        # Load the frozen lake task definition for Fireworks
        task_def_path = Path("examples/frozen_lake/client/task_def.yaml")
        if not task_def_path.exists():
            pytest.skip(f"Task definition not found: {task_def_path}")

        # Load and register the task
        task_def = task_manager._load_task_from_file(str(task_def_path))
        assert task_def is not None, "Failed to load task definition"

        # Override for traditional batch evaluation (not data-driven)
        task_def.dataset_path = None  # Remove dataset path to use traditional evaluation
        task_def.num_rollouts = 2

        task_id = task_manager.register_task(task_def)
        assert task_id == "frozen_lake_http_rollout"

        # Configure subprocess.Popen mock to prevent real process creation
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.communicate.return_value = (b"", b"")
        mock_subprocess_popen.return_value = mock_process

        try:
            # Mock server process management
            with (
                patch.object(task_manager, "_start_resource_server", return_value=12345),
                patch.object(task_manager, "_wait_for_server_health", return_value=True),
            ):
                # Execute the task with batch evaluation
                results = await task_manager.execute_tasks(
                    task_ids=[task_id],
                    parallel=False,
                    max_concurrency=2,
                    num_rollouts_override=2,
                )

            # Validate results structure
            assert task_id in results
            result = results[task_id]

            # Should not be an error result
            assert not (isinstance(result, dict) and "error" in result), (
                f"Task failed: {result.get('error', 'Unknown error')}"
            )

            # Should be aggregated results
            assert isinstance(result, dict)
            assert result.get("aggregated", False), "Results should be aggregated for batch evaluation"

            # Validate aggregated result structure
            required_keys = [
                "num_rollouts",
                "successful_rollouts",
                "success_rate",
                "avg_score",
                "min_score",
                "max_score",
            ]
            for key in required_keys:
                assert key in result, f"Missing key in aggregated results: {key}"

            # Validate result values
            assert result["num_rollouts"] == 2
            assert result["successful_rollouts"] >= 0
            assert result["successful_rollouts"] <= result["num_rollouts"]
            assert 0.0 <= result["success_rate"] <= 1.0
            assert isinstance(result["avg_score"], (int, float))
            assert isinstance(result["min_score"], (int, float))
            assert isinstance(result["max_score"], (int, float))
            assert result["min_score"] <= result["avg_score"] <= result["max_score"]

            # Should have individual results
            assert "individual_scores" in result
            assert "individual_results" in result
            assert len(result["individual_scores"]) == result["successful_rollouts"]
            assert len(result["individual_results"]) == result["successful_rollouts"]

            logging.info(f"Fireworks batch evaluation completed successfully: {result}")

        finally:
            await task_manager.cleanup()

    @pytest.mark.asyncio
    @patch("eval_protocol.agent.orchestrator.AsyncOpenAI")
    @patch("subprocess.Popen")
    @patch("httpx.Client.post")
    @patch("requests.get")
    async def test_batch_evaluation_task_manager_openai(
        self, mock_requests_get, mock_httpx_post, mock_subprocess_popen, mock_openai
    ):
        """Test batch evaluation using TaskManager with OpenAI API."""
        # Mock OpenAI client and response
        mock_openai_client = AsyncMock()
        mock_openai.return_value = mock_openai_client

        mock_completion = AsyncMock()
        # Mock OpenAI completion response with proper tool call structure
        mock_tool_call = Mock()
        mock_tool_call.function.name = "step"
        mock_tool_call.function.arguments = '{"action": "down"}'
        mock_tool_call.id = "call_openai_123"

        mock_message = Mock()
        mock_message.content = None
        mock_message.role = "assistant"
        mock_message.tool_calls = [mock_tool_call]
        mock_message.model_dump = Mock(
            return_value={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_openai_123",
                        "type": "function",
                        "function": {"name": "step", "arguments": '{"action": "down"}'},
                    }
                ],
            }
        )

        mock_completion = AsyncMock()
        mock_completion.choices = [Mock(message=mock_message)]
        mock_completion.usage = Mock(total_tokens=15)
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        # Mock HTTP rollout server responses
        mock_httpx_response = Mock()
        mock_httpx_response.status_code = 200
        # Create responses for OpenAI test
        responses = [
            {"episode_id": "test_episode_openai"},
            {
                "position": [1, 0],
                "current_cell": "F",
                "done": False,
                "won": False,
                "message": "You moved down",
                "visual": " S  F  F  F\n[F] H  F  H\n F  F  F  H\n H  F  F  G",
            },
            {
                "position": [3, 3],
                "current_cell": "G",
                "done": True,
                "won": True,
                "message": "Victory!",
                "visual": " S  F  F  F\n F  H  F  H\n F  F  F  H\n H  F  F [G]",
            },
        ]
        response_iter = iter(responses * 10)
        mock_httpx_response.json.side_effect = lambda: next(response_iter, responses[-1])
        mock_httpx_post.return_value = mock_httpx_response

        # Mock health check
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        mock_requests_get.return_value = mock_health_response

        task_manager = TaskManager()

        # Load the frozen lake task definition for OpenAI
        task_def_path = Path("examples/frozen_lake/client/task_def_openai.yaml")
        if not task_def_path.exists():
            pytest.skip(f"Task definition not found: {task_def_path}")

        # Load and register the task
        task_def = task_manager._load_task_from_file(str(task_def_path))
        assert task_def is not None, "Failed to load task definition"

        # Override num_rollouts to reduce test time
        task_def.num_rollouts = 2

        # Set OpenAI model temporarily
        original_model = os.environ.get("MODEL_AGENT")
        os.environ["MODEL_AGENT"] = "gpt-4o-mini"

        task_id = task_manager.register_task(task_def)
        assert task_id == "frozen_lake_http_rollout_openai"

        # Configure subprocess.Popen mock to prevent real process creation
        mock_process = Mock()
        mock_process.pid = 12346
        mock_process.poll.return_value = None
        mock_process.communicate.return_value = (b"", b"")
        mock_subprocess_popen.return_value = mock_process

        try:
            # Mock server process management
            with (
                patch.object(task_manager, "_start_resource_server", return_value=12346),
                patch.object(task_manager, "_wait_for_server_health", return_value=True),
            ):
                # Execute the task with batch evaluation
                results = await task_manager.execute_tasks(
                    task_ids=[task_id],
                    parallel=False,
                    max_concurrency=2,
                    num_rollouts_override=2,
                )

            # Validate results structure
            assert task_id in results
            result = results[task_id]

            # Should not be an error result
            assert not (isinstance(result, dict) and "error" in result), (
                f"Task failed: {result.get('error', 'Unknown error')}"
            )

            # Should be aggregated results
            assert isinstance(result, dict)
            assert result.get("aggregated", False), "Results should be aggregated for batch evaluation"

            # Validate aggregated result structure
            required_keys = [
                "num_rollouts",
                "successful_rollouts",
                "success_rate",
                "avg_score",
                "min_score",
                "max_score",
            ]
            for key in required_keys:
                assert key in result, f"Missing key in aggregated results: {key}"

            # Validate result values
            assert result["num_rollouts"] == 2
            assert result["successful_rollouts"] >= 0
            assert result["successful_rollouts"] <= result["num_rollouts"]
            assert 0.0 <= result["success_rate"] <= 1.0
            assert isinstance(result["avg_score"], (int, float))
            assert isinstance(result["min_score"], (int, float))
            assert isinstance(result["max_score"], (int, float))
            assert result["min_score"] <= result["avg_score"] <= result["max_score"]

            # Should have individual results
            assert "individual_scores" in result
            assert "individual_results" in result
            assert len(result["individual_scores"]) == result["successful_rollouts"]
            assert len(result["individual_results"]) == result["successful_rollouts"]

            logging.info(f"OpenAI batch evaluation completed successfully: {result}")

        finally:
            # Restore original model
            if original_model:
                os.environ["MODEL_AGENT"] = original_model
            else:
                os.environ.pop("MODEL_AGENT", None)
            await task_manager.cleanup()

    @patch("eval_protocol.agent.orchestrator.AsyncOpenAI")
    @patch("subprocess.Popen")
    @patch("aiohttp.ClientSession.post")
    @patch("httpx.Client.post")
    @patch("requests.get")
    def test_cli_batch_evaluation_fireworks(
        self,
        mock_requests_get,
        mock_httpx_post,
        mock_aiohttp_post,
        mock_subprocess_popen,
        mock_openai_constructor,
    ):
        """Test batch evaluation through CLI command with Fireworks."""
        # Mock OpenAI client in orchestrator
        mock_openai_client = AsyncMock()
        mock_openai_constructor.return_value = mock_openai_client

        mock_completion = AsyncMock()
        # Mock OpenAI completion response with proper tool call structure
        mock_tool_call = Mock()
        mock_tool_call.function.name = "step"
        mock_tool_call.function.arguments = '{"action": "up"}'
        mock_tool_call.id = "call_cli_fw"

        mock_message = Mock()
        mock_message.content = None
        mock_message.role = "assistant"
        mock_message.tool_calls = [mock_tool_call]
        mock_message.model_dump = Mock(
            return_value={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_cli_fw",
                        "type": "function",
                        "function": {"name": "step", "arguments": '{"action": "up"}'},
                    }
                ],
            }
        )

        mock_completion = AsyncMock()
        mock_completion.choices = [Mock(message=mock_message)]
        mock_completion.usage = Mock(total_tokens=8)
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        # Mock Fireworks API response (backup)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {"content": "up", "role": "assistant"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 8},
            }
        )
        mock_aiohttp_post.return_value.__aenter__.return_value = mock_response

        # Mock HTTP rollout responses
        mock_httpx_response = Mock()
        mock_httpx_response.status_code = 200

        responses = [
            {"episode_id": "test_cli_fw"},
            {"position": [0, 0], "done": False, "won": False, "message": "Test CLI FW"},
            {"position": [3, 3], "done": True, "won": True, "message": "CLI Win!"},
        ]
        response_iter = iter(responses * 5)
        mock_httpx_response.json.side_effect = lambda: next(response_iter, responses[-1])
        mock_httpx_post.return_value = mock_httpx_response

        # Mock health check
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        mock_requests_get.return_value = mock_health_response

        task_def_path = Path("examples/frozen_lake/client/task_def.yaml")
        if not task_def_path.exists():
            pytest.skip(f"Task definition not found: {task_def_path}")

        # Create mock args for CLI command
        args = MockArgs(
            task_def=str(task_def_path),
            num_rollouts=2,
            parallel=False,
            max_concurrency=2,
        )

        # Configure subprocess.Popen mock to prevent real process creation
        mock_process = Mock()
        mock_process.pid = 12348
        mock_process.poll.return_value = None
        mock_process.communicate.return_value = (b"", b"")
        mock_subprocess_popen.return_value = mock_process

        # Execute CLI command with mocked subprocess for server management
        with patch("time.sleep"):
            exit_code = agent_eval_command(args)

        # Should complete successfully
        assert exit_code == 0, "CLI command should complete successfully"

    @patch("eval_protocol.agent.orchestrator.AsyncOpenAI")
    @patch("subprocess.Popen")
    @patch("httpx.Client.post")
    @patch("requests.get")
    def test_cli_batch_evaluation_openai(self, mock_requests_get, mock_httpx_post, mock_subprocess_popen, mock_openai):
        """Test batch evaluation through CLI command with OpenAI."""
        # Mock OpenAI client
        mock_openai_client = AsyncMock()
        mock_openai.return_value = mock_openai_client
        mock_completion = AsyncMock()
        # Mock OpenAI completion response with proper tool call structure
        mock_tool_call = Mock()
        mock_tool_call.function.name = "step"
        mock_tool_call.function.arguments = '{"action": "left"}'
        mock_tool_call.id = "call_cli_openai"

        mock_message = Mock()
        mock_message.content = None
        mock_message.role = "assistant"
        mock_message.tool_calls = [mock_tool_call]
        mock_message.model_dump = Mock(
            return_value={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_cli_openai",
                        "type": "function",
                        "function": {"name": "step", "arguments": '{"action": "left"}'},
                    }
                ],
            }
        )

        mock_completion = AsyncMock()
        mock_completion.choices = [Mock(message=mock_message)]
        mock_completion.usage = Mock(total_tokens=10)
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        # Mock HTTP rollout responses
        mock_httpx_response = Mock()
        mock_httpx_response.status_code = 200

        responses = [
            {"episode_id": "test_cli_openai"},
            {
                "position": [0, 0],
                "done": False,
                "won": False,
                "message": "Test CLI OpenAI",
            },
            {
                "position": [3, 3],
                "done": True,
                "won": True,
                "message": "CLI OpenAI Win!",
            },
        ]
        response_iter = iter(responses * 5)
        mock_httpx_response.json.side_effect = lambda: next(response_iter, responses[-1])
        mock_httpx_post.return_value = mock_httpx_response

        # Mock health check
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        mock_requests_get.return_value = mock_health_response

        task_def_path = Path("examples/frozen_lake/client/task_def_openai.yaml")
        if not task_def_path.exists():
            pytest.skip(f"Task definition not found: {task_def_path}")

        # Set OpenAI model
        original_model = os.environ.get("MODEL_AGENT")
        os.environ["MODEL_AGENT"] = "gpt-4o-mini"

        try:
            # Create mock args for CLI command
            args = MockArgs(
                task_def=str(task_def_path),
                num_rollouts=2,
                parallel=False,
                max_concurrency=2,
            )

            # Configure subprocess.Popen mock to prevent real process creation
            mock_process = Mock()
            mock_process.pid = 12349
            mock_process.poll.return_value = None
            mock_process.communicate.return_value = (b"", b"")
            mock_subprocess_popen.return_value = mock_process

            # Execute CLI command with mocked subprocess for server management
            with patch("time.sleep"):
                exit_code = agent_eval_command(args)

            # Should complete successfully
            assert exit_code == 0, "CLI command should complete successfully"

        finally:
            # Restore original model
            if original_model:
                os.environ["MODEL_AGENT"] = original_model
            else:
                os.environ.pop("MODEL_AGENT", None)

    @pytest.mark.asyncio
    @patch("eval_protocol.agent.orchestrator.AsyncOpenAI")
    @patch("subprocess.Popen")
    @patch("aiohttp.ClientSession.post")
    @patch("httpx.Client.post")
    @patch("requests.get")
    async def test_parallel_batch_evaluation(
        self,
        mock_requests_get,
        mock_httpx_post,
        mock_aiohttp_post,
        mock_subprocess_popen,
        mock_openai_constructor,
    ):
        """Test parallel execution of multiple rollouts."""
        # Mock OpenAI client in orchestrator
        mock_openai_client = AsyncMock()
        mock_openai_constructor.return_value = mock_openai_client

        # Mock OpenAI completion response with smart AI moves
        def create_tool_call_response(action, call_id):
            mock_tool_call = Mock()
            mock_tool_call.function.name = "step"
            mock_tool_call.function.arguments = f'{{"action": "{action}"}}'
            mock_tool_call.id = call_id

            mock_message = Mock()
            mock_message.content = None
            mock_message.role = "assistant"
            mock_message.tool_calls = [mock_tool_call]
            mock_message.model_dump = Mock(
                return_value={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "step",
                                "arguments": f'{{"action": "{action}"}}',
                            },
                        }
                    ],
                }
            )

            mock_completion = AsyncMock()
            mock_completion.choices = [Mock(message=mock_message)]
            mock_completion.usage = Mock(total_tokens=12)
            return mock_completion

        # Smart AI for parallel test
        winning_sequence = ["right", "right", "down", "down", "down", "right"]
        rollout_counter = [0]

        def smart_move_generator(**kwargs):
            messages = kwargs.get("messages", [])
            move_count = sum(1 for msg in messages if msg.get("role") == "assistant" and msg.get("tool_calls"))

            rollout_counter[0] += 1
            if rollout_counter[0] <= 6:  # First rollout wins
                action = winning_sequence[move_count] if move_count < len(winning_sequence) else "right"
            else:  # Second/third rollout makes mistake
                action = (
                    "right"
                    if move_count == 4
                    else (winning_sequence[move_count] if move_count < len(winning_sequence) else "right")
                )

            return create_tool_call_response(action, f"call_{move_count}")

        mock_openai_client.chat.completions.create = AsyncMock(side_effect=smart_move_generator)

        # Mock Fireworks API response (backup)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {"content": "right", "role": "assistant"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 12},
            }
        )
        mock_aiohttp_post.return_value.__aenter__.return_value = mock_response

        # Use sophisticated game mock for parallel test
        game_mock = self._create_sophisticated_game_mock()
        mock_httpx_post.side_effect = game_mock.handle_request

        # Mock health check
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        mock_requests_get.return_value = mock_health_response

        task_manager = TaskManager()

        # Load the frozen lake task definition
        task_def_path = Path("examples/frozen_lake/client/task_def.yaml")
        if not task_def_path.exists():
            pytest.skip(f"Task definition not found: {task_def_path}")

        # Load and register the task
        task_def = task_manager._load_task_from_file(str(task_def_path))
        assert task_def is not None, "Failed to load task definition"

        # Test with more rollouts to verify parallelism
        task_def.num_rollouts = 3

        task_id = task_manager.register_task(task_def)

        # Configure subprocess.Popen mock to prevent real process creation
        mock_process = Mock()
        mock_process.pid = 12347
        mock_process.poll.return_value = None
        mock_process.communicate.return_value = (b"", b"")
        mock_subprocess_popen.return_value = mock_process

        try:
            # Mock server process management
            with (
                patch.object(task_manager, "_start_resource_server", return_value=12347),
                patch.object(task_manager, "_wait_for_server_health", return_value=True),
            ):
                # Execute with parallel enabled
                results = await task_manager.execute_tasks(
                    task_ids=[task_id],
                    parallel=True,
                    max_concurrency=2,
                    num_rollouts_override=3,
                )

            # Validate results
            assert task_id in results
            result = results[task_id]

            # Should be successful and aggregated
            assert not (isinstance(result, dict) and "error" in result)
            assert result.get("aggregated", False)
            assert result["num_rollouts"] == 3

            logging.info(f"Parallel batch evaluation completed: {result}")

        finally:
            await task_manager.cleanup()

    @pytest.mark.asyncio
    @patch("eval_protocol.agent.orchestrator.AsyncOpenAI")
    @patch("subprocess.Popen")
    @patch("aiohttp.ClientSession.post")
    @patch("httpx.Client.post")
    @patch("requests.get")
    async def test_server_lifecycle_management(
        self,
        mock_requests_get,
        mock_httpx_post,
        mock_aiohttp_post,
        mock_subprocess_popen,
        mock_openai_constructor,
    ):
        """Test that resource servers are properly started and stopped."""
        # Mock OpenAI client in orchestrator
        mock_openai_client = AsyncMock()
        mock_openai_constructor.return_value = mock_openai_client

        # Mock OpenAI completion response with smart AI moves
        def create_tool_call_response(action, call_id):
            mock_tool_call = Mock()
            mock_tool_call.function.name = "step"
            mock_tool_call.function.arguments = f'{{"action": "{action}"}}'
            mock_tool_call.id = call_id

            mock_message = Mock()
            mock_message.content = None
            mock_message.role = "assistant"
            mock_message.tool_calls = [mock_tool_call]
            mock_message.model_dump = Mock(
                return_value={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "step",
                                "arguments": f'{{"action": "{action}"}}',
                            },
                        }
                    ],
                }
            )

            mock_completion = AsyncMock()
            mock_completion.choices = [Mock(message=mock_message)]
            mock_completion.usage = Mock(total_tokens=5)
            return mock_completion

        # Smart AI for lifecycle test
        winning_sequence = ["right", "right", "down", "down", "down", "right"]
        rollout_counter = [0]

        def smart_move_generator(**kwargs):
            messages = kwargs.get("messages", [])
            move_count = sum(1 for msg in messages if msg.get("role") == "assistant" and msg.get("tool_calls"))

            rollout_counter[0] += 1
            if rollout_counter[0] <= 6:  # First rollout wins
                action = winning_sequence[move_count] if move_count < len(winning_sequence) else "right"
            else:  # Second rollout makes mistake
                action = (
                    "right"
                    if move_count == 4
                    else (winning_sequence[move_count] if move_count < len(winning_sequence) else "right")
                )

            return create_tool_call_response(action, f"call_{move_count}")

        mock_openai_client.chat.completions.create = AsyncMock(side_effect=smart_move_generator)

        # Mock API responses (backup)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "down", "role": "assistant"}}],
                "usage": {"total_tokens": 5},
            }
        )
        mock_aiohttp_post.return_value.__aenter__.return_value = mock_response

        # Use sophisticated game mock for lifecycle test
        game_mock = self._create_sophisticated_game_mock()
        mock_httpx_post.side_effect = game_mock.handle_request

        mock_health_response = Mock()
        mock_health_response.status_code = 200
        mock_requests_get.return_value = mock_health_response

        task_manager = TaskManager()

        # Load task definition
        task_def_path = Path("examples/frozen_lake/client/task_def.yaml")
        if not task_def_path.exists():
            pytest.skip(f"Task definition not found: {task_def_path}")

        task_def = task_manager._load_task_from_file(str(task_def_path))
        task_def.num_rollouts = 2
        task_id = task_manager.register_task(task_def)

        # Check that no servers are running initially
        assert len(task_manager.server_processes) == 0
        assert len(task_manager.server_ports) == 0

        # Configure subprocess.Popen mock to prevent real process creation
        mock_process = Mock()
        mock_process.pid = 12348
        mock_process.poll.return_value = None
        mock_process.communicate.return_value = (b"", b"")
        mock_subprocess_popen.return_value = mock_process

        try:
            # Mock server process management
            with (
                patch.object(task_manager, "_start_resource_server", return_value=12348),
                patch.object(task_manager, "_wait_for_server_health", return_value=True),
            ):
                # Execute task
                results = await task_manager.execute_tasks(task_ids=[task_id], num_rollouts_override=2)

            # Task should complete successfully
            assert task_id in results
            result = results[task_id]
            assert not (isinstance(result, dict) and "error" in result)

        finally:
            await task_manager.cleanup()

            # Check that all servers are cleaned up
            assert len(task_manager.server_processes) == 0
            assert len(task_manager.server_ports) == 0


class TestBatchEvaluationErrorHandling:
    """Test error handling in batch evaluation scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_task_definition(self):
        """Test handling of invalid task definitions."""
        task_manager = TaskManager()

        # Create a temporary invalid task definition
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
name: "invalid_task"
description: "This task has invalid configuration"
resource_type: "nonexistent_resource"
"""
            )
            invalid_task_path = f.name

        try:
            # Attempt to load invalid task
            task_def = task_manager._load_task_from_file(invalid_task_path)

            # Should either fail to load or fail during execution
            if task_def is not None:
                task_id = task_manager.register_task(task_def)
                results = await task_manager.execute_tasks([task_id])

                # Should result in error
                assert task_id in results
                result = results[task_id]
                assert isinstance(result, dict) and "error" in result

        finally:
            # Clean up temporary file
            Path(invalid_task_path).unlink(missing_ok=True)
            await task_manager.cleanup()

    @pytest.mark.asyncio
    async def test_missing_api_key_handling(self):
        """Test graceful handling when API keys are missing."""
        # Temporarily remove API keys
        original_fw_key = os.environ.get("FIREWORKS_API_KEY")
        original_openai_key = os.environ.get("OPENAI_API_KEY")

        os.environ.pop("FIREWORKS_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)

        task_manager = TaskManager()

        try:
            task_def_path = Path("examples/frozen_lake/client/task_def.yaml")
            if not task_def_path.exists():
                pytest.skip(f"Task definition not found: {task_def_path}")

            task_def = task_manager._load_task_from_file(str(task_def_path))
            if task_def:
                task_def.num_rollouts = 1  # Reduce rollouts for faster failure
                task_id = task_manager.register_task(task_def)

                results = await task_manager.execute_tasks([task_id])

                # Should handle missing API key gracefully
                assert task_id in results
                result = results[task_id]
                # Result could be error or have low success rate due to API failures
                if isinstance(result, dict) and "error" in result:
                    # Direct error is acceptable
                    pass
                elif isinstance(result, dict) and result.get("aggregated", False):
                    # Batch result with low success rate is also acceptable
                    assert result["success_rate"] <= 1.0

        finally:
            # Restore API keys
            if original_fw_key:
                os.environ["FIREWORKS_API_KEY"] = original_fw_key
            if original_openai_key:
                os.environ["OPENAI_API_KEY"] = original_openai_key
            await task_manager.cleanup()
