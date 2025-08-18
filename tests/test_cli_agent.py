"""
Tests for the agent-eval CLI command.
"""

import argparse
import asyncio  # Added import
import json
import logging
from unittest.mock import (  # Added AsyncMock and Mock
    AsyncMock,
    MagicMock,
    Mock,
    mock_open,
    patch,
)

import pytest
import yaml

from eval_protocol.cli_commands.agent_eval_cmd import agent_eval_command
from eval_protocol.models import TaskDefinitionModel

MINIMAL_TASK_DEF_CONTENT_DICT = {
    "name": "CLI Test Task",
    "description": "Task for CLI test.",
    "resource_type": "PythonStateResource",
    "base_resource_config": {},
    "reward_function_path": "test_module.test_reward",
    "goal_description": "Test goal.",
    "poc_max_turns": 1,
}
MINIMAL_TASK_DEF_YAML_CONTENT = yaml.dump(MINIMAL_TASK_DEF_CONTENT_DICT)
MINIMAL_TASK_DEF_JSON_CONTENT = json.dumps(MINIMAL_TASK_DEF_CONTENT_DICT)


class TestAgentEvalCommand:
    """Tests for the agent_eval_command function."""

    @patch("eval_protocol.cli_commands.agent_eval_cmd.TaskManager")
    @patch("eval_protocol.cli_commands.agent_eval_cmd.Path")
    def test_agent_eval_success_yaml(self, MockPath, MockTaskManager, caplog):
        # Configure caplog to capture logs from the agent_eval logger
        caplog.set_level(logging.INFO, logger="agent_eval")

        # Setup Path mock
        mock_path_instance = Mock()
        MockPath.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        mock_path_instance.__str__ = MagicMock(return_value="dummy_task.yaml")

        # Setup TaskManager mock
        mock_task_manager = MagicMock()
        MockTaskManager.return_value = mock_task_manager
        mock_task_manager._load_task_from_file = MagicMock(
            return_value=TaskDefinitionModel(**MINIMAL_TASK_DEF_CONTENT_DICT)
        )
        mock_task_manager.register_task.return_value = "task1"
        mock_task_manager.execute_tasks = AsyncMock(return_value={"task1": {"score": 1.0}})
        mock_task_manager.cleanup = AsyncMock()

        args = argparse.Namespace(task_def="dummy_task.yaml", verbose=False, debug=False)

        # Run the command
        result = agent_eval_command(args)

        # Verify the result
        assert result == 0
        MockPath.assert_called_once_with("dummy_task.yaml")
        mock_task_manager._load_task_from_file.assert_called_once()
        mock_task_manager.register_task.assert_called_once()
        mock_task_manager.execute_tasks.assert_awaited_once()
        mock_task_manager.cleanup.assert_awaited_once()
        assert "agent-eval command finished successfully" in caplog.text

    @patch("eval_protocol.cli_commands.agent_eval_cmd.TaskManager")
    @patch("eval_protocol.cli_commands.agent_eval_cmd.Path")
    @patch("eval_protocol.cli_commands.agent_eval_cmd.yaml", None)
    def test_agent_eval_success_json_no_yaml_lib(self, MockPath, MockTaskManager, caplog):
        # Setup Path mock
        mock_path_instance = Mock()
        MockPath.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        mock_path_instance.__str__ = MagicMock(return_value="dummy_task.json")

        # Setup TaskManager mock
        mock_task_manager = MagicMock()
        MockTaskManager.return_value = mock_task_manager
        mock_task_manager._load_task_from_file = MagicMock(
            return_value=TaskDefinitionModel(**MINIMAL_TASK_DEF_CONTENT_DICT)
        )
        mock_task_manager.register_task.return_value = "task1"
        mock_task_manager.execute_tasks = AsyncMock(return_value={"task1": {"score": 1.0}})
        mock_task_manager.cleanup = AsyncMock()

        args = argparse.Namespace(task_def="dummy_task.json")
        result = agent_eval_command(args)

        assert result == 0
        # The yaml import message is in TaskManager._load_task_from_file now
        mock_task_manager._load_task_from_file.assert_called_once()
        mock_task_manager.execute_tasks.assert_awaited_once()

    def test_agent_eval_no_task_def_arg(self, caplog):
        args = argparse.Namespace(task_def=None)
        result = agent_eval_command(args)
        assert result == 1
        assert "Error: --task-def (path to task definition YAML file or directory) is required." in caplog.text

    @patch("eval_protocol.cli_commands.agent_eval_cmd.TaskManager")
    @patch("eval_protocol.cli_commands.agent_eval_cmd.Path")
    def test_agent_eval_task_def_file_not_found(self, MockPath, MockTaskManager, caplog):
        # Setup path mock
        mock_path_instance = Mock()
        MockPath.return_value = mock_path_instance
        mock_path_instance.exists.return_value = False
        mock_path_instance.is_file.return_value = False
        mock_path_instance.is_dir.return_value = False
        mock_path_instance.__str__ = MagicMock(return_value="non_existent_task.yaml")

        args = argparse.Namespace(task_def="non_existent_task.yaml")
        result = agent_eval_command(args)

        assert result == 1
        assert "Task definition path not found or invalid" in caplog.text

    @patch("eval_protocol.cli_commands.agent_eval_cmd.TaskManager")
    @patch("eval_protocol.cli_commands.agent_eval_cmd.Path")
    def test_agent_eval_invalid_yaml_content(self, MockPath, MockTaskManager, caplog):
        # Setup path mock
        mock_path_instance = Mock()
        MockPath.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        mock_path_instance.__str__ = MagicMock(return_value="invalid_task.yaml")

        # Setup TaskManager mock to simulate a file load failure
        mock_task_manager = MagicMock()
        MockTaskManager.return_value = mock_task_manager
        mock_task_manager._load_task_from_file = MagicMock(return_value=None)

        args = argparse.Namespace(task_def="invalid_task.yaml")
        result = agent_eval_command(args)

        assert result == 1
        assert "Failed to load task definition" in caplog.text

    @patch("eval_protocol.cli_commands.agent_eval_cmd.TaskManager")
    @patch("eval_protocol.cli_commands.agent_eval_cmd.Path")
    def test_agent_eval_pydantic_validation_error(self, MockPath, MockTaskManager, caplog):
        # Setup path mock
        mock_path_instance = Mock()
        MockPath.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        mock_path_instance.__str__ = MagicMock(return_value="incomplete_task.yaml")

        # Setup TaskManager to have the validation error happen in _load_task_from_file
        mock_task_manager = MagicMock()
        MockTaskManager.return_value = mock_task_manager
        mock_task_manager._load_task_from_file = MagicMock(return_value=None)

        args = argparse.Namespace(task_def="incomplete_task.yaml")
        result = agent_eval_command(args)

        assert result == 1
        assert "Failed to load task definition" in caplog.text

    @patch("eval_protocol.cli_commands.agent_eval_cmd.TaskManager")
    @patch("eval_protocol.cli_commands.agent_eval_cmd.Path")
    def test_agent_eval_orchestrator_instantiation_fails(self, MockPath, MockTaskManager, caplog):
        # Setup path mock to fail in a way that matches error message
        mock_path_instance = Mock()
        MockPath.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        mock_path_instance.__str__ = MagicMock(return_value="dummy_task.yaml")

        # Setup TaskManager mock to make loading the file fail
        mock_task_manager = MagicMock()
        MockTaskManager.return_value = mock_task_manager
        mock_task_manager._load_task_from_file = MagicMock(return_value=None)

        args = argparse.Namespace(task_def="dummy_task.yaml")
        result = agent_eval_command(args)

        assert result == 1
        assert "Failed to load task definition" in caplog.text

    @patch("eval_protocol.cli_commands.agent_eval_cmd.TaskManager")
    @patch("eval_protocol.cli_commands.agent_eval_cmd.Path")
    def test_agent_eval_orchestrator_execution_fails(self, MockPath, MockTaskManager, caplog):
        # Setup path mock
        mock_path_instance = Mock()
        MockPath.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        mock_path_instance.__str__ = MagicMock(return_value="dummy_task.yaml")

        # Setup TaskManager mock
        mock_task_manager = MagicMock()
        MockTaskManager.return_value = mock_task_manager
        mock_task_manager._load_task_from_file = MagicMock(
            return_value=TaskDefinitionModel(**MINIMAL_TASK_DEF_CONTENT_DICT)
        )
        mock_task_manager.register_task.return_value = "task1"

        # Make execute_tasks raise an exception
        mock_task_manager.execute_tasks = AsyncMock(side_effect=RuntimeError("Execution failed"))  # type: ignore[attr-defined]
        mock_task_manager.cleanup = AsyncMock()

        args = argparse.Namespace(task_def="dummy_task.yaml")
        result = agent_eval_command(args)

        assert result == 1
        assert "Error during agent-eval execution: Execution failed" in caplog.text
