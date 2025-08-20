#!/usr/bin/env python3
"""
Mock Environment for τ²-Bench Integration

This module implements a MockEnvironment that integrates the τ²-Bench simulation
pattern (Agent/User/Environment communication) with the MCP-Gym framework.
"""

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from vendor.tau2.domains.mock.data_model import MockDB
from vendor.tau2.domains.mock.tools import MockTools

logger = logging.getLogger(__name__)


def _get_mock_db_path():
    """Get mock database path, downloading if necessary."""
    import os
    import tempfile
    import urllib.request
    from pathlib import Path

    # Try local development path first
    try:
        from vendor.tau2.domains.mock.utils import MOCK_DB_PATH

        if Path(MOCK_DB_PATH).exists():
            return MOCK_DB_PATH
    except (ImportError, FileNotFoundError):
        pass

    # Use a cache directory in user's temp/cache area
    cache_dir = Path(tempfile.gettempdir()) / "tau2_bench_cache"
    cache_dir.mkdir(exist_ok=True)
    mock_db_path = cache_dir / "mock_db.json"

    if not mock_db_path.exists():
        print(f"📥 Downloading mock database to {mock_db_path}...")
        url = "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/mock/db.json"
        try:
            urllib.request.urlretrieve(url, mock_db_path)
            print("✅ Download complete!")
        except Exception as e:
            raise RuntimeError(f"Failed to download mock database: {e}")

    return mock_db_path


MOCK_DB_PATH = _get_mock_db_path()


class MockEnvironment:
    """
    Mock environment that integrates τ²-Bench simulation pattern
    with MCP-Gym framework.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.db = MockDB.load(MOCK_DB_PATH)
        self.mock_tools = MockTools(self.db)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state"""
        return {}, {}

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Perform one step of the τ²-Bench simulation.
        """

        action_name = action.get("action", "")
        parameters = action.get("parameters", {})

        result = self._execute_mock_action(action_name, parameters)

        # In tau2-bench, if there's a simulated user, the agent cannot terminate the rollout, and there are no per step rewards.
        observation = result
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _execute_mock_action(self, action_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action using mock tools."""
        action_map = {
            "create_task": self.mock_tools.create_task,
            "get_users": self.mock_tools.get_users,
            "update_task_status": self.mock_tools.update_task_status,
            "assert_number_of_tasks": self.mock_tools.assert_number_of_tasks,
            "assert_task_status": self.mock_tools.assert_task_status,
            "transfer_to_human_agents": self.mock_tools.transfer_to_human_agents,
        }

        if action_name in action_map:
            tool_method = action_map[action_name]
            # Call the tool method with parameters
            if parameters:
                return tool_method(**parameters)
            else:
                return tool_method()
        else:
            return {"error": f"Unknown action: {action_name}"}

    @property
    def observation_space(self):
        """Return the observation space"""
        return {}

    @property
    def action_space(self):
        """Return the action space"""
        return {}

    def render(self, mode="human"):
        """Render the environment"""
        pass

    def close(self):
        """Close the environment"""
        pass
