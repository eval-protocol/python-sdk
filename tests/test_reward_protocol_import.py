"""Test that eval_protocol imports work correctly and provide the same functionality as eval_protocol."""

import importlib
import sys
from unittest.mock import patch

import pytest


class TestRewardProtocolImports:
    """Test that eval_protocol provides the same functionality as eval_protocol."""

    def test_basic_imports(self):
        """Test that both packages can be imported successfully."""
        import eval_protocol

        # Both should be importable
        assert eval_protocol is not None
        assert eval_protocol is not None

    def test_version_consistency(self):
        """Test that both packages have the same version."""
        import eval_protocol

        assert hasattr(eval_protocol, "__version__")
        assert hasattr(eval_protocol, "__version__")
        assert eval_protocol.__version__ == eval_protocol.__version__

    def test_all_exports_consistency(self):
        """Test that both packages export the same __all__ list."""
        import eval_protocol

        assert hasattr(eval_protocol, "__all__")
        assert hasattr(eval_protocol, "__all__")
        assert eval_protocol.__all__ == eval_protocol.__all__

    def test_core_classes_available(self):
        """Test that core classes are available through both imports."""
        from eval_protocol import (
            EvaluateResult,
            EvaluateResult as RPEvaluateResult,
            Message,
            Message as RPMessage,
            MetricResult,
            MetricResult as RPMetricResult,
            RewardFunction,
            RewardFunction as RPRewardFunction,
        )

        # Classes should be the same
        assert RewardFunction is RPRewardFunction
        assert Message is RPMessage
        assert MetricResult is RPMetricResult
        assert EvaluateResult is RPEvaluateResult

    def test_functions_available(self):
        """Test that core functions are available through both imports."""
        from eval_protocol import (
            load_jsonl,
            load_jsonl as rp_load_jsonl,
            make,
            make as rp_make,
            reward_function,
            reward_function as rp_reward_function,
            rollout,
            rollout as rp_rollout,
            test_mcp,
            test_mcp as rp_test_mcp,
        )

        # Functions should be the same
        assert reward_function is rp_reward_function
        assert load_jsonl is rp_load_jsonl
        assert make is rp_make
        assert rollout is rp_rollout
        assert test_mcp is rp_test_mcp

    def test_submodules_available(self):
        """Test that submodules are available through both imports."""
        import eval_protocol

        # Test a few key submodules
        submodules_to_test = ["models", "auth", "config", "rewards", "mcp"]

        for submodule in submodules_to_test:
            assert hasattr(eval_protocol, submodule)
            assert hasattr(eval_protocol, submodule)
            # The submodules should be the same object
            assert getattr(eval_protocol, submodule) is getattr(eval_protocol, submodule)

    def test_star_import_works(self):
        """Test that star imports work for both packages."""
        # This needs to be done in separate namespaces to avoid conflicts

        # Test eval_protocol star import
        rk_globals = {}
        exec("from eval_protocol import *", rk_globals)

        # Test eval_protocol star import
        rp_globals = {}
        exec("from eval_protocol import *", rp_globals)

        # Both should have the same set of imported names (minus built-ins)
        rk_names = {k for k in rk_globals.keys() if not k.startswith("__")}
        rp_names = {k for k in rp_globals.keys() if not k.startswith("__")}

        assert rk_names == rp_names

        # Test that key items are available
        expected_items = ["RewardFunction", "Message", "reward_function", "load_jsonl"]
        for item in expected_items:
            assert item in rk_names
            assert item in rp_names

    def test_reward_function_decorator_works(self):
        """Test that the @reward_function decorator works through both imports."""
        from eval_protocol import (
            EvaluateResult,
            reward_function as rk_reward_function,
            reward_function as rp_reward_function,
        )

        # Create a simple reward function using eval_protocol
        @rk_reward_function
        def test_reward_rk(response: str, **kwargs) -> EvaluateResult:
            score = len(response) / 10.0
            return EvaluateResult(
                score=score,
                reason=f"Score based on response length: {len(response)} characters",
                is_score_valid=True,
            )

        # Create the same reward function using eval_protocol
        @rp_reward_function
        def test_reward_rp(response: str, **kwargs) -> EvaluateResult:
            score = len(response) / 10.0
            return EvaluateResult(
                score=score,
                reason=f"Score based on response length: {len(response)} characters",
                is_score_valid=True,
            )

        # Both should work the same way
        test_input = "Hello, world!"
        result_rk = test_reward_rk(test_input)
        result_rp = test_reward_rp(test_input)

        # Both should return EvaluateResult objects with the same score
        assert isinstance(result_rk, EvaluateResult)
        assert isinstance(result_rp, EvaluateResult)
        assert result_rk.score == result_rp.score
        assert result_rk.score == len(test_input) / 10.0

    def test_message_class_works(self):
        """Test that Message class works through both imports."""
        from eval_protocol import Message as RKMessage, Message as RPMessage

        # They should be the same class
        assert RKMessage is RPMessage

        # Test creating instances
        msg_data = {"role": "user", "content": "Hello"}
        rk_msg = RKMessage(**msg_data)
        rp_msg = RPMessage(**msg_data)

        assert rk_msg.role == rp_msg.role
        assert rk_msg.content == rp_msg.content

    def test_console_scripts_in_setup(self):
        """Test that console scripts are defined in setup.py."""
        import os

        # Read setup.py content directly to avoid running it
        setup_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
        with open(setup_path, "r") as f:
            setup_content = f.read()

        # Check for console scripts in the file content
        expected_scripts = [
            'fireworks-reward = "eval_protocol.cli:main"',
            'eval-protocol = "eval_protocol.cli:main"',
        ]

        for script in expected_scripts:
            assert script in setup_content, f"Console script '{script}' not found in pyproject.toml"

    def test_package_structure_in_setup(self):
        """Test that both packages are included in setup.py."""
        from setuptools import find_packages

        packages = find_packages(include=["eval_protocol*"])

        # Should include main package
        assert "eval_protocol" in packages

        # Should include subpackages
        assert any(pkg.startswith("eval_protocol.") for pkg in packages)

    def test_deep_import_consistency(self):
        """Test that deep imports work consistently."""
        try:
            # Test importing from submodules
            from eval_protocol.models import Message as RKMessage, Message as RPMessage

            # Should be the same class
            assert RKMessage is RPMessage
        except ImportError:
            # If submodule imports don't work, that's expected in some install scenarios
            # Just verify the star import works
            from eval_protocol import Message as RKMessage, Message as RPMessage

            assert RKMessage is RPMessage

        try:
            # Test another submodule - use a function that actually exists
            from eval_protocol.auth import (
                get_fireworks_account_id,
                get_fireworks_account_id as rp_get_fireworks_account_id,
            )

            assert get_fireworks_account_id is rp_get_fireworks_account_id
        except ImportError:
            # If submodule imports don't work, verify through star import
            from eval_protocol import auth as rk_auth, auth as rp_auth

            assert rk_auth is rp_auth


class TestRewardProtocolFunctionality:
    """Test that eval_protocol functionality works correctly."""

    def test_reward_function_creation(self):
        """Test creating reward functions with eval_protocol."""
        from eval_protocol import EvaluateResult, reward_function

        @reward_function
        def simple_reward(response: str, **kwargs) -> EvaluateResult:
            """Simple reward based on response length."""
            score = float(len(response))
            return EvaluateResult(
                score=score,
                reason=f"Score based on response length: {len(response)} characters",
                is_score_valid=True,
            )

        # Test the reward function
        result = simple_reward("Hello")
        assert isinstance(result, EvaluateResult)
        assert result.score == 5.0
        assert result.is_score_valid is True
        assert "5 characters" in result.reason

        # Test that the function is callable (the decorator returns a callable)
        assert callable(simple_reward)

    def test_message_creation(self):
        """Test creating Message objects with eval_protocol."""
        from eval_protocol import Message

        msg = Message(role="user", content="Test message")
        assert msg.role == "user"
        assert msg.content == "Test message"

    def test_utility_functions(self):
        """Test that utility functions work through eval_protocol."""
        from eval_protocol import create_llm_resource, load_jsonl

        # These should be callable
        assert callable(load_jsonl)
        assert callable(create_llm_resource)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
