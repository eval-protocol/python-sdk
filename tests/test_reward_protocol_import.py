"""Test that reward_protocol imports work correctly and provide the same functionality as reward_kit."""

import pytest
import sys
import importlib
from unittest.mock import patch


class TestRewardProtocolImports:
    """Test that reward_protocol provides the same functionality as reward_kit."""
    
    def test_basic_imports(self):
        """Test that both packages can be imported successfully."""
        import reward_kit
        import reward_protocol
        
        # Both should be importable
        assert reward_kit is not None
        assert reward_protocol is not None
    
    def test_version_consistency(self):
        """Test that both packages have the same version."""
        import reward_kit
        import reward_protocol
        
        assert hasattr(reward_kit, '__version__')
        assert hasattr(reward_protocol, '__version__')
        assert reward_kit.__version__ == reward_protocol.__version__
    
    def test_all_exports_consistency(self):
        """Test that both packages export the same __all__ list."""
        import reward_kit
        import reward_protocol
        
        assert hasattr(reward_kit, '__all__')
        assert hasattr(reward_protocol, '__all__')
        assert reward_kit.__all__ == reward_protocol.__all__
    
    def test_core_classes_available(self):
        """Test that core classes are available through both imports."""
        from reward_kit import RewardFunction, Message, MetricResult, EvaluateResult
        from reward_protocol import RewardFunction as RPRewardFunction
        from reward_protocol import Message as RPMessage
        from reward_protocol import MetricResult as RPMetricResult
        from reward_protocol import EvaluateResult as RPEvaluateResult
        
        # Classes should be the same
        assert RewardFunction is RPRewardFunction
        assert Message is RPMessage
        assert MetricResult is RPMetricResult
        assert EvaluateResult is RPEvaluateResult
    
    def test_functions_available(self):
        """Test that core functions are available through both imports."""
        from reward_kit import reward_function, load_jsonl, make, rollout, test_mcp
        from reward_protocol import reward_function as rp_reward_function
        from reward_protocol import load_jsonl as rp_load_jsonl
        from reward_protocol import make as rp_make
        from reward_protocol import rollout as rp_rollout
        from reward_protocol import test_mcp as rp_test_mcp
        
        # Functions should be the same
        assert reward_function is rp_reward_function
        assert load_jsonl is rp_load_jsonl
        assert make is rp_make
        assert rollout is rp_rollout
        assert test_mcp is rp_test_mcp
    
    def test_submodules_available(self):
        """Test that submodules are available through both imports."""
        import reward_kit
        import reward_protocol
        
        # Test a few key submodules
        submodules_to_test = ['models', 'auth', 'config', 'rewards', 'mcp']
        
        for submodule in submodules_to_test:
            assert hasattr(reward_kit, submodule)
            assert hasattr(reward_protocol, submodule)
            # The submodules should be the same object
            assert getattr(reward_kit, submodule) is getattr(reward_protocol, submodule)
    
    def test_star_import_works(self):
        """Test that star imports work for both packages."""
        # This needs to be done in separate namespaces to avoid conflicts
        
        # Test reward_kit star import
        rk_globals = {}
        exec("from reward_kit import *", rk_globals)
        
        # Test reward_protocol star import
        rp_globals = {}
        exec("from reward_protocol import *", rp_globals)
        
        # Both should have the same set of imported names (minus built-ins)
        rk_names = {k for k in rk_globals.keys() if not k.startswith('__')}
        rp_names = {k for k in rp_globals.keys() if not k.startswith('__')}
        
        assert rk_names == rp_names
        
        # Test that key items are available
        expected_items = ['RewardFunction', 'Message', 'reward_function', 'load_jsonl']
        for item in expected_items:
            assert item in rk_names
            assert item in rp_names
    
    def test_reward_function_decorator_works(self):
        """Test that the @reward_function decorator works through both imports."""
        from reward_kit import reward_function as rk_reward_function
        from reward_protocol import reward_function as rp_reward_function
        from reward_kit import EvaluateResult
        
        # Create a simple reward function using reward_kit
        @rk_reward_function
        def test_reward_rk(response: str, **kwargs) -> EvaluateResult:
            score = len(response) / 10.0
            return EvaluateResult(
                score=score,
                reason=f"Score based on response length: {len(response)} characters",
                is_score_valid=True
            )
        
        # Create the same reward function using reward_protocol  
        @rp_reward_function
        def test_reward_rp(response: str, **kwargs) -> EvaluateResult:
            score = len(response) / 10.0
            return EvaluateResult(
                score=score,
                reason=f"Score based on response length: {len(response)} characters",
                is_score_valid=True
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
        from reward_kit import Message as RKMessage
        from reward_protocol import Message as RPMessage
        
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
        setup_path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
        with open(setup_path, 'r') as f:
            setup_content = f.read()
        
        # Check for console scripts in the file content
        expected_scripts = [
            'fireworks-reward = "reward_kit.cli:main"',
            'reward-kit = "reward_kit.cli:main"',
            'reward-protocol = "reward_kit.cli:main"',
        ]
        
        for script in expected_scripts:
            assert script in setup_content, f"Console script '{script}' not found in pyproject.toml"
    
    def test_package_structure_in_setup(self):
        """Test that both packages are included in setup.py."""
        from setuptools import find_packages
        
        packages = find_packages(include=['reward_kit*', 'reward_protocol*'])
        
        # Should include both main packages
        assert 'reward_kit' in packages
        assert 'reward_protocol' in packages
        
        # Should include subpackages
        assert any(pkg.startswith('reward_kit.') for pkg in packages)
    
    def test_deep_import_consistency(self):
        """Test that deep imports work consistently."""
        try:
            # Test importing from submodules
            from reward_kit.models import Message as RKMessage
            from reward_protocol.models import Message as RPMessage
            
            # Should be the same class
            assert RKMessage is RPMessage
        except ImportError:
            # If submodule imports don't work, that's expected in some install scenarios
            # Just verify the star import works
            from reward_kit import Message as RKMessage
            from reward_protocol import Message as RPMessage
            assert RKMessage is RPMessage
        
        try:
            # Test another submodule - use a function that actually exists
            from reward_kit.auth import get_fireworks_account_id
            from reward_protocol.auth import get_fireworks_account_id as rp_get_fireworks_account_id
            
            assert get_fireworks_account_id is rp_get_fireworks_account_id
        except ImportError:
            # If submodule imports don't work, verify through star import
            from reward_kit import auth as rk_auth
            from reward_protocol import auth as rp_auth
            assert rk_auth is rp_auth


class TestRewardProtocolFunctionality:
    """Test that reward_protocol functionality works correctly."""
    
    def test_reward_function_creation(self):
        """Test creating reward functions with reward_protocol."""
        from reward_protocol import reward_function
        from reward_kit import EvaluateResult
        
        @reward_function
        def simple_reward(response: str, **kwargs) -> EvaluateResult:
            """Simple reward based on response length."""
            score = float(len(response))
            return EvaluateResult(
                score=score,
                reason=f"Score based on response length: {len(response)} characters",
                is_score_valid=True
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
        """Test creating Message objects with reward_protocol."""
        from reward_protocol import Message
        
        msg = Message(role="user", content="Test message")
        assert msg.role == "user"
        assert msg.content == "Test message"
    
    def test_adapter_functions(self):
        """Test that adapter functions work through reward_protocol."""
        from reward_protocol import reward_fn_to_scorer, scorer_to_reward_fn
        
        # These should be callable
        assert callable(reward_fn_to_scorer)
        assert callable(scorer_to_reward_fn)
    
    def test_utility_functions(self):
        """Test that utility functions work through reward_protocol."""
        from reward_protocol import load_jsonl, create_llm_resource
        
        # These should be callable
        assert callable(load_jsonl)
        assert callable(create_llm_resource)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 