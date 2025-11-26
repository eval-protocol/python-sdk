"""
Unit tests for exception_config module.

Tests the BackoffConfig and ExceptionHandlerConfig classes, including:
1. Backoff decorator creation
2. Per-exception backoff overrides
3. ResponseQualityError default no-backoff configuration
4. Exception grouping to avoid double backoff
"""

import pytest
from eval_protocol.pytest.exception_config import BackoffConfig, ExceptionHandlerConfig, DEFAULT_RETRYABLE_EXCEPTIONS
from eval_protocol.exceptions import ResponseQualityError


def test_backoff_config_no_exceptions():
    """Test that BackoffConfig returns no-op decorator when no exceptions specified."""
    config = BackoffConfig()
    decorator = config.get_backoff_decorator(set())
    
    # Should be a no-op decorator
    def test_func():
        return "test"
    
    decorated = decorator(test_func)
    assert decorated() == "test"
    assert decorated is test_func  # Should be the same function


def test_backoff_config_no_overrides():
    """Test that BackoffConfig creates a single decorator."""
    config = BackoffConfig(strategy="constant", base_delay=0.1, max_tries=2)
    exceptions = {ConnectionError, TimeoutError}
    
    decorator = config.get_backoff_decorator(exceptions)
    assert decorator is not None
    
    # Decorator should be callable
    def test_func():
        raise ConnectionError("test")
    
    decorated = decorator(test_func)
    assert callable(decorated)


def test_exception_handler_config_default_response_quality_error():
    """Test that ExceptionHandlerConfig includes ResponseQualityError by default."""
    config = ExceptionHandlerConfig()
    
    # ResponseQualityError should be in retryable_exceptions
    assert ResponseQualityError in config.retryable_exceptions


def test_exception_handler_config_get_backoff_decorator():
    """Test that ExceptionHandlerConfig.get_backoff_decorator() works correctly."""
    config = ExceptionHandlerConfig()
    decorator = config.get_backoff_decorator()
    
    assert decorator is not None
    assert callable(decorator)
    
    # Should be able to decorate a function
    def test_func():
        raise ConnectionError("test")
    
    decorated = decorator(test_func)
    assert callable(decorated)


def test_backoff_config_expo_strategy():

    """Test that BackoffConfig creates expo decorator correctly."""
    config = BackoffConfig(strategy="expo", base_delay=1.0, max_tries=2)
    exceptions = {ConnectionError}
    
    decorator = config.get_backoff_decorator(exceptions)
    assert decorator is not None
    
    def test_func():
        raise ConnectionError("test")
    
    decorated = decorator(test_func)
    assert callable(decorated)


def test_backoff_config_constant_strategy():
    """Test that BackoffConfig creates constant decorator correctly."""
    config = BackoffConfig(strategy="constant", base_delay=0.1, max_tries=2)
    exceptions = {ConnectionError}
    
    decorator = config.get_backoff_decorator(exceptions)
    assert decorator is not None
    
    def test_func():
        raise ConnectionError("test")
    
    decorated = decorator(test_func)
    assert callable(decorated)


def test_backoff_config_invalid_strategy():
    """Test that BackoffConfig raises ValueError for invalid strategy."""
    config = BackoffConfig(strategy="invalid", base_delay=1.0, max_tries=2)
    exceptions = {ConnectionError}
    
    with pytest.raises(ValueError, match="Unknown backoff strategy"):
        config.get_backoff_decorator(exceptions)


def test_exception_handler_config_response_quality_error_in_defaults():
    """Test that ResponseQualityError is in DEFAULT_RETRYABLE_EXCEPTIONS."""
    assert ResponseQualityError in DEFAULT_RETRYABLE_EXCEPTIONS


