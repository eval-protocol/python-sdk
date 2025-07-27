"""
Pytest configuration for Taxi MCP tests.

This configuration ensures:
1. Proper cleanup of environment variables
2. Server process management
3. Isolation between test runs
"""

import os

import pytest


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up eval-protocol environment variables before and after each test."""
    # Environment variables that might affect eval-protocol behavior
    env_vars_to_clean = ["EP_PLAYBACK_FILE", "EP_FORCE_RECORD", "PORT"]

    # Store original values
    original_values = {}
    for var in env_vars_to_clean:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (may take more than 30 seconds)")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "record_replay: marks tests that use record/replay functionality")
