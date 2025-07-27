"""
Pytest configuration file for Eval Protocol tests.
"""

import sys
from typing import Any, Dict, List, Optional

import pytest

from eval_protocol.models import EvaluateResult, MetricResult

# Check if e2b is available and skip related tests if not
try:
    import e2b

    HAS_E2B = True
except ImportError:
    HAS_E2B = False

# Mark to skip tests requiring e2b
skip_e2b = pytest.mark.skipif(not HAS_E2B, reason="e2b module not installed")


@pytest.fixture
def sample_messages():
    """Sample conversation messages."""
    return [
        {"role": "user", "content": "What is the weather like today?"},
        {
            "role": "assistant",
            "content": "I don't have real-time weather data. You should check a weather service.",
        },
    ]


@pytest.fixture
def sample_ground_truth_messages(sample_messages):  # Renamed fixture
    """Sample ground truth messages (e.g., user context or expected full conversation)."""
    return [sample_messages[0]]  # Keeping the same logic for now, assuming it represents context


@pytest.fixture
def sample_reward_output():
    """Sample reward output structure."""
    metrics = {
        "helpfulness": MetricResult(score=0.7, reason="Response acknowledges limitations", success=True),
        "accuracy": MetricResult(
            score=0.8,
            reason="Response correctly states lack of access to weather data",
            success=True,
        ),
    }
    return EvaluateResult(score=0.75, reason="Overall assessment", metrics=metrics)


@pytest.fixture
def sample_function_call_schema():
    """Sample function call schema for testing."""
    return {
        "name": "get_weather",
        "arguments": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
    }
