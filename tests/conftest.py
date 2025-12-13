import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import _HAS_E2B to create skip decorator
try:
    from eval_protocol.rewards.code_execution import _HAS_E2B
except ImportError:
    _HAS_E2B = False

# Decorator to skip E2B tests when E2B is not available
skip_e2b = pytest.mark.skipif(not _HAS_E2B, reason="E2B not installed")


# ============================================================================
# Test isolation for TinyDB storage
# ============================================================================
# Each test session gets an isolated .eval_protocol directory to prevent
# concurrent test workers from corrupting the shared logs.json file.
# This is especially important in CI where pytest-xdist runs tests in parallel.


@pytest.fixture(scope="session", autouse=True)
def isolated_eval_protocol_dir(tmp_path_factory):
    """
    Create an isolated .eval_protocol directory for the test session.

    This prevents concurrent test workers from corrupting the shared
    ~/.eval_protocol/logs.json file when using TinyDB storage.
    """
    # Create a unique temp directory for this test session/worker
    isolated_dir = tmp_path_factory.mktemp("eval_protocol")

    # Monkeypatch the find_eval_protocol_dir function to return our isolated dir
    import eval_protocol.directory_utils as dir_utils

    original_find_eval_protocol_dir = dir_utils.find_eval_protocol_dir

    def isolated_find_eval_protocol_dir() -> str:
        os.makedirs(str(isolated_dir), exist_ok=True)
        return str(isolated_dir)

    dir_utils.find_eval_protocol_dir = isolated_find_eval_protocol_dir

    yield isolated_dir

    # Restore original function after tests
    dir_utils.find_eval_protocol_dir = original_find_eval_protocol_dir
