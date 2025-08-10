import sys
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
