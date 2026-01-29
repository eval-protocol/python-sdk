"""
Pytest configuration for solver_judge tests.
"""

import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import from main.py
solver_judge_dir = Path(__file__).parent.parent
sys.path.insert(0, str(solver_judge_dir))
