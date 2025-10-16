"""
Quickstart modules for various evaluation scenarios.
"""

from eval_protocol.quickstart.aha_judge.llm_judge import aha_judge
from eval_protocol.utils.evaluation_row_utils import multi_turn_assistant_to_ground_truth, assistant_to_ground_truth

__all__ = ["aha_judge", "multi_turn_assistant_to_ground_truth", "assistant_to_ground_truth"]
